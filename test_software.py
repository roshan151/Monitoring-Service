#!/usr/bin/env python3
"""Test the robot's software end to end from the Mac: nothing moves, robot-voice keeps running.

test_hardware.py tests the robot as a machine (moves, follow, e-stop). This tests the
software it runs on: the Vision-Microservice on this Mac, called directly the way the
robot calls it, then the robot's own code on the Pi over SSH, fed made-up situations
(a flat floor, a box, a drop, a stair) with the robot's real robot.yaml, and the robot
calling this Mac with its own vision client. It replaces Robot-RPi-Core's old tests/.
Each run is saved to test-runs/<time>.json and as a plain-English page,
test-runs/<time>.html, opened in the browser.

    python test_software.py              pick a mode from a menu
    python test_software.py all          vision, then robot
    python test_software.py vision       Vision-Microservice on this Mac: up, quick, each endpoint answers right (no Pi)
    python test_software.py robot        the robot's code on the Pi: installed, same as this Mac, decisions right, reaches the Mac

Options: --host user@pi and --repo (shared with calibrate.py in .calibrate.json),
--service http://127.0.0.1:8080 (this Mac's Vision-Microservice), --domain 42 (ROS_DOMAIN_ID)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path
from urllib import error, request

from calibrate import mode_service as _service_health, yes
from test_hardware import HERE, Results, _get, connect

MODES = ["vision", "robot"]
INFO = {
    "vision": ("Vision service on this Mac", "The program on this Mac that sees for the robot: objects, faces, "
               "depth, plants and the room map. Is it running, quick, and answering correctly?"),
    "robot": ("Robot's software (nothing moves)", "The robot's own code, run on the robot with made-up situations: "
              "is it installed and up to date, does it make the right decisions, and can it reach this Mac?"),
}
ROBOT_REPO = Path(os.environ.get("ROBOT_REPO") or HERE.parent / "Robot-RPi-Core")
ROBOT_YAML = "ros2_ws/src/robot_bringup/config/robot.yaml"

# --------------------------------------------------------------------------- the Pi side (sent over SSH)

PI_SIDE = r'''
import hashlib, importlib, json, math, os, pkgutil, re, subprocess, sys, time
from pathlib import Path

def say(**m):
    print(json.dumps(m, default=str), flush=True)

REPO = Path(os.getcwd())
sys.path.insert(0, str(REPO))

def yaml_params():
    import yaml
    return yaml.safe_load((REPO / "ros2_ws/src/robot_bringup/config/robot.yaml").read_text())

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

# Each check: (plain name, function returning (ok, detail)). One crashing never stops the rest.
CHECKS = []
def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap

@check("Robot code installed and loads")
def _():
    import robot_core
    skip = ("run", "calibrate_pi", "__main__")
    bad, n = [], 0
    for m in pkgutil.walk_packages(robot_core.__path__, "robot_core."):
        if m.name.rsplit(".", 1)[-1] in skip:
            continue
        n += 1
        try:
            importlib.import_module(m.name)
        except Exception as e:
            bad.append(f"{m.name} ({type(e).__name__}: {e})"[:120])
    return not bad, f"{n} parts loaded" if not bad else f"{len(bad)} of {n} failed: " + "; ".join(bad[:3])

@check("Robot code keeps ROS out (works on its own)")
def _():
    forbidden = ("rclpy", "rosidl", "ament", "robot_interfaces", "std_srvs")
    bad = [f"{p.relative_to(REPO)}:{i}" for p in sorted((REPO / "robot_core").rglob("*.py"))
           for i, l in enumerate(p.read_text().splitlines(), 1)
           if l.strip().startswith(("import ", "from ")) and any(f in l for f in forbidden)]
    no_init = [str(d.relative_to(REPO)) for d in (REPO / "robot_core").rglob("*")
               if d.is_dir() and d.name != "__pycache__" and not (d / "__init__.py").exists()]
    ok = not bad and not no_init
    return ok, "clean" if ok else f"ROS imports at {bad[:3]}; folders missing __init__.py {no_init[:3]}"

@check("Settings file (robot.yaml) readable")
def _():
    y = yaml_params()
    need = ("/drivetrain", "/voice", "/explorer", "/display")
    missing = [s for s in need if s not in y]
    return not missing, f"sections {', '.join(need)} present" if not missing else f"missing {missing}"

@check("Voice assistant's tools ready")
def _():
    from robot_core.live.tools import declarations
    d = {x.name: x for x in declarations()}
    need = {"drive", "turn", "stop", "run_task", "shutdown"}
    stop_ok = "stop" in d and not getattr(d["stop"].parameters, "required", [])
    ok = need <= set(d) and stop_ok
    return ok, f"{len(d)} tools: {', '.join(sorted(d))}" if ok else \
        f"missing {sorted(need - set(d))}" + ("" if stop_ok else "; 'stop' must take no arguments")

def floor(height, x0=0.15, x1=1.0, step=0.025, ys=(-0.1, -0.03, 0.03, 0.1)):
    n = int(round((x1 - x0) / step))
    return [(x0 + i * step, y, height(x0 + i * step)) for i in range(n + 1) for y in ys]

@check("Terrain check: drives over flat floor and a sill, stops at a box and stairs")
def _():
    from robot_core import terrain
    p = terrain.params_from(yaml_params()["/explorer"]["ros__parameters"])
    go, stop = (terrain.FLAT, terrain.CLIMBABLE), (terrain.BLOCKED,)
    cases = [("flat floor", floor(lambda x: 0.008 * math.sin(40 * x)), go),
             ("2 cm door sill", floor(lambda x: 0.02 if 0.40 <= x < 0.44 else 0.0), go),
             ("12 cm box", floor(lambda x: 0.12 if x >= 0.40 else 0.0, x1=0.45), stop),
             ("stairs up", floor(lambda x: 0.0 if x < 0.35 else 0.02 if x < 0.45 else 0.17), stop)]
    got = [(n, terrain.classify(pts, p, 0.6).kind, want) for n, pts, want in cases]
    wrong = [f"{n} read {k} (should be {' or '.join(w)})" for n, k, w in got if k not in w]
    return not wrong, "; ".join(f"{n}: {k}" for n, k, _ in got) if not wrong else "; ".join(wrong)

@check("Path planning: around a box, straight when clear, stops at a wall")
def _():
    from robot_core import local_plan as lp, terrain
    r = lp.radius(terrain.params_from(yaml_params()["/explorer"]["ros__parameters"]))
    box = [(1.0, y / 20.0) for y in range(-4, 5)]
    path = lp.plan((0.0, 0.0), (2.0, 0.0), box, r)
    round_ok = bool(path) and len(path) > 2 and all(lp.line_clear(a, b, box, r * 0.95) for a, b in zip(path, path[1:]))
    straight = lp.plan((0.0, 0.0), (2.0, 0.0), [], r) == [(0.0, 0.0), (2.0, 0.0)]
    wall = lp.plan((0.0, 0.0), (2.0, 0.0), [(1.0, y / 20.0) for y in range(-40, 41)], r) is None
    ok = round_ok and straight and wall
    return ok, f"clearance {r * 100:.0f} cm; round the box in {len(path or []) - 1} legs" if ok else \
        f"round the box {round_ok}, straight {straight}, gives up at a wall {wall}"

@check("Edge guard: brakes at a drop, ignores one bad reading")
def _():
    from robot_core.drop_guard import DropGuard
    g = DropGuard()
    for i in range(g.learn_n):
        g.update(33 + (i % 3) - 1, forward=False)
    seq = [g.update(cm, True) for cm in (34, 60, 33, 60, 61)]
    ok = g.armed and seq[:4] == [None] * 4 and bool(seq[4])
    return ok, f"learned floor {g.floor_cm} cm; then: {seq[4]}" if ok else f"armed {g.armed}, answers {seq}"

@check("Speed control: gentle start, straight, smooth stop")
def _():
    from robot_core.velocity import WheelSpeedController, params_from
    d = yaml_params()["/drivetrain"]["ros__parameters"]
    vp = params_from(d)
    c = WheelSpeedController(float(d.get("wheelbase_m") or 0.22), vp)
    l1, r1 = c.step(0.2, 0.0, None, 0.1)
    gentle = c.v <= vp.accel_mps2 * 0.1 + 1e-9
    for _ in range(20):
        l, r = c.step(0.2, 0.0, None, 0.1)
    straight = l == r > 0
    stop = [c.step(0.0, 0.0, None, 0.1) for _ in range(15)]
    smooth = stop[0][0] > 0 and stop[-1] == (0, 0)
    c.reset()
    for _ in range(20):
        tl, tr = c.step(0.0, 0.8, None, 0.1)
    spin = tl < 0 < tr
    ok = gentle and straight and smooth and spin
    return ok, f"cruise PWM {l}/{r}, stops in {next(i for i, s in enumerate(stop) if s == (0, 0)) + 1} steps" if ok else \
        f"gentle {gentle}, straight {straight}, smooth stop {smooth}, turns on the spot {spin}"

@check("Position tracking: 1 m straight and a quarter turn")
def _():
    from robot_core.odometry import DiffDriveOdometry
    tpm, wb = 3850.0, 0.22
    def run(samples):
        o, st = DiffDriveOdometry(tpm, wb), None
        for i, (l, r) in enumerate(samples):
            st = o.update(r, l, i * 100, i * 0.1)
        return st.pose
    p1 = run([(i * 385, i * 385) for i in range(11)])
    arc = wb / 2 * math.pi / 2 * tpm
    p2 = run([(-arc * k / 10, arc * k / 10) for k in range(11)])
    ok = abs(p1.x - 1.0) < 1e-3 and abs(p1.y) < 1e-3 and abs(p2.theta - math.pi / 2) < 1e-3
    return ok, f"{p1.x:.2f} m and {math.degrees(p2.theta):.0f} deg" if ok else f"read {p1.x:.3f} m, {math.degrees(p2.theta):.1f} deg"

def cmd_hello(a):
    y = REPO / "ros2_ws/src/robot_bringup/config/robot.yaml"
    return {"host": sh("hostname"), "repo": str(REPO), "git": sh("git rev-parse --short HEAD"),
            "dirty": ", ".join(l.strip() for l in sh("git status --porcelain -- robot_core ros2_ws firmware "
                     "':!ros2_ws/src/robot_bringup/config/robot.yaml' | head -5").splitlines()),
            "yaml_md5": hashlib.md5(y.read_bytes()).hexdigest() if y.exists() else None}

def cmd_software(a):
    out = []
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"crashed: {type(e).__name__}: {e}"[:200]
        out.append([name, bool(ok), detail])
        say(progress=f"{'PASS' if ok else 'FAIL'} {name}")
    return out

def cmd_vision(a):
    """The robot's own VisionClient against this Mac, as follow/explore/enroll use it."""
    from urllib.parse import urlparse
    from robot_core import settings
    from robot_core.vision_client import VisionClient
    url = settings.VISION_SERVICE_BASE_URL
    u = urlparse(url)
    if u.hostname in ("127.0.0.1", "localhost"):          # not set for this login: the Mac is the SSH client
        url = f"http://{(os.environ.get('SSH_CLIENT') or '').split(' ')[0]}:{a.get('port') or u.port or 8080}"
    out = {"url": url}
    vc = VisionClient(url, timeout_s=10)
    t0 = time.monotonic()
    try:
        out["health"] = vc.health().get("status")
        out["health_ms"] = round((time.monotonic() - t0) * 1000)
        import cv2, numpy as np
        pic = np.full((480, 640, 3), 128, np.uint8)
        cv2.rectangle(pic, (240, 160), (400, 360), (40, 40, 40), -1)
        jpg = cv2.imencode(".jpg", pic)[1].tobytes()
        t0 = time.monotonic()
        d = vc.detect(jpg, seq=11)
        out["detect_ms"] = round((time.monotonic() - t0) * 1000)
        out["detect_ok"] = d.get("seq") == 11 and d.get("image_size") == [640, 480]
        out["face_none"] = vc.face_embed(jpg, seq=12) is None
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    return out

CMDS = {k[4:]: v for k, v in globals().items() if k.startswith("cmd_")}
say(ready=True)
for line in sys.stdin:
    try:
        req = json.loads(line)
    except ValueError:
        continue
    if req.get("cmd") == "quit":
        say(ok=True, result=None)
        break
    try:
        say(ok=True, result=CMDS[req["cmd"]](req.get("args") or {}))
    except Exception as e:
        say(ok=False, error=f"{type(e).__name__}: {e}")
'''


# --------------------------------------------------------------------------- helpers

def _post(url: str, fields: dict, image: bytes | None = None, timeout: float = 20):
    """multipart/form-data POST, as the Pi sends it: (status, json body) or (None, error text)."""
    b = uuid.uuid4().hex
    body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                    for k, v in fields.items())
    if image is not None:
        body += (f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="f.jpg"\r\n'
                 f"Content-Type: image/jpeg\r\n\r\n").encode() + image + b"\r\n"
    body += f"--{b}--\r\n".encode()
    req = request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        with request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except error.HTTPError as ex:
        try:
            return ex.code, json.loads(ex.read() or b"{}")
        except ValueError:
            return ex.code, {}
    except Exception as ex:                                 # noqa: BLE001
        return None, f"{type(ex).__name__}: {ex}"


def _local(*args: str) -> str:
    r = subprocess.run(["git", "-C", str(ROBOT_REPO), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


# --------------------------------------------------------------------------- modes

def mode_vision(pi, res: Results, a) -> None:
    url = a.service
    n = len(res.checks)
    _service_health(url, res)                               # calibrate.py's: service up, detect latency
    if not res.checks[n]["ok"]:
        return
    m = _get(url + "/v1/models") or {}
    res.check("models loaded", bool(m.get("versions")),
              ", ".join(f"{k} {v}" for k, v in (m.get("versions") or {}).items()) or "no answer from /v1/models")
    try:
        import cv2
        import numpy as np
    except ImportError:
        res.check("vision answers", False, "OpenCV missing on this Mac: pip install -r requirements.txt")
        return
    pic = np.full((480, 640, 3), 128, np.uint8)             # a plain test picture: grey wall, dark box
    cv2.rectangle(pic, (240, 160), (400, 360), (40, 40, 40), -1)
    img = cv2.imencode(".jpg", pic)[1].tobytes()

    st, d = _post(url + "/v1/detect", {"seq": 7}, img)
    ok = st == 200 and d.get("seq") == 7 and d.get("image_size") == [640, 480] and isinstance(d.get("detections"), list)
    res.check("object detection answers correctly", ok,
              f"{len(d['detections'])} objects in the test picture, {d.get('inference_ms', 0):.0f} ms" if ok
              else f"HTTP {st}: {d}")
    st, d = _post(url + "/v1/depth", {"seq": 8, "points": "[[320,240],[100,100]]"}, img)
    ok = st == 200 and len(d.get("depths_m") or []) == 2
    res.check("depth answers correctly", ok, (f"2 readings, {'metres' if d.get('metric') else 'relative (not metres)'}, "
              f"{d.get('inference_ms', 0):.0f} ms") if ok else f"HTTP {st}: {d}")
    st, d = _post(url + "/v1/depth", {"seq": 9, "points": "not a list"}, img)
    res.check("bad request refused cleanly", st == 422, f"HTTP {st} (should be 422, not a crash)")
    st, d = _post(url + "/v1/faces/embed", {"seq": 10}, img)
    res.check("face check says 'no face' in a faceless picture", st == 404,
              f"HTTP {st}" + ("" if st == 404 else f": {d}"))
    h = _get(url + "/healthz") or {}
    p = _get(url + "/v1/plants")
    res.check("saved plants and faces readable", p is not None,
              f"{len(p.get('plants', []))} plants, people {h.get('gallery')}" if p is not None
              else "/v1/plants did not answer")
    res.check("room map readable", _get(url + "/v1/slam/state") is not None, "/v1/slam/state")
    res.check("queue not backed up", h.get("queue_depth", 99) <= 2, f"{h.get('queue_depth')} requests waiting")


def mode_robot(pi, res: Results, a) -> None:
    print("\n== robot: the robot's code on the Pi (nothing moves)")
    hello = res.data["hello"]
    mine = _local("rev-parse", "--short", "HEAD")
    if mine:
        res.check("Robot runs the same code as this Mac", hello["git"] == mine and not hello["dirty"],
                  f"robot {hello['git']}, Mac {mine}" + (f"; changed on the robot: {hello['dirty']}" if hello["dirty"] else "")
                  + ("" if hello["git"] == mine else " (git pull on the robot, or push from the Mac)"))
        y = ROBOT_REPO / ROBOT_YAML
        same = y.exists() and hashlib.md5(y.read_bytes()).hexdigest() == hello["yaml_md5"]
        res.check("Robot's settings match this Mac's robot.yaml", same,
                  "identical" if same else "different: copy it over (apply_calibration.py offers to)")
    for name, ok, detail in pi("software"):
        res.check(name, ok, detail)
    port = int(a.service.rsplit(":", 1)[-1]) if a.service.count(":") > 1 else 8080
    v = pi("vision", port=port)
    res.data["robot_vision"] = v
    if v.get("error") and v.get("health") is None:
        res.check("Robot reaches the vision service", False,
                  f"{v['url']}: {v['error']} (service running? same network? firewall?)")
        return
    res.check("Robot reaches the vision service", v.get("health") == "ok", f"{v['url']} in {v.get('health_ms')} ms")
    res.check("Robot's vision requests answered correctly", v.get("detect_ok") and v.get("face_none"),
              f"detect {v.get('detect_ms')} ms over Wi-Fi; no-face answered as 'no face'" if not v.get("error")
              else v["error"])


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", nargs="?", choices=MODES + ["all"])
    ap.add_argument("--host")
    ap.add_argument("--repo")
    ap.add_argument("--domain", type=int, default=42)
    ap.add_argument("--service", default=os.environ.get("VS_URL", "http://127.0.0.1:8080"))
    a = ap.parse_args()
    a.service = a.service.rstrip("/")
    mode = a.mode
    if not mode:
        print("Modes: " + ", ".join(["all"] + MODES))
        mode = input("Which? [all] ").strip() or "all"
        if mode not in MODES + ["all"]:
            raise SystemExit(f"unknown mode {mode}")
    modes = MODES if mode == "all" else [mode]
    res = Results(INFO, "Robot software test")
    pi = None
    try:
        if "vision" in modes:
            res.mode = "vision"
            mode_vision(None, res, a)
        if "robot" in modes:
            res.mode = "robot"
            if not yes("Is the Pi on and on the same network as this Mac?"):
                res.skipped["robot"] = "the robot was off: switch it on and run this again"
            else:
                pi, _, _ = connect(a, PI_SIDE)
                res.data["hello"] = pi("hello")
                print(f"Connected to {res.data['hello']['host']} ({res.data['hello']['repo']}).")
                try:
                    mode_robot(pi, res, a)
                except RuntimeError as e:
                    res.check("robot", False, str(e))
    except KeyboardInterrupt:
        print("\n   stopped")
    finally:
        if pi:
            pi.close()
        res.report()


if __name__ == "__main__":
    main()
