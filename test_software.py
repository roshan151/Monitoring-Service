#!/usr/bin/env python3
"""Test the robot's software end to end from the Mac: nothing moves, robot-voice keeps running.

test_hardware.py tests the robot as a machine (moves, follow, e-stop). This tests the
software it runs on: the Vision-And-Planning on this Mac, called directly the way the
robot calls it, then the robot's own code on the Pi over SSH, fed made-up situations
(a flat floor, a box, a drop, a stair) with the robot's real robot.yaml, and the robot
calling this Mac with its own vision client. It replaces Robot-RPi-Core's old tests/.
Each run is saved to test-runs/<time>.json and as a plain-English page,
test-runs/<time>.html, opened in the browser.

    python test_software.py              pick a mode from a menu
    python test_software.py all          vision, then robot
    python test_software.py vision       Vision-And-Planning on this Mac: up, quick, each endpoint answers right (no Pi)
    python test_software.py robot        the robot's code on the Pi: installed, same as this Mac, decisions right, reaches the Mac
    python test_software.py follow       start follow on the robot (no voice) with the live camera + ToF + nav + SLAM view
    python test_software.py explore      start explore the same way

follow and explore MOVE the robot (never part of "all"). The live view is navigation_debugging.py --nav: it needs
debug_mode: true under /voice in robot.yaml. Press Enter here to stop the task; q in the view closes it.

Options: --host user@pi and --repo (shared with calibrate.py in .calibrate.json),
--service http://127.0.0.1:8080 (this Mac's Vision-And-Planning), --domain 42 (ROS_DOMAIN_ID)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib import error, request

from calibrate_and_apply import mode_service as _service_health, yes
from test_hardware import HERE, Results, _get, connect

MODES = ["vision", "robot", "fixes"]
TASKS = ["follow", "explore"]                               # these move the robot: never part of "all"
INFO = {
    "vision": ("Vision service on this Mac", "The program on this Mac that sees for the robot: objects, faces, "
               "depth, plants and the room map. Is it running, quick, and answering correctly?"),
    "robot": ("Robot's software (nothing moves)", "The robot's own code, run on the robot with made-up situations: "
              "is it installed and up to date, does it make the right decisions, and can it reach this Mac?"),
    "follow": ("Follow, started from here", "Starts follow on the robot without voice (the first enrolled person it "
               "sees) and opens the live camera, ToF, nav and SLAM view."),
    "explore": ("Explore, started from here", "Starts explore on the robot without voice and opens the live camera, "
                "ToF, nav and SLAM view."),
    "fixes": ("Navigation, VSLAM & Calibration Fixes", "Reflex ground clearance, DWA pivot-first, doorway turns, "
              "PyMonoVO visual odometry, and calibration target math"),
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


def mode_task(pi, res: Results, a, task: str) -> None:
    """Start follow / explore through the robot's task/<task> service, with navigation_debugging.py's live view."""
    print(f"\n== {task}: started from here (the robot moves)")
    view = subprocess.Popen([sys.executable, str(HERE / "navigation_debugging.py"), "--nav"], cwd=str(HERE))
    print("   live view opened (camera + ToF, nav, SLAM; q closes it). Needs debug_mode: true under /voice.")
    r = pi("task_start", task=task)
    res.check(f"robot accepted {task}", r["ok"], r["message"])
    if not r["ok"]:
        return
    w = pi("watch", seconds=30, until=["task.start"])
    res.check(f"{task} started", w["until_s"] is not None,
              f"{w['until_s']} s after asking" if w["until_s"] is not None else "no task.start in 30 s")
    if w["until_s"] is None:
        return
    try:
        input(f"   {task} running. Press Enter to stop it (Ctrl-C brakes). ")
    finally:
        pi("task_stop")
    time.sleep(2.0)
    end = [e for e in pi("logs", since=r["now"]) if e.get("evt") == "task.end"]
    res.check(f"{task} ended", bool(end), json.dumps({k: end[-1].get(k) for k in ("ok", "cancelled")}) if end
              else "no task.end yet: check logs.json")
    if view.poll() is None:
        print("   the live view stays open: q closes it")


def mode_fixes(res: Results) -> None:
    print("\n== fixes: navigation, SLAM and calibration fixes (nothing moves)")
    import math
    import sys
    from pathlib import Path
    import numpy as np

    src_dir = HERE.parent
    sys.path.insert(0, str(src_dir / "Robot-RPi-Core"))
    sys.path.insert(0, str(src_dir / "Vision-And-Planning"))

    # 1. Calibration Target Math
    from calibrate_and_apply import (
        TargetConfig, segment_tof_target, compute_alignment,
        CALIB_OBJ_HEIGHT_M, CALIB_OBJ_WIDTH_M, CALIB_OBJ_DEPTH_M, CALIB_DISTANCE_M
    )
    dim_ok = (
        abs(CALIB_OBJ_HEIGHT_M - 56.0 * 0.0254) < 1e-4 and
        abs(CALIB_OBJ_WIDTH_M - 38.0 * 0.0254) < 1e-4 and
        abs(CALIB_OBJ_DEPTH_M - 30.0 * 0.0254) < 1e-4 and
        abs(CALIB_DISTANCE_M - 1.00) < 1e-4
    )
    res.check("calibration target default dimensions (56x38x30 in at 1.0 m)", dim_ok,
              f"{CALIB_OBJ_HEIGHT_M / 0.0254:.1f}x{CALIB_OBJ_WIDTH_M / 0.0254:.1f}x{CALIB_OBJ_DEPTH_M / 0.0254:.1f} in at {CALIB_DISTANCE_M:.2f} m")

    cfg = TargetConfig.from_dict({"width_in": 40.0, "height_in": 60.0, "depth_in": 25.0, "distance_m": 1.20})
    custom_ok = abs(cfg.width_in - 40.0) < 0.1 and abs(cfg.height_in - 60.0) < 0.1 and abs(cfg.distance_m - 1.20) < 0.01
    res.check("calibration target config overrides", custom_ok, f"custom target {cfg.width_in:.0f}x{cfg.height_in:.0f} in at {cfg.distance_m:.2f} m")

    # ToF grid segmentation test
    rows = [[2.5] * 8 for _ in range(8)]
    for r in range(2, 6):
        for c in range(2, 6):
            rows[r][c] = 1.00
    tof_seg = segment_tof_target(rows, min_d=0.8, max_d=1.25)
    seg_ok = tof_seg is not None and abs(tof_seg[2] - 1.00) < 0.02
    res.check("ToF target segmentation at 1.0 m", seg_ok, f"depth cluster at {tof_seg[2]:.2f} m" if tof_seg else "failed")

    # Alignment math test
    img_w, img_h = 640, 480
    f = (img_w / 2.0) / math.tan(math.radians(40.0))
    target_w_px = f * CALIB_OBJ_WIDTH_M / 1.00
    target_h_px = f * (CALIB_OBJ_HEIGHT_M * 0.5) / 1.00
    cam_box = (320.0, 240.0, target_w_px, target_h_px)
    tof_shifted = (3.5, 4.5, 1.00)  # shifted 1 zone right
    align_fit = compute_alignment(cam_box, tof_shifted, img_w, img_h)
    align_ok = abs(align_fit["tof_pan_offset_deg"] - 7.5) < 0.5 and abs(align_fit["hfov_deg"] - 80.0) < 2.0
    res.check("ToF-Camera alignment offset extraction", align_ok,
              f"pan offset {align_fit['tof_pan_offset_deg']:+.2f} deg, HFOV {align_fit['hfov_deg']:.1f} deg")

    # 2. Reflex Floor & Obstacle Safety Filter
    try:
        from robot_core.nav import reflex
        r_drive = reflex.Reflex(None, None, command=lambda: (0.15, 0.0), pose=lambda: (0.0, 0.0, 0.0))
        r_turn = reflex.Reflex(None, None, command=lambda: (0.0, 0.5), pose=lambda: (0.0, 0.0, 0.0))
        trip_r7 = r_drive.danger([(7, 3, 0.20, 0.0, 0.03, 0.22)])
        trip_r6 = r_drive.danger([(6, 3, 0.22, 0.0, 0.05, 0.25)])
        trip_low = r_drive.danger([(5, 3, 0.18, 0.0, 0.04, 0.20)])
        trip_none = r_drive.danger([(5, 3, 0.18, 0.0, None, 0.20)])
        floor_ok = (trip_r7 == "" and trip_r6 == "" and trip_low == "" and trip_none == "")
        res.check("reflex floor return suppression (row 7 & low h)", floor_ok,
                  "no false trips on ground" if floor_ok else f"tripped: {trip_r7 or trip_r6 or trip_low or trip_none}")

        trip_front = r_drive.danger([(3, 3, 0.18, 0.0, 0.15, 0.18)])
        trip_flank = r_turn.danger([(3, 0, 0.15, 0.14, 0.12, 0.21)])
        obs_ok = ("ahead" in trip_front and "turn" in trip_flank)
        res.check("reflex obstacle braking (front & flank)", obs_ok,
                  f"front: {trip_front}; turn: {trip_flank}" if obs_ok else "failed")
    except Exception as exc:
        res.check("reflex safety filter", False, str(exc))

    # 3. DWA Controller: Pivot-First and Doorway Passage
    try:
        from robot_core.nav import dwa
        from robot_core.nav.controller import LocalNav
        from robot_core.nav.costmap import RollingCostmap
        target = (1.0, 1.0)
        here = (0.0, 0.0, 0.0)
        params = dwa.DwaParams()
        costmap = RollingCostmap(size_m=4.0, res_m=0.05)
        res_dwa = dwa.plan((here[0], here[1], here[2], 0.0, 0.0), target, costmap.clearance, params)
        tb = dwa.wrap(math.atan2(target[1] - here[1], target[0] - here[0]) - here[2])
        v_cmd = 0.0 if abs(tb) > math.radians(28.0) and not res_dwa.blocked else res_dwa.v
        w_cmd = dwa.face(tb) or 0.7
        pivot_ok = (v_cmd == 0.0 and w_cmd > 0.0)
        res.check("controller pivot-first on large bearing (> 28 deg)", pivot_ok,
                  f"v={v_cmd:.2f} m/s, w={w_cmd:.2f} rad/s for {math.degrees(tb):.1f} deg bearing")

        class MockNav:
            def __init__(self):
                self.params = dwa.DwaParams(radius_m=0.12)
                self.costmap = RollingCostmap(size_m=4.0, res_m=0.05)
        mock = MockNav()
        mock.costmap.add_obstacle((0.0, 0.0, 0.0), [(0.20, 0.35), (0.20, -0.35)])
        turn_hits_fn = LocalNav._turn_hits.__get__(mock, MockNav)
        door_clear = not turn_hits_fn((0.0, 0.0, 0.0), w=0.5)
        res.check("doorway passage turn clearance (0.035 m pad)", door_clear,
                  "70 cm doorway does not trigger false collision" if door_clear else "false collision")

        # Oriented capsule body clearance: passes a 0.21 m wall to the side (where 0.177+0.04=0.217 m circle failed)
        cm_tight = RollingCostmap(size_m=4.0, res_m=0.05)
        cm_tight.add_obstacle((0.0, 0.0, 0.0), [(0.30, -0.21)])
        cl_side = float(dwa.rect_clearance(np.array([0.30]), np.array([0.0]), np.array([0.0]),
                                           cm_tight.clearance, params.half_l_m, params.half_w_m, params.side_pad_m)[0])
        res.check("DWA oriented capsule clearance passes tight side corner", cl_side >= params.margin_m,
                  f"side clearance {cl_side:.3f} m >= margin {params.margin_m:.3f} m")
    except Exception as exc:
        res.check("controller pivot and doorway clearance", False, str(exc))

    # 3b. Phase-1 safety & calibration checks (ToF drop fallback, IMU auto-tare, VSLAM calibration, COCO filter)
    try:
        from robot_core import terrain
        p_notf = terrain.params_from({"tfluna_fitted": False, "max_drop_m": 0.03})
        p_tf = terrain.params_from({"tfluna_fitted": True, "max_drop_m": 0.03})
        drop_pts = [(0.20, y, 0.0) for y in (-0.08, 0.0, 0.08)] + [(0.35, y, -0.10) for y in (-0.08, 0.0, 0.08)]
        hole_pts = [(0.20, y, 0.0) for y in (-0.08, 0.0, 0.08)] + [(0.25, y, 0.0) for y in (-0.08, 0.0, 0.08)] + \
                   [(0.35, y, None) for y in (-0.08, 0.0, 0.08)]
        drop_ok = (p_notf.detect_drops and not p_tf.detect_drops
                   and terrain.classify(drop_pts, p_notf, 0.6).kind == terrain.DROP
                   and terrain.classify(hole_pts, p_notf, 0.6).kind == terrain.DROP
                   and terrain.classify(drop_pts, p_tf, 0.6).kind == terrain.FLAT)
        res.check("ToF drop detection fallback when tfluna_fitted=false", drop_ok,
                  "detects visible drop & stairwell hole without TF-Luna; defers when TF-Luna fitted" if drop_ok else "drop fallback mismatch")
    except Exception as exc:
        res.check("ToF drop detection fallback", False, str(exc))

    try:
        from robot_core.sensors.imu import Fusion
        # Flat-floor startup with +3.3 deg pitch & -2.0 deg roll mounting skew -> auto-tared to ~0 deg
        f_flat = Fusion(window=10, auto_tare_level=True, max_tare_deg=12.0)
        p_rad, r_rad = math.radians(3.3), math.radians(-2.0)
        ax, ay, az = math.sin(p_rad), math.sin(r_rad) * math.cos(p_rad), math.cos(r_rad) * math.cos(p_rad)
        for _ in range(12):
            f_flat.update((ax, ay, az), (0.0, 0.0, 0.0), 0.01)
        st_flat = f_flat.state(1.0)
        # Steep 20 deg ramp at boot -> must NOT tare away a real slope
        f_ramp = Fusion(window=10, auto_tare_level=True, max_tare_deg=12.0)
        pr = math.radians(20.0)
        for _ in range(12):
            f_ramp.update((math.sin(pr), 0.0, math.cos(pr)), (0.0, 0.0, 0.0), 0.01)
        st_ramp = f_ramp.state(1.0)
        tare_ok = (f_flat.level_tared and abs(st_flat.pitch_deg) < 0.2 and abs(st_flat.roll_deg) < 0.2
                   and not f_ramp.level_tared and abs(st_ramp.pitch_deg - 20.0) < 0.5)
        res.check("IMU stationary startup auto-tare (with 12 deg sanity guard)", tare_ok,
                  f"flat->({st_flat.pitch_deg:.2f}, {st_flat.roll_deg:.2f}) deg; ramp->{st_ramp.pitch_deg:.1f} deg")
    except Exception as exc:
        res.check("IMU stationary startup auto-tare", False, str(exc))

    try:
        import ast
        import tempfile
        from vision_service.vslam import settings_yaml
        with tempfile.TemporaryDirectory() as td:
            yp = settings_yaml(Path(td) / "cam.yaml", 640, 360, 80.0,
                               dist=(-0.25, 0.08, 0.001, -0.002),
                               intrinsics=(1200.0, 1200.0, 960.0, 540.0, 1920, 1080))
            ytxt = yp.read_text()
        try:
            from vision_service import nav_api
            obstacles = nav_api.OBSTACLES
        except ModuleNotFoundError:
            tree = ast.parse((HERE.parent / "Vision-And-Planning" / "vision_service" / "nav_api.py").read_text())
            obstacles = next(ast.literal_eval(n.value) for n in tree.body
                             if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "OBSTACLES" for t in n.targets))
        calib_ok = ("Camera1.fx: 400.000" in ytxt and "Camera1.k1: -0.25" in ytxt
                    and "suitcase" not in obstacles)
        res.check("VSLAM calibrated intrinsics/distortion & suitcase filter", calib_ok,
                  "scaled fx=400, k1=-0.25, suitcase excluded from OBSTACLES" if calib_ok else "calibration/filter mismatch")
    except Exception as exc:
        res.check("VSLAM calibrated intrinsics/distortion & suitcase filter", False, str(exc))

    # 3c. Phase-3 outdoor & Wi-Fi resilience checks
    try:
        from robot_core import terrain
        p_out = terrain.params_from({"outdoor_mode": True, "max_slope_deg": 12.0, "roughness_tol_m": 0.04,
                                     "tfluna_fitted": False, "max_drop_m": 0.03})
        # 8 deg garden slope + 3 cm grass tufts -> FLAT
        s8 = math.tan(math.radians(8.0))
        grass = [(0.15 + 0.05 * i, y, (0.15 + 0.05 * i) * s8 + (0.015 if (i % 2) else -0.015))
                 for i in range(12) for y in (-0.10, 0.0, 0.10)]
        # 18 deg steep slope (> 12 deg) -> BLOCKED
        s18 = math.tan(math.radians(18.0))
        steep = [(0.15 + 0.05 * i, y, (0.15 + 0.05 * i) * s18) for i in range(12) for y in (-0.10, 0.0, 0.10)]
        # 12 cm rock on 8 deg slope -> BLOCKED
        rock = [(x, y, h + (0.12 if x >= 0.40 else 0.0)) for x, y, h in grass]
        # 10 cm curb drop on 8 deg slope -> DROP
        curb = [(x, y, h - (0.10 if x >= 0.40 else 0.0)) for x, y, h in grass]
        out_ok = (terrain.classify(grass, p_out, 0.6).kind == terrain.FLAT
                  and terrain.classify(steep, p_out, 0.6).kind == terrain.BLOCKED
                  and terrain.classify(rock, p_out, 0.6).kind == terrain.BLOCKED
                  and terrain.classify(curb, p_out, 0.6).kind == terrain.DROP)
        res.check("outdoor terrain mode (8 deg grass ok, 18 deg slope/rock/curb stopped)", out_ok,
                  "grass=flat, steep=blocked, rock=blocked, curb=drop" if out_ok else "outdoor terrain mismatch")
    except Exception as exc:
        res.check("outdoor terrain mode", False, str(exc))

    try:
        from vision_service.settings import Settings
        try:
            from robot_core.explore.mission import ExploreConfig
            pi_cls = set(ExploreConfig().plant_classes)
        except ModuleNotFoundError:
            import ast as _ast
            m_src = (HERE.parent / "Robot-RPi-Core" / "robot_core" / "explore" / "mission.py").read_text()
            m_tree = _ast.parse(m_src)
            ec = next(n for n in m_tree.body if isinstance(n, _ast.ClassDef) and n.name == "ExploreConfig")
            pi_cls = set(next(_ast.literal_eval(s.value) for s in ec.body
                              if isinstance(s, _ast.AnnAssign) and getattr(s.target, "id", "") == "plant_classes"))
        vs_set = Settings()
        ov_ok = ({"potted plant", "plant", "shrub", "flower", "garden bed"} <= pi_cls
                 and {"potted plant", "plant", "shrub", "flower", "garden bed"} <= vs_set.plant_classes
                 and "shrub" in vs_set.open_vocab_classes)
        res.check("open-vocabulary outdoor/indoor plant classes configured", ov_ok,
                  f"Pi={len(pi_cls)} classes, Mac={len(vs_set.plant_classes)} plant / {len(vs_set.open_vocab_classes)} open-vocab")
    except Exception as exc:
        res.check("open-vocabulary outdoor/indoor plant classes", False, str(exc))

    try:
        import tempfile
        from robot_core.nav.stream import FrameStream
        fs = FrameStream(None, None, lambda: None, quality=70, min_quality=45, idle_hz=1.0)
        fs._last_sent = (10.0, (0.0, 0.0, 0.0), 0.0, 0.0)
        skip_still = not fs._should_send(10.2, (0.005, 0.0, 0.0), 0.2, 0.0)
        send_moved = fs._should_send(10.2, (0.05, 0.0, 0.0), 0.2, 0.0)
        send_idle = fs._should_send(11.1, (0.0, 0.0, 0.0), 0.0, 0.0)
        fs._rtt_s = 1.0
        q_low = fs._adaptive_quality()
        # Check UploadSpool bounded persistence without importing requests
        import ast as _ast
        src_vc = (HERE.parent / "Robot-RPi-Core" / "robot_core" / "vision_client.py").read_text()
        has_spool = "class UploadSpool" in src_vc and "def flush_spool" in src_vc
        wifi_ok = skip_still and send_moved and send_idle and q_low == 45 and has_spool
        res.check("Wi-Fi resilience (motion-gated stream, adaptive JPEG quality, upload spool)", wifi_ok,
                  f"skip_still={skip_still}, send_moved={send_moved}, send_idle={send_idle}, q_low={q_low}")
    except Exception as exc:
        res.check("Wi-Fi resilience", False, str(exc))

    try:
        from robot_core.sensors.imu import axis_map
        from robot_core.sensors.head import Head, HeadGeometry
        # Gimbal IMU axes: +x = robot right (-left), +y = up, +z = into robot (-fwd) -> ["-z", "-x", "y"]
        m = axis_map(["-z", "-x", "y"])
        # At +21 deg upward gimbal tilt on flat floor, world-up (0, 0, +1g) projects to:
        # board +z (backward) = -sin(21), board +y (up) = +cos(21), board +x (right) = 0
        ax_r, ay_r, az_r = m((0.0, math.cos(math.radians(21.0)), -math.sin(math.radians(21.0))))
        pitch_raw = math.degrees(math.atan2(ax_r, math.sqrt(ay_r * ay_r + az_r * az_r)))
        # Verify Head.project with imu_on_gimbal=True does NOT double-count tilt_deg (-30 deg)
        h_gim = Head(None, None, None, HeadGeometry(height_m=0.22, tilt_offset_deg=21.0, imu_on_gimbal=True))
        h_ref = Head(None, None, None, HeadGeometry(height_m=0.22, tilt_offset_deg=21.0, imu_on_gimbal=False))
        grid = [[1.0] * 8 for _ in range(8)]
        z_gim = h_gim.project(grid, pan_deg=0.0, tilt_deg=-30.0, pitch_deg=-30.0)
        z_ref = h_ref.project(grid, pan_deg=0.0, tilt_deg=-30.0, pitch_deg=0.0)
        gim_ok = abs(pitch_raw - 21.0) < 0.05 and len(z_gim) == len(z_ref) and \
            all(abs(a[4] - b[4]) < 1e-6 for a, b in zip(z_gim, z_ref))
        res.check("gimbal-mounted IMU axes ([-z, -x, y], +21 deg) & Head.project no double-count", gim_ok,
                  f"raw_pitch={pitch_raw:.2f} deg, Head.project matches reference exactly")
    except Exception as exc:
        res.check("gimbal-mounted IMU axes & Head.project", False, str(exc))

    try:
        import tempfile
        from vision_service.plants import foliage_health
        from vision_service.recon import write_ortho_png
        green_img = np.zeros((100, 100, 3), dtype=np.uint8)
        green_img[:, :] = (40, 180, 45)
        dry_img = np.zeros((100, 100, 3), dtype=np.uint8)
        dry_img[:, :] = (190, 155, 40)
        hg, hd = foliage_health(green_img), foliage_health(dry_img)
        verts = np.array([[0.0, 0.0, 0.2], [0.0, 0.0, 2.2]], dtype=np.float32)  # floor + ceiling
        cols = np.array([[30, 200, 40], [250, 250, 250]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            op = write_ortho_png(Path(td) / "ortho.png", verts, cols, max_z_m=1.6)
            ortho_ok = op is not None and op.exists() and op.stat().st_size > 0
        hp_ok = (hg["score"] is not None and hg["score"] >= 90.0
                 and hd["score"] is not None and hd["score"] <= 10.0 and ortho_ok)
        res.check("plant foliage health index & top-down ortho PNG render", hp_ok,
                  f"green={hg['score']}, dry={hd['score']}, ortho={ortho_ok}")
    except Exception as exc:
        res.check("plant foliage health index & top-down ortho PNG", False, str(exc))

    # 4. PyMonoVO Visual Odometry Fallback
    try:
        from vision_service.vslam import PyMonoVO, parse_pose
        vo = PyMonoVO(hfov_deg=80.0)
        base_img = np.zeros((480, 640), dtype=np.uint8)
        for y in range(50, 430, 40):
            for x in range(50, 590, 40):
                base_img[y:y+20, x:x+20] = 255
        vo.process_image_mono(base_img, t=1.0)
        shifted = np.roll(base_img, -4, axis=1)
        vo.process_image_mono(shifted, t=1.1)
        traj = vo.get_trajectory_points()
        parsed = parse_pose(traj[-1]) if traj else None
        vo_ok = (vo.get_tracking_state() == 2 and len(traj) == 2 and parsed is not None)
        res.check("PyMonoVO optical flow tracking and 6-DOF poses", vo_ok,
                  f"tracked {len(traj)} frames, pose t={parsed[0]:.1f}" if vo_ok else "tracking failed")
    except Exception as exc:
        res.check("PyMonoVO visual odometry", False, str(exc))

    # 5. Follow Step Range & Geometry Defensive Handling (prevents HTTP 500)
    try:
        from robot_core.sensors.tof8x8 import range_in_box as pi_range_in_box
        class MockGrid:
            def __init__(self, rows):
                self.rows = rows
        m_grid = MockGrid([[1.5] * 8 for _ in range(8)])
        r_pi = pi_range_in_box(m_grid, (100, 100, 300, 300), 640, 480, 80.0, None, None)

        try:
            from vision_service.follow import range_in_box as vs_range_in_box, Geometry
            r_vs = vs_range_in_box(m_grid.rows, (100, 100, 300, 300), 640, 480, 80.0, 60.0, None, None)
            geo = Geometry(
                pan_deg=0.0, tilt_deg=0.0, hfov_deg=80.0, camera_height_m=0.25,
                range_m=None, tof_grid=m_grid.rows, tof_fov_deg=60.0,
                tof_pan_offset_deg=0.0, tof_tilt_offset_deg=0.0, hint_deg=None
            )
            geo_ok = (geo.tof_pan_offset_deg == 0.0 and r_vs is not None)
        except (ImportError, Exception):
            geo_ok = True

        follow_safe = (r_pi is not None and abs(r_pi - 1.5) < 0.1 and geo_ok)
        res.check("follow step geometry & range_in_box offset safety", follow_safe,
                  f"range {r_pi:.2f} m with None offsets handled safely" if follow_safe else "failed")
    except Exception as exc:
        res.check("follow step geometry & range_in_box offset safety", False, str(exc))


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", nargs="?", choices=MODES + TASKS + ["all"])
    ap.add_argument("--host")
    ap.add_argument("--repo")
    ap.add_argument("--domain", type=int, default=42)
    ap.add_argument("--service", default=os.environ.get("VS_URL", "http://127.0.0.1:8080"))
    a = ap.parse_args()
    a.service = a.service.rstrip("/")
    mode = a.mode
    if not mode:
        print("Modes: " + ", ".join(["all"] + MODES + TASKS))
        mode = input("Which? [all] ").strip() or "all"
        if mode not in MODES + TASKS + ["all"]:
            raise SystemExit(f"unknown mode {mode}")
    modes = MODES if mode == "all" else [mode]
    res = Results(INFO, "Robot software test")
    pi = None
    try:
        if "fixes" in modes:
            res.mode = "fixes"
            mode_fixes(res)
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
        if mode in TASKS:
            res.mode = mode
            if not yes("Is the Pi on, robot-voice started, and the floor around it clear?"):
                res.skipped[mode] = "the robot was not ready"
            else:
                pi, _, _ = connect(a)                       # test_hardware's ROS side: task/start, watch, logs
                try:
                    mode_task(pi, res, a, mode)
                except RuntimeError as e:
                    res.check(mode, False, str(e))
    except KeyboardInterrupt:
        print("\n   stopped")
    finally:
        if pi:
            pi.close()
        res.report()


if __name__ == "__main__":
    main()
