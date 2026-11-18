#!/usr/bin/env python3
"""Calibrate and check the robot from the Mac: one script, one SSH session.

It asks whether the Pi is on, connects over SSH, stops robot-voice (it owns the
serial port, head and camera), and runs robot_core/calibrate_pi.py on the Pi. The Pi
takes the readings and works the numbers out from its own sensors (gyro, encoders,
ToF, TF-Luna); this script tells you what to do, with a countdown, and asks for a
tape-measure reading only where nothing on the robot can measure it. At the end it
prints the new robot.yaml values (and any firmware constant), and saves the run to
calibrations/<time>.json. It changes nothing on the robot by itself.

    python calibrate.py              pick a mode from a menu
    python calibrate.py all          everything below, in order
    python calibrate.py check        health: power, devices, firmware, ToF, IMU, camera (+ buzzer, OLED)
    python calibrate.py encoders     hand-roll each track: counts go up, on the right side   (robot lifted)
    python calibrate.py imu          level, nose up, left side up: pitch and roll signs; optional drift
    python calibrate.py tof          ToF orientation (floor rows, hand on the left) + a camera/ToF overlay picture
    python calibrate.py tfluna       the TF-Luna's floor range for the drop guard
    python calibrate.py head         head directions, pan zero (wall), tilt zero and head height (floor)
    python calibrate.py drive        ticks_per_cm and the ToF's range correction (backing from a wall, 2 tape
                                     readings), ticks_per_degree (gyro, 2 full turns), REVERSE_TRIM_PCT
    python calibrate.py service      the Vision-Microservice on this Mac: health and detect latency (no Pi needed)

Options: --host user@pi (remembered in ~/.config/robot-calibrate.json), --repo path on the Pi
(default ~/Robot-RPi-Core), --service http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import select
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib import request

HERE = Path(__file__).resolve().parent
CONFIG = Path.home() / ".config" / "robot-calibrate.json"
MODES = ["check", "encoders", "imu", "tof", "tfluna", "head", "drive", "service"]
ALL = ["check", "encoders", "imu", "tof", "tfluna", "head", "drive"]


# --------------------------------------------------------------------------- talking to the person

def yes(q: str, default: bool = True) -> bool:
    a = input(f"{q} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    return default if not a else a.startswith("y")


def countdown(seconds: int, what: str) -> None:
    """Say what to do, then count down; Enter starts at once."""
    print(f"\n>> {what}")
    end = time.time() + seconds
    while (left := end - time.time()) > 0:
        sys.stdout.write(f"\r   starting in {left:3.0f} s (Enter = now) ")
        sys.stdout.flush()
        r, _, _ = select.select([sys.stdin], [], [], min(1.0, left))
        if r:
            sys.stdin.readline()
            break
    sys.stdout.write("\r   go!" + " " * 30 + "\n")


def tape_cm(what: str, seconds: int = 30) -> float:
    countdown(seconds, f"Measure with a tape: {what}")
    while True:
        a = input("   tape reading in cm: ").strip().replace(",", ".")
        try:
            v = float(a)
            if 5 <= v <= 500:
                return v
        except ValueError:
            pass
        print("   a number of centimetres, e.g. 62.5")


def show_grid(rows) -> None:
    for row in rows:
        print("   " + " ".join(" --" if v is None else f"{v * 100:3.0f}" for v in row))


# --------------------------------------------------------------------------- the Pi session

class Pi:
    def __init__(self, host: str, repo: str) -> None:
        cmd = (f"cd {repo} && if [ -f .venv/bin/activate ]; then . .venv/bin/activate; fi && "
               f"exec python3 -u -m robot_core.calibrate_pi")
        self.p = subprocess.Popen(["ssh", "-T", host, cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        self.owed = 0                                      # answers still due for interrupted commands
        first = self._line()
        if not first.get("ready"):
            raise SystemExit(f"the Pi side did not start: {first}")

    def _line(self) -> dict:
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise SystemExit("the SSH session ended (see the messages above)")
            try:
                return json.loads(line)
            except ValueError:
                print(f"   (pi) {line.rstrip()}")          # stray output: show it, keep going

    def _answer(self, quiet: bool = False):
        while True:
            m = self._line()
            if "progress" in m:
                if not quiet:
                    print(f"   . {m['progress']}")
                continue
            return m

    def __call__(self, cmd: str, **args):
        while self.owed:                                   # Ctrl-C'd earlier: its answer comes first
            self._answer(quiet=True)
            self.owed -= 1
        self.p.stdin.write(json.dumps({"cmd": cmd, "args": args}) + "\n")
        self.p.stdin.flush()
        try:
            m = self._answer()
        except KeyboardInterrupt:
            self.owed += 1
            raise
        if m.get("ok"):
            return m["result"]
        raise RuntimeError(m.get("error"))

    def close(self) -> None:
        try:
            self("quit")
        except Exception:                                   # noqa: BLE001
            pass
        self.p.wait(timeout=15)


# --------------------------------------------------------------------------- results

class Results:
    def __init__(self, current: dict) -> None:
        self.cur = current
        self.new = {"/explorer": {}, "/drivetrain": {}, "firmware": {}}
        self.checks = []                                   # (name, ok, detail)

    def set(self, section: str, key: str, value, why: str) -> None:
        src = {"/explorer": "explorer", "/drivetrain": "drivetrain", "firmware": "firmware"}[section]
        self.new[section][key] = (value, self.cur.get(src, {}).get(key), why)
        print(f"   -> {key}: {value}   (was {self.cur.get(src, {}).get(key)})   {why}")

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, bool(ok), detail))
        print(f"   {'OK ' if ok else 'FIX'} {name}" + (f": {detail}" if detail else ""))

    def report(self) -> None:
        print("\n" + "=" * 72)
        if self.checks:
            print("Checks:")
            for name, ok, detail in self.checks:
                print(f"  {'OK ' if ok else 'FIX'} {name}" + (f": {detail}" if detail else ""))
        changes = {s: v for s, v in self.new.items() if v and s != "firmware"}
        if changes:
            print("\nUpdate ros2_ws/src/robot_bringup/config/robot.yaml:")
            for section, items in changes.items():
                print(f"{section}:\n  ros__parameters:")
                for k, (v, was, why) in items.items():
                    val = json.dumps(v) if isinstance(v, (bool, list)) else v
                    print(f"    {k}: {val}".ljust(46) + f"# was {was}; {why}")
        if self.new["firmware"]:
            print("\nFirmware (firmware/drivetrain/drivetrain.ino, then ./flash.sh):")
            for k, (v, was, why) in self.new["firmware"].items():
                print(f"  #define {k} {v}".ljust(46) + f"// was {was}; {why}")
        if changes or self.new["firmware"]:
            print("\nThen restart the robot: sudo systemctl restart robot-voice")
        else:
            print("\nNothing to change.")
        out = HERE / "calibrations"
        out.mkdir(exist_ok=True)
        path = out / f"{datetime.now():%Y%m%d-%H%M%S}.json"
        path.write_text(json.dumps({"current": self.cur, "new": self.new, "checks": self.checks}, indent=2))
        print(f"Saved {path}")


# --------------------------------------------------------------------------- modes

def mode_check(pi: Pi, res: Results) -> None:
    print("\n== check: health")
    for name, r in pi("diag").items():
        res.check(name, r["ok"], r["detail"])
    if yes("Play the buzzer?", False):
        r = pi("buzzer")
        res.check("buzzer", r.get("played") and yes("   Did you hear two tunes?"), r.get("detail", ""))
    if yes("Show a line on the OLED?", False):
        r = pi("oled")
        res.check("OLED", r.get("shown") and yes("   Could you read 'OLED check' on the screen?"), r.get("detail", ""))


def mode_encoders(pi: Pi, res: Results) -> None:
    print("\n== encoders: lift the robot so both tracks are off the floor")
    for side in ("left", "right"):
        other = "right" if side == "left" else "left"
        countdown(10, f"Get ready to roll the {side.upper()} track FORWARD by hand (the way the robot drives), "
                      f"for 5 s once it says go")
        w = pi("encoders_watch", seconds=5)
        if w[side] > 50 and abs(w[other]) < 0.2 * w[side]:
            res.check(f"{side} encoder", True, f"{w[side]} ticks forward")
        elif w[side] < -50:
            res.check(f"{side} encoder", False, f"counts DOWN ({w[side]}): flip its ENC_x_INVERT in the firmware")
        elif abs(w[other]) > 50:
            res.check(f"{side} encoder", False, f"the {other} side counted ({w[other]}): odom_swap_sides is the wrong "
                      f"way round (or the encoder plugs are swapped)")
            res.set("/drivetrain", "odom_swap_sides", not res.cur["drivetrain"].get("odom_swap_sides", True),
                    "the other channel counted")
        else:
            res.check(f"{side} encoder", False, f"no counts ({w}): wiring, or the track did not move")


def mode_imu(pi: Pi, res: Results) -> None:
    print("\n== imu")
    countdown(5, "Robot on flat floor, still")
    lv = pi("imu_read", seconds=2)
    res.check("IMU level", abs(lv["pitch_deg"]) < 3 and abs(lv["roll_deg"]) < 3,
              f"pitch {lv['pitch_deg']:+.1f}, roll {lv['roll_deg']:+.1f} (flat floor should read near 0)")
    for what, key, label in (("Lift the FRONT about 3 cm and hold it", "pitch_deg", "pitch"),
                             ("Lift the LEFT side about 3 cm and hold it", "roll_deg", "roll")):
        countdown(6, what)
        r = pi("imu_read", seconds=2)
        moved = r[key] - lv[key]
        other = (r["roll_deg"] - lv["roll_deg"]) if key == "pitch_deg" else (r["pitch_deg"] - lv["pitch_deg"])
        if moved > 2:
            res.check(f"IMU {label}", True, f"{moved:+.1f} deg")
        elif moved < -2:
            res.check(f"IMU {label}", False, f"reads {moved:+.1f} (should be +): flip its sign in imu_axes")
        elif abs(other) > 2:
            res.check(f"IMU {label}", False, "it moved the other angle: swap the first two imu_axes entries")
        else:
            res.check(f"IMU {label}", False, f"barely moved ({moved:+.1f}): was it lifted? imu_axes?")
        print("   put it down")
    if yes("Run the 60 s gyro drift check (robot still)?", False):
        countdown(5, "Leave the robot completely still")
        d = pi("imu_drift", seconds=60)
        res.check("gyro drift", abs(d["yaw_drift_deg"]) < 1.0, f"{d['yaw_drift_deg']:+.2f} deg in 60 s (under 1 is fine)")


def mode_tof(pi: Pi, res: Results) -> None:
    print("\n== tof: orientation")
    countdown(8, "Robot on the floor, open space ahead (at least 1.5 m)")
    rows = pi("level_grid")["rows"]
    show_grid(rows)
    far = lambda r: statistics.mean(4.0 if v is None else v for v in r)       # noqa: E731
    sides = {"bottom": far(rows[-1] + rows[-2]), "top": far(rows[0] + rows[1]),
             "left": far([r[0] for r in rows] + [r[1] for r in rows]),
             "right": far([r[-1] for r in rows] + [r[-2] for r in rows])}
    floor = min(sides, key=sides.get)                  # the floor is the nearest edge of the view
    flip_rows = bool(res.cur["explorer"].get("tof_flip_rows", False))
    if floor == "bottom":
        res.check("ToF rows", True, "floor in the bottom rows")
    elif floor == "top":
        res.check("ToF rows", False, "the floor shows in the TOP rows")
        res.set("/explorer", "tof_flip_rows", not flip_rows, "floor read in the top rows")
    else:
        tilt = int(res.cur["explorer"].get("tof_tilt", 0) or 0)
        new = ((tilt + (90 if floor == "right" else -90) + 180) % 360) - 180
        res.check("ToF rows", False, f"the floor shows in the {floor.upper()} columns: the board's turn is off by 90")
        res.set("/explorer", "tof_tilt", new, f"floor in the {floor} columns; run 'calibrate.py tof' again after "
                "(floor then in the top rows: add 180)")
        return                                         # the hand test means nothing until this is right
    countdown(8, "Hold a hand about 25 cm in front of the head, on the ROBOT'S LEFT, and keep it there")
    near = pi("watch_grid", seconds=3)["rows"] or []
    cols = [c for row in near for c, v in enumerate(row) if v is not None and v < 0.45]
    if not cols:
        res.check("ToF columns", False, "no hand seen within 45 cm: try again a little closer")
    elif statistics.mean(cols) > (len(near[0]) - 1) / 2:
        res.check("ToF columns", False, "the hand showed on the RIGHT")
        res.set("/explorer", "tof_flip_cols", not bool(res.cur["explorer"].get("tof_flip_cols", False)),
                "hand on the left read on the right")
    else:
        res.check("ToF columns", True, "hand on the left reads on the left")
    if yes("Save a camera + ToF overlay picture to check by eye?"):
        snap = pi("snapshot")
        try:
            import cv2
            import numpy as np
            sys.path.insert(0, str(HERE))
            import sensor_view
            img = cv2.imdecode(np.frombuffer(base64.b64decode(snap["jpeg"]), np.uint8), cv2.IMREAD_GRAYSCALE)
            meta = {"session": "calibrate", "n": 0, "geometry": snap["geometry"], "tof": {"rows": snap["rows"]}}
            out = HERE / "calibrations" / f"tof-overlay-{datetime.now():%Y%m%d-%H%M%S}.png"
            out.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(out), sensor_view.render(meta, img, mode="CALIBRATE"))
            print(f"   saved {out}: the red dots should sit on what is near")
            if sys.platform == "darwin":
                subprocess.run(["open", str(out)])
        except ImportError as e:
            print(f"   could not draw it ({e}): pip install -r requirements.txt")


def mode_tfluna(pi: Pi, res: Results) -> None:
    print("\n== tfluna: floor range for the drop guard")
    countdown(8, "Robot on flat floor, nothing within 1 m in front of it")
    r = pi("tfluna", seconds=3)
    ok = r["spread_cm"] <= 3 and r["missed"] <= r["readings"] // 4
    res.check("TF-Luna on the floor", ok, f"{r['median_cm']} cm, spread {r['spread_cm']} cm, "
              f"{r['missed']} missed of {r['readings'] + r['missed']}")
    res.set("/drivetrain", "tfluna_floor_cm", float(r["median_cm"]), "median on flat floor")


def mode_head(pi: Pi, res: Results) -> None:
    print("\n== head")
    countdown(5, "Watch the head: it goes LEFT, RIGHT, UP, DOWN, then centre")
    pi("head_moves")
    ok = yes("   Was it left, right, up, down, in that order?")
    res.check("head directions", ok, "" if ok else "flip pan_invert (left/right) or tilt_invert (up/down) in /explorer")
    countdown(30, "Put the robot square-on to a flat wall 0.6-1.5 m away (back it against a straight edge "
                  "parallel to the wall)")
    p = pi("pan_sweep")
    res.check("pan zero", abs(p["pan_zero_deg"]) < 0.5, f"head truly faces straight at {p['pan_zero_deg']:+.2f} deg "
              f"(wall {p['wall_m']:.2f} m)")
    if abs(p["pan_zero_deg"]) >= 0.5:
        res.set("/explorer", "pan_center_us", p["pan_center_us"], f"pan zero was {p['pan_zero_deg']:+.2f} deg off")
    countdown(20, "Turn the robot to face open, flat floor: nothing within 2 m in front")
    t = pi("tilt_sweep")
    res.check("tilt zero", abs(t["tilt_error_deg"]) < 0.5 and t["height_spread_m"] < 0.02,
              f"head points {t['tilt_error_deg']:+.2f} deg from where it is told; floor fit spread "
              f"{t['height_spread_m'] * 100:.1f} cm")
    if abs(t["tilt_error_deg"]) >= 0.5:
        res.set("/explorer", "tilt_center_us", t["tilt_center_us"], f"tilt was {t['tilt_error_deg']:+.2f} deg off")
    res.set("/explorer", "head_height_m", t["head_height_m"], "from the floor sweep (no tape needed)")


def mode_drive(pi: Pi, res: Results) -> None:
    print("\n== drive: distance, ToF ranges, turns, track trim")
    countdown(30, "Put the robot square-on to a flat wall, 0.5-0.8 m from it, with 1.5 m clear BEHIND it")
    w0 = pi("wall")
    if w0["corner_ratio"] is not None:
        perp = w0["corner_ratio"] < 1.1
        if perp != bool(res.cur["explorer"].get("tof_perpendicular", True)):
            res.set("/explorer", "tof_perpendicular", perp, f"corners/centre {w0['corner_ratio']} against a wall")
    d0 = tape_cm("from the wall to the FRONT FACE of the ToF sensor (head straight)") / 100
    readings, moved = [w0["centre_m"]], [0.0]
    if not yes("The robot will now drive BACKWARD 1 m in two steps. Clear behind it?"):
        return
    for step in (0.5, 0.5):
        pi("move", op="reverse", value=step)
        moved.append(moved[-1] + step)
        readings.append(pi("wall")["centre_m"])
    d1 = tape_cm("again, from the wall to the front face of the ToF sensor") / 100
    f = pi("fit", kind="drive", d0_m=d0, d_end_m=d1, readings=readings, moved_cmd_m=moved)
    res.check("drive distance", abs(f["moved_m"] - 1.0) < 0.02, f"told 1.00 m, drove {f['moved_m']:.3f} m")
    if abs(f["moved_m"] - 1.0) >= 0.01:
        res.set("/drivetrain", "ticks_per_cm", f["ticks_per_cm"], f"drove {f['moved_m']:.3f} m for 1.00 m")
    res.check("ToF ranges", max(abs(e) for e in f["tof_error_cm"]) < 2, f"raw error {f['tof_error_cm']} cm")
    res.set("/explorer", "tof_range_scale", f["tof_range_scale"], "fit to the tape")
    res.set("/explorer", "tof_range_offset_m", f["tof_range_offset_m"], "fit to the tape")
    if not res.cur["drivetrain"].get("imu_fitted"):
        print("   no IMU (imu_fitted false): skipping the turn calibration")
        return
    if not yes("The robot will now turn a full circle RIGHT, then LEFT, on the spot. Clear around it?"):
        return
    turns = [pi("move", op=op, value=360) for op in ("right", "left")]
    meas = [abs(t["yaw_deg"]) for t in turns if t.get("yaw_deg") is not None]
    if meas:
        tf = pi("fit", kind="turn", commanded=[360] * len(meas), measured=meas)
        res.check("turns", all(abs(m - 360) < 5 for m in meas), f"told 360, turned {tf['turned_deg']}")
        if any(abs(m - 360) >= 3 for m in meas):
            res.set("/drivetrain", "ticks_per_degree", tf["ticks_per_degree"], f"turned {tf['turned_deg']} for 360")
    ratios = [t["fwd_bwd_ratio"] for t in turns if t.get("fwd_bwd_ratio")]
    cur_trim = res.cur["firmware"].get("REVERSE_TRIM_PCT")
    if ratios and cur_trim is not None:
        tr = pi("fit", kind="trim", ratios=ratios, current=cur_trim)
        res.check("track balance", abs(tr["forward_leads_by_pct"]) < 3,
                  f"the forward-running track led by {tr['forward_leads_by_pct']:+.1f} %")
        if tr["REVERSE_TRIM_PCT"] != cur_trim:
            res.set("firmware", "REVERSE_TRIM_PCT", tr["REVERSE_TRIM_PCT"], f"forward track led by "
                    f"{tr['forward_leads_by_pct']:+.1f} % in the turns")


def mode_service(url: str, res: Results) -> None:
    print(f"\n== service: {url}")
    try:
        t0 = time.time()
        with request.urlopen(url + "/healthz", timeout=5) as r:
            h = json.loads(r.read())
        res.check("service up", h.get("status") == "ok", f"backend {h.get('backend')}, "
                  f"{(time.time() - t0) * 1000:.0f} ms")
    except Exception as e:                                  # noqa: BLE001
        res.check("service up", False, f"{e}: is it running (make start in Vision-Microservice)?")
        return
    try:
        import cv2
        import numpy as np
        img = cv2.imencode(".jpg", np.full((480, 640, 3), 128, np.uint8))[1].tobytes()
    except ImportError:
        print("   (opencv missing: detect latency skipped)")
        return
    times = []
    for i in range(10):
        b = uuid.uuid4().hex
        body = (f"--{b}\r\nContent-Disposition: form-data; name=\"seq\"\r\n\r\n{i}\r\n"
                f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"f.jpg\"\r\n"
                f"Content-Type: image/jpeg\r\n\r\n").encode() + img + f"\r\n--{b}--\r\n".encode()
        req = request.Request(url + "/v1/detect", data=body,
                              headers={"Content-Type": f"multipart/form-data; boundary={b}"})
        t0 = time.time()
        try:
            request.urlopen(req, timeout=10).read()
            times.append((time.time() - t0) * 1000)
        except Exception as e:                              # noqa: BLE001
            res.check("detect", False, str(e))
            return
    times.sort()
    res.check("detect latency", times[len(times) // 2] < 300,
              f"p50 {times[len(times) // 2]:.0f} ms, max {times[-1]:.0f} ms over 10 (on this Mac; add Wi-Fi from the Pi)")


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", nargs="?", choices=MODES + ["all"])
    ap.add_argument("--host")
    ap.add_argument("--repo")
    ap.add_argument("--service", default=os.environ.get("VS_URL", "http://127.0.0.1:8080"))
    a = ap.parse_args()
    mode = a.mode
    if not mode:
        print("Modes: " + ", ".join(["all"] + MODES))
        mode = input("Which? [all] ").strip() or "all"
        if mode not in MODES + ["all"]:
            raise SystemExit(f"unknown mode {mode}")
    if mode == "service":
        res = Results({})
        mode_service(a.service.rstrip("/"), res)
        res.report()
        return

    if not yes("Is the Pi switched on and on the same network as this Mac?"):
        print("Switch it on, wait about a minute for it to boot, then run this again.")
        return
    cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    host = a.host or cfg.get("host") or input("Pi SSH address (user@hostname or user@ip): ").strip()
    repo = a.repo or cfg.get("repo") or "~/Robot-RPi-Core"
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps({"host": host, "repo": repo}))
    if subprocess.run(["ssh", "-o", "ConnectTimeout=8", host, "true"]).returncode != 0:
        raise SystemExit(f"cannot SSH to {host}: check the address (--host), the network, and your SSH key")
    stopped = False
    if yes("Stop robot-voice on the Pi for calibration (it holds the serial port, head and camera)?"):
        stopped = subprocess.run(["ssh", "-t", host, "sudo systemctl stop robot-voice"]).returncode == 0
    pi = Pi(host, repo)
    res = Results({})
    try:
        hello = pi("hello")
        res.cur = hello
        print(f"Connected to {hello['host']} ({hello['repo']}).")
        for m in (ALL if mode == "all" else [mode]):
            try:
                globals()[f"mode_{m}"](pi, res)
            except RuntimeError as e:
                res.check(m, False, str(e))
            except KeyboardInterrupt:
                print("\n   skipped")
    finally:
        pi.close()
        res.report()
        if stopped and yes("Start robot-voice again?"):
            subprocess.run(["ssh", "-t", host, "sudo systemctl start robot-voice"])


if __name__ == "__main__":
    main()
