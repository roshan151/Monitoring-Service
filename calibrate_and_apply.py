#!/usr/bin/env python3
"""Calibrate, check, and apply settings to the robot from the Mac: one script, one SSH session.

It asks whether the Pi is on, connects over SSH, stops robot-voice (it owns the
serial port, head and camera), and runs robot_core/calibrate_pi.py on the Pi. The Pi
takes the readings and works the numbers out from its own sensors (gyro, encoders,
ToF, TF-Luna); this script tells you what to do, with a countdown, and asks for a
tape-measure reading only where nothing on the robot can measure it. At the end it
prints the new robot.yaml values, writes them to robot.yaml (with --apply or on confirmation),
syncs to the Pi via SCP, and saves the run to calibrations/<time>.json.

    python calibrate_and_apply.py              pick a mode from a menu
    python calibrate_and_apply.py all          everything below, in order
    python calibrate_and_apply.py check        health: power, devices, firmware, ToF, IMU, camera (+ buzzer, OLED)
    python calibrate_and_apply.py encoders     hand-roll each track: counts go up, on the right side   (robot lifted)
    python calibrate_and_apply.py imu          level, nose up, left side up, a left turn: pitch, roll and yaw signs; drift
    python calibrate_and_apply.py tof          ToF orientation (floor rows, hand on the left) + a camera/ToF overlay picture
    python calibrate_and_apply.py align        ToF 8x8 sensor vs Camera calibration and FOV fitting
    python calibrate_and_apply.py tfluna       the TF-Luna's floor range for the drop guard
    python calibrate_and_apply.py head         head directions, pan zero (wall), tilt zero and head height (floor)
    python calibrate_and_apply.py drive        ticks_per_cm and the ToF's range correction (backing from a wall, 2 tape
                                               readings), ticks_per_degree (gyro, 2 full turns), REVERSE_TRIM_PCT
    python calibrate_and_apply.py door         a doorway of known width from 0.5 / 1 m, centred and at each edge: how
                                               wide the ToF makes it and why; then turns, for the gyro scale (not in 'all')
    python calibrate_and_apply.py service      the Vision-And-Planning on this Mac: health and detect latency (no Pi needed)
    python calibrate_and_apply.py apply        apply newest run from calibrations/ directly to robot.yaml

Every setting of every mode is in .calibrate.json (written back in full on each run): edit it there.

Options: --apply (apply directly to robot.yaml), --dry-run, --yes (-y),
--host user@pi (remembered in .calibrate.json here), --repo path on the Pi
(default ~/Robot-RPi-Core), --service http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import select
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib import request

HERE = Path(__file__).resolve().parent
CONFIG = HERE / ".calibrate.json"            # every setting of every mode, and the Pi's address: edit it there
YAML_REL = Path("ros2_ws/src/robot_bringup/config/robot.yaml")
FIRMWARE_REL = Path("firmware/drivetrain/drivetrain.ino")

MODES = ["check", "encoders", "imu", "tof", "align", "tfluna", "head", "drive", "door", "service", "apply"]
ALL = ["check", "encoders", "imu", "tof", "align", "tfluna", "head", "drive"]


# --------------------------------------------------------------------------- settings: all in .calibrate.json

# Every number a mode uses that you might want to change. .calibrate.json overrides any of them; on each run the full
# set is written back to it, so the file always shows them all.
CFG_DEFAULTS = {
    "host": None,                                   # the Pi's SSH address (asked once)
    "repo": "~/Robot-RPi-Core",                     # the repo on the Pi
    "target": {"width_in": 38.0, "height_in": 56.0, "depth_in": 30.0, "distance_m": 1.0},   # align
    "door": {                                       # door: what the ToF makes of a doorway of known width
        "width_m": 0.75,                            # jamb to jamb, tape-measured
        "distances_m": [0.5, 1.0],                  # the door's wall to the ToF sensor's front face
        "positions": ["centre", "left", "right"],   # left / right: the robot's side edge_margin_m inside that jamb
        "edge_margin_m": 0.05,
        "pan_deg": [-45, 45], "step_deg": 5, "dwell_s": 0.5, "grids_per_step": 3,   # step-and-hold looks
        "sweep": True, "sweep_dps": 30,             # then one sweep each way, as the robot sweeps while driving
        "turn_deg": 60, "turn_distance_m": 1.0,     # the gyro check: turns of this much, at this distance (0: skip)
        "width_tol_m": 0.05, "edge_spread_tol_m": 0.04, "place_s": 40,
    },
    "head": {"pan_sweep_deg": [-30, 30, 3], "tilt_sweep_deg": [-42, -14, 3]},   # head: range and step (commanded)
    "drive": {"back_steps_m": [0.5, 0.5], "turn_deg": 360},                      # drive
    "encoders": {"watch_s": 8},
    "imu": {"drift_s": 60},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(base.get(k), dict) and isinstance(v, dict) else v
    return out


def load_cfg() -> dict:
    try:
        return _merge(CFG_DEFAULTS, json.loads(CONFIG.read_text()) if CONFIG.exists() else {})
    except (OSError, ValueError) as e:
        print(f"(could not read {CONFIG}: {e}; using the defaults)")
        return _merge(CFG_DEFAULTS, {})


def save_cfg() -> None:
    try:
        CONFIG.write_text(json.dumps(CFG, indent=2))
    except OSError as e:                            # only a convenience: never stop for it
        print(f"(could not write {CONFIG}: {e})")


CFG = load_cfg()


# --------------------------------------------------------------------------- target calibration math

@dataclass
class TargetConfig:
    width_m: float = 38.0 * 0.0254    # 0.9652 m
    height_m: float = 56.0 * 0.0254   # 1.4224 m from floor
    depth_m: float = 30.0 * 0.0254    # 0.7620 m front to back
    distance_m: float = 1.00          # 1.00 m from sensor face to object front face

    @property
    def width_in(self) -> float:
        return self.width_m / 0.0254

    @property
    def height_in(self) -> float:
        return self.height_m / 0.0254

    @property
    def depth_in(self) -> float:
        return self.depth_m / 0.0254

    @property
    def aspect_ratio(self) -> float:
        return self.width_m / max(1e-4, self.height_m)

    @classmethod
    def from_dict(cls, d: dict) -> "TargetConfig":
        cfg = cls()
        unit = d.get("unit", "in")
        scale = 0.0254 if unit == "in" else 1.0

        if "width_m" in d:
            cfg.width_m = float(d["width_m"])
        elif "width_in" in d:
            cfg.width_m = float(d["width_in"]) * 0.0254
        elif "width" in d:
            cfg.width_m = float(d["width"]) * scale

        if "height_m" in d:
            cfg.height_m = float(d["height_m"])
        elif "height_in" in d:
            cfg.height_m = float(d["height_in"]) * 0.0254
        elif "height" in d:
            cfg.height_m = float(d["height"]) * scale

        if "depth_m" in d:
            cfg.depth_m = float(d["depth_m"])
        elif "depth_in" in d:
            cfg.depth_m = float(d["depth_in"]) * 0.0254
        elif "depth" in d:
            cfg.depth_m = float(d["depth"]) * scale

        if "distance_m" in d:
            cfg.distance_m = float(d["distance_m"])
        elif "distance" in d:
            cfg.distance_m = float(d["distance"])
        return cfg

    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> "TargetConfig":
        if path:
            p = Path(path).expanduser()
            if p.exists():
                try:
                    data = json.loads(p.read_text())
                    target_dict = data.get("target", data)
                    return cls.from_dict(target_dict)
                except Exception:
                    pass
        env_dict = {}
        if os.environ.get("CALIB_TARGET_WIDTH_IN"):
            env_dict["width_in"] = float(os.environ["CALIB_TARGET_WIDTH_IN"])
        elif os.environ.get("CALIB_TARGET_WIDTH_M"):
            env_dict["width_m"] = float(os.environ["CALIB_TARGET_WIDTH_M"])

        if os.environ.get("CALIB_TARGET_HEIGHT_IN"):
            env_dict["height_in"] = float(os.environ["CALIB_TARGET_HEIGHT_IN"])
        elif os.environ.get("CALIB_TARGET_HEIGHT_M"):
            env_dict["height_m"] = float(os.environ["CALIB_TARGET_HEIGHT_M"])

        if os.environ.get("CALIB_TARGET_DEPTH_IN"):
            env_dict["depth_in"] = float(os.environ["CALIB_TARGET_DEPTH_IN"])
        elif os.environ.get("CALIB_TARGET_DEPTH_M"):
            env_dict["depth_m"] = float(os.environ["CALIB_TARGET_DEPTH_M"])

        if os.environ.get("CALIB_TARGET_DIST_M"):
            env_dict["distance_m"] = float(os.environ["CALIB_TARGET_DIST_M"])

        if env_dict:
            return cls.from_dict(env_dict)

        return cls()


DEFAULT_TARGET_CONFIG = TargetConfig()
CALIB_OBJ_WIDTH_M = DEFAULT_TARGET_CONFIG.width_m
CALIB_OBJ_HEIGHT_M = DEFAULT_TARGET_CONFIG.height_m
CALIB_OBJ_DEPTH_M = DEFAULT_TARGET_CONFIG.depth_m
CALIB_DISTANCE_M = DEFAULT_TARGET_CONFIG.distance_m

TOF_FOV_DEG = 60.0
TOF_SIZE = 8
ZONE_DEG = TOF_FOV_DEG / TOF_SIZE     # 7.5 deg


def segment_tof_target(rows: list, min_d: Optional[float] = None, max_d: Optional[float] = None,
                       target: Optional[TargetConfig] = None) -> Optional[Tuple[float, float, float]]:
    """Find the target's depth cluster in the 8x8 grid.
    Returns (center_row, center_col, median_distance_m) or None if not found."""
    import numpy as np
    if not rows or len(rows) != TOF_SIZE:
        return None
    tgt = target or DEFAULT_TARGET_CONFIG
    lo = (tgt.distance_m - 0.20) if min_d is None else min_d
    hi = (tgt.distance_m + 0.25) if max_d is None else max_d

    pts = []
    dists = []
    for r in range(TOF_SIZE):
        for c in range(len(rows[r])):
            d = rows[r][c]
            if d is not None and lo <= d <= hi:
                pts.append((r, c))
                dists.append(d)
    if len(pts) < 3:
        return None
    mean_r = float(np.mean([p[0] for p in pts]))
    mean_c = float(np.mean([p[1] for p in pts]))
    med_d = float(np.median(dists))
    return mean_r, mean_c, med_d


def detect_camera_target(img, expected_aspect: Optional[float] = None,
                         target: Optional[TargetConfig] = None) -> Optional[Tuple[float, float, float, float]]:
    """Detect the rectangular target in the camera image.
    Returns (center_u, center_v, width_px, height_px) or None."""
    import cv2
    if img is None:
        return None
    tgt = target or DEFAULT_TARGET_CONFIG
    aspect_target = tgt.aspect_ratio if expected_aspect is None else expected_aspect

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 4)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_score = -1.0
    img_center_x, img_center_y = w / 2.0, h / 2.0

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if bw < w * 0.15 or bh < h * 0.20 or area < (w * h * 0.05):
            continue
        cx, cy = x + bw / 2.0, y + bh / 2.0
        dist_from_center = math.hypot(cx - img_center_x, cy - img_center_y) / (w / 2.0)
        aspect = bw / float(bh)
        aspect_error = abs(aspect - aspect_target) / aspect_target
        score = area / (1.0 + 3.0 * dist_from_center + 2.0 * aspect_error)
        if score > best_score:
            best_score = score
            best_box = (cx, cy, float(bw), float(bh))

    return best_box


def compute_alignment(cam_center: Tuple[float, float, float, float],
                      tof_center: Tuple[float, float, float],
                      img_w: int, img_h: int,
                      nominal_hfov_deg: float = 80.0,
                      target: Optional[TargetConfig] = None) -> Dict[str, float]:
    """Compute pan offset, tilt offset, calibrated HFOV, and range scale."""
    tgt = target or DEFAULT_TARGET_CONFIG
    cx, cy, bw, bh = cam_center
    mean_r, mean_c, med_d = tof_center

    f_calib = (bw * tgt.distance_m) / tgt.width_m
    hfov_calib = math.degrees(2.0 * math.atan2(img_w / 2.0, f_calib))
    hfov_final = max(60.0, min(110.0, hfov_calib))
    f = (img_w / 2.0) / math.tan(math.radians(hfov_final) / 2.0)

    cam_pan_deg = math.degrees(math.atan2(img_w / 2.0 - cx, f))
    cam_tilt_deg = math.degrees(math.atan2(img_h / 2.0 - cy, f))

    tof_pan_deg = (3.5 - mean_c) * ZONE_DEG
    tof_tilt_deg = (3.5 - mean_r) * ZONE_DEG

    pan_offset = cam_pan_deg - tof_pan_deg
    tilt_offset = cam_tilt_deg - tof_tilt_deg
    dist_scale = tgt.distance_m / max(0.1, med_d)

    return {
        "tof_pan_offset_deg": round(pan_offset, 2),
        "tof_tilt_offset_deg": round(tilt_offset, 2),
        "hfov_deg": round(hfov_final, 1),
        "tof_range_scale": round(dist_scale, 3),
        "measured_dist_m": round(med_d, 3),
        "target_cam_center": (round(cx, 1), round(cy, 1)),
        "target_tof_grid_rc": (round(mean_r, 2), round(mean_c, 2)),
        "target_config": {
            "width_m": round(tgt.width_m, 4),
            "height_m": round(tgt.height_m, 4),
            "depth_m": round(tgt.depth_m, 4),
            "distance_m": round(tgt.distance_m, 2),
            "width_in": round(tgt.width_in, 1),
            "height_in": round(tgt.height_in, 1),
            "depth_in": round(tgt.depth_in, 1),
        }
    }


# --------------------------------------------------------------------------- applying calibration to robot.yaml

def fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    return str(v)


def set_yaml(lines: list, section: str, key: str, value) -> tuple:
    """Edit `key` under `section` in place. Returns (old text or None if added, line number)."""
    start = next((i for i, l in enumerate(lines) if l.rstrip() == f"{section}:"), None)
    if start is None:
        raise KeyError(f"section {section} not in robot.yaml")
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"^\S", lines[i])), len(lines))
    pat = re.compile(rf"^(\s+{re.escape(key)}:\s*)([^#\n]*?)(\s*#.*)?$")
    for i in range(start, end):
        m = pat.match(lines[i].rstrip("\n"))
        if m:
            old = m.group(2).strip()
            comment = m.group(3) or ""
            text = f"{m.group(1)}{fmt(value)}"
            if comment:
                col = len(m.group(1)) + len(m.group(2)) + len(comment) - len(comment.lstrip())
                text = (text.ljust(col) if len(text) < col else text + " ") + comment.lstrip()
            lines[i] = text + "\n"
            return old, i + 1
    params = next((i for i in range(start, end) if lines[i].strip() == "ros__parameters:"), start)
    indent = re.match(r"^(\s*)", lines[params + 1] if params + 1 < end else "    ").group(1) or "    "
    lines.insert(params + 1, f"{indent}{key}: {fmt(value)}    # added by calibrate.py\n")
    return None, params + 2


def apply_changes(new: dict, repo: Path, dry_run: bool = False, yes_flag: bool = False, host: Optional[str] = None) -> bool:
    """Apply new calibration values directly into robot.yaml (and firmware/drivetrain.ino),
    then optionally sync to the Pi via SCP."""
    yaml_path, fw_path = repo / YAML_REL, repo / FIRMWARE_REL
    if not yaml_path.exists():
        print(f"no robot.yaml at {yaml_path}: pass --repo <Robot-RPi-Core>")
        return False

    lines = yaml_path.read_text().splitlines(keepends=True)
    changes = []
    for section in ("/explorer", "/drivetrain", "/voice", "/display"):
        for key, (value, _was, why) in (new.get(section) or {}).items():
            old, line = set_yaml(lines, section, key, value)
            if old is None or old != fmt(value):
                changes.append(f"  robot.yaml {section} {key}: {old if old is not None else '(new)'} -> {fmt(value)}"
                               f"   (line {line}; {why})")
    fw_text = fw_path.read_text() if fw_path.exists() else ""
    for key, (value, _was, why) in (new.get("firmware") or {}).items():
        m = re.search(rf"(#define\s+{re.escape(key)}\s+)(-?\d+)", fw_text)
        if not m:
            continue
        if m.group(2) != str(value):
            changes.append(f"  drivetrain.ino {key}: {m.group(2)} -> {value}   ({why})")
            fw_text = fw_text[:m.start(2)] + str(value) + fw_text[m.end(2):]

    if not changes:
        print("Nothing to change: robot.yaml already has these values.")
        return True

    print("\nProposed changes:")
    print("\n".join(changes))
    if dry_run:
        print("(dry run: nothing written)")
        return True

    if not yes_flag and not yes("Write these changes to robot.yaml now?", True):
        print("nothing written")
        return False

    yaml_path.write_text("".join(lines))
    if new.get("firmware") and fw_text:
        fw_path.write_text(fw_text)
    print(f"Written: {yaml_path}" + (f" and {fw_path}" if new.get("firmware") else ""))

    target_host = host or CFG.get("host")

    if target_host:
        dest = f"{target_host}:{repo}/{YAML_REL}"
        if yes_flag or yes(f"Copy updated robot.yaml to the Pi ({dest}) now?", True):
            ok = subprocess.run(["scp", str(yaml_path), dest]).returncode == 0
            if ok:
                print(f"Copied to Pi. You can restart robot-voice with: ssh {target_host} sudo systemctl restart robot-voice")
            else:
                print("SCP copy failed: please copy robot.yaml to the Pi manually.")
    return True


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


def tape_cm(what: str, seconds: int = 45) -> float:
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
        countdown(15, f"Get ready to roll the {side.upper()} track FORWARD by hand (the way the robot drives), "
                      f"for {CFG['encoders']['watch_s']} s once it says go")
        w = pi("encoders_watch", seconds=CFG["encoders"]["watch_s"])
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
    on_gimbal = bool(res.cur["drivetrain"].get("imu_on_gimbal") or res.cur["explorer"].get("imu_on_gimbal"))
    expected_pitch = float(res.cur["explorer"].get("head_tilt_offset_deg") or 0.0) if on_gimbal else 0.0
    if on_gimbal:
        print("   centering gimbal to pan=0, tilt=0 before IMU calibration...")
        g = pi("gimbal_move", pan=0.0, tilt=0.0)
        print(f"   gimbal at pan={g['pan_deg']:+.1f} deg, tilt={g['tilt_deg']:+.1f} deg")
    countdown(10, "Robot on flat floor, still")
    lv = pi("imu_read", seconds=3)
    pitch_err = lv["pitch_deg"] - expected_pitch
    res.check("IMU level", abs(pitch_err) < 15 and abs(lv["roll_deg"]) < 3,
              f"raw pitch {lv['pitch_deg']:+.1f} (expected {expected_pitch:+.1f} -> err {pitch_err:+.1f}), "
              f"roll {lv['roll_deg']:+.1f} on flat floor (roll should be near 0; "
              "over 15 deg pitch error: is the board mounted as imu_axes says?)")
    res.set("/drivetrain", "imu_pitch_offset_deg", round(lv["pitch_deg"], 1),
            "pitch read on flat floor (at gimbal tilt=0): taken off every reading, so flat reads 0")
    for what, key, label in (("Lift the FRONT about 3 cm and hold it", "pitch_deg", "pitch"),
                             ("Lift the LEFT side about 3 cm and hold it", "roll_deg", "roll")):
        countdown(12, what + " (it measures for 4 s once it says go)")
        r = pi("imu_read", seconds=4)
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
    # Yaw: the heading follow and explore steer by. A wrong sign here makes every turn look backwards.
    countdown(12, "When it says go: turn the robot LEFT (anticlockwise seen from above) about a quarter turn, "
                  "on the floor, within 10 s")
    r = pi("imu_read", seconds=10)
    turned = r["yaw_end_deg"] - r["yaw_start_deg"]
    if turned > 45:
        res.check("IMU yaw", True, f"{turned:+.0f} deg for a left turn")
    elif turned < -45:
        res.check("IMU yaw", False, f"reads {turned:+.0f} for a LEFT turn (should be +): the z entry of imu_axes "
                  "is the wrong way; flip it, then re-run imu to re-check pitch and roll")
    else:
        res.check("IMU yaw", False, f"only {turned:+.0f} deg: was it turned a quarter turn? (gyro on z?)")
    secs = CFG["imu"]["drift_s"]
    if yes(f"Run the {secs} s gyro drift check (robot still)?", False):
        countdown(5, "Leave the robot completely still")
        d = pi("imu_drift", seconds=secs)
        res.check("gyro drift", abs(d["yaw_drift_deg"]) < secs / 60.0,
                  f"{d['yaw_drift_deg']:+.2f} deg in {secs} s (under {secs / 60.0:.1f} is fine)")


def mode_tof(pi: Pi, res: Results) -> None:
    print("\n== tof: orientation")
    countdown(15, "Robot on the floor, open space ahead (at least 1.5 m)")
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
    countdown(15, "Hold a hand about 25 cm in front of the head, on the ROBOT'S LEFT, and keep it there")
    near = pi("watch_grid", seconds=5)["rows"] or []
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
            import navigation_debugging as sensor_view
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


def mode_align(pi: Pi, res: Results, target: Optional[object] = None) -> None:
    tgt = target if isinstance(target, TargetConfig) else TargetConfig.load(CONFIG)

    print("\n== align: ToF 8x8 sensor vs Camera calibration")
    print("Configured calibration target:")
    print(f"  * Height from floor: {tgt.height_in:.1f} inches ({tgt.height_m:.2f} m)")
    print(f"  * Width: {tgt.width_in:.1f} inches ({tgt.width_m:.2f} m)")
    print(f"  * Depth: {tgt.depth_in:.1f} inches ({tgt.depth_m:.2f} m)")
    print(f"  * Distance: {tgt.distance_m:.2f} m centered from sensor face")
    print(f"  * Note: there should be a wall / open background behind it (> {tgt.distance_m + 0.4:.1f} m)")
    countdown(35, f"Place the robot centered and exactly {tgt.distance_m:.2f} m ({tgt.distance_m * 100:.0f} cm) away from the front face of the object")

    snap = pi("snapshot")
    import base64
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(base64.b64decode(snap["jpeg"]), np.uint8), cv2.IMREAD_COLOR)
    rows = snap.get("rows") or []

    # 1. Segment depth cluster in ToF
    tof_res = segment_tof_target(rows, target=tgt)
    if not tof_res:
        res.check("align ToF target", False, f"could not isolate target front face ({tgt.distance_m - 0.2:.2f} - {tgt.distance_m + 0.25:.2f} m) in ToF grid")
        return
    res.check("align ToF target", True, f"found depth cluster at {tof_res[2]:.2f} m (grid row {tof_res[0]:.2f}, col {tof_res[1]:.2f})")

    # 2. Segment rectangular target in camera image
    cam_res = detect_camera_target(img, target=tgt)
    if not cam_res:
        # Fallback to center region if image segmentation cannot isolate contour cleanly
        h, w = img.shape[:2]
        cam_res = (w / 2.0, h / 2.0, w * 0.55, h * 0.65)
        print("   (notice: contour detection fuzzy, using image center for camera axis)")

    # 3. Compute alignment
    h, w = img.shape[:2]
    cur_hfov = float(res.cur.get("explorer", {}).get("hfov_deg", 80.0) or 80.0)
    fit = compute_alignment(cam_res, tof_res, w, h, nominal_hfov_deg=cur_hfov, target=tgt)

    d_pan = fit["tof_pan_offset_deg"]
    d_tilt = fit["tof_tilt_offset_deg"]
    hfov = fit["hfov_deg"]
    scale = fit["tof_range_scale"]

    ok = abs(d_pan) <= 8.0 and abs(d_tilt) <= 8.0
    res.check("ToF-Camera alignment", ok,
              f"pan offset {d_pan:+.2f} deg, tilt offset {d_tilt:+.2f} deg, calibrated HFOV {hfov:.1f} deg")

    res.set("/explorer", "tof_pan_offset_deg", d_pan, "align ToF with camera")
    res.set("/explorer", "tof_tilt_offset_deg", d_tilt, "align ToF with camera")
    res.set("/explorer", "hfov_deg", hfov, f"calibrated from {tgt.width_in:.0f} in target at {tgt.distance_m:.1f} m")
    if abs(scale - 1.0) >= 0.03:
        res.set("/explorer", "tof_range_scale", scale, f"fit to {tgt.distance_m:.1f} m target")

    # Save visual verification image
    out_dir = HERE / "calibrations"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"tof-camera-calibrated-{datetime.now():%Y%m%d-%H%M%S}.png"
    try:
        sys.path.insert(0, str(HERE))
        import navigation_debugging as sensor_view
        meta = {
            "session": "calibrate_align", "n": 0,
            "geometry": {
                "camera_hfov_deg": hfov,
                "tof_fov_deg": 60.0,
                "tof_pan_offset_deg": d_pan,
                "tof_tilt_offset_deg": d_tilt
            },
            "tof": {"rows": rows}
        }
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        view = sensor_view.render(meta, gray, mode="CALIBRATED")
        # Draw green bounding box around detected camera target
        cx, cy, bw, bh = cam_res
        cv2.rectangle(view, (int(cx - bw/2), int(cy - bh/2 + 54)), (int(cx + bw/2), int(cy + bh/2 + 54)), (0, 255, 0), 2)
        cv2.imwrite(str(out_path), view)
        print(f"   saved calibrated overlay: {out_path}")
        if sys.platform == "darwin":
            subprocess.run(["open", str(out_path)])
    except Exception as exc:
        print(f"   (could not save visual overlay: {exc})")


def mode_apply(_pi: Optional[Pi], res: Results, repo: Optional[Path] = None, dry_run: bool = False, yes_flag: bool = False) -> None:
    print("\n== apply: apply newest calibration to robot.yaml")
    runs = sorted((HERE / "calibrations").glob("2*.json"))
    if not runs:
        res.check("apply calibration", False, "no runs in calibrations/: run calibrate.py first")
        return
    newest = runs[-1]
    print(f"Loading {newest.name}...")
    data = json.loads(newest.read_text())
    new = data.get("new", {})
    target_repo = repo or (HERE.parent / "Robot-RPi-Core")
    ok = apply_changes(new, target_repo, dry_run=dry_run, yes_flag=yes_flag)
    res.check("apply calibration", ok, f"applied from {newest.name}")


def mode_tfluna(pi: Pi, res: Results) -> None:
    print("\n== tfluna: floor range for the drop guard")
    countdown(15, "Robot on flat floor, nothing within 1 m in front of it")
    r = pi("tfluna", seconds=3)
    ok = r["spread_cm"] <= 3 and r["missed"] <= r["readings"] // 4
    res.check("TF-Luna on the floor", ok, f"{r['median_cm']} cm, spread {r['spread_cm']} cm, "
              f"{r['missed']} missed of {r['readings'] + r['missed']}")
    res.set("/drivetrain", "tfluna_floor_cm", float(r["median_cm"]), "median on flat floor")


def mode_head(pi: Pi, res: Results) -> None:
    print("\n== head")
    countdown(10, "Watch the head: it goes LEFT, RIGHT, UP, DOWN, then centre")
    pi("head_moves")
    ok = yes("   Was it left, right, up, down, in that order?")
    res.check("head directions", ok, "" if ok else "flip pan_invert (left/right) or tilt_invert (up/down) in /explorer")
    countdown(45, "Put the robot square-on to a flat wall 0.6-1.5 m away (back it against a straight edge "
                  "parallel to the wall)")
    lo, hi, step = CFG["head"]["pan_sweep_deg"]
    p = pi("pan_sweep", lo=lo, hi=hi, step=step)
    res.check("pan zero", abs(p["pan_zero_deg"]) < 0.5, f"head truly faces straight at {p['pan_zero_deg']:+.2f} deg "
              f"(wall {p['wall_m']:.2f} m)")
    if abs(p["pan_zero_deg"]) >= 0.5:
        res.set("/explorer", "pan_center_us", p["pan_center_us"], f"pan zero was {p['pan_zero_deg']:+.2f} deg off")
    countdown(30, "Turn the robot to face open, flat floor: nothing within 2 m in front")
    lo, hi, step = CFG["head"]["tilt_sweep_deg"]
    t = pi("tilt_sweep", lo=lo, hi=hi, step=step)
    res.check("tilt zero", abs(t["tilt_error_deg"]) < 0.5 and t["height_spread_m"] < 0.02,
              f"head points {t['tilt_error_deg']:+.2f} deg from where it is told; floor fit spread "
              f"{t['height_spread_m'] * 100:.1f} cm")
    if abs(t["tilt_error_deg"]) >= 0.5:
        res.set("/explorer", "tilt_center_us", t["tilt_center_us"], f"tilt was {t['tilt_error_deg']:+.2f} deg off")
    res.set("/explorer", "head_height_m", t["head_height_m"], "from the floor sweep (no tape needed)")


def mode_drive(pi: Pi, res: Results) -> None:
    print("\n== drive: distance, ToF ranges, turns, track trim")
    steps, turn = CFG["drive"]["back_steps_m"], CFG["drive"]["turn_deg"]
    total = sum(steps)
    countdown(45, f"Put the robot square-on to a flat wall, 0.5-0.8 m from it, with {total + 0.5:.1f} m clear BEHIND it")
    w0 = pi("wall")
    if w0["corner_ratio"] is not None:
        perp = w0["corner_ratio"] < 1.1
        if perp != bool(res.cur["explorer"].get("tof_perpendicular", True)):
            res.set("/explorer", "tof_perpendicular", perp, f"corners/centre {w0['corner_ratio']} against a wall")
    d0 = tape_cm("from the wall to the FRONT FACE of the ToF sensor (head straight)") / 100
    readings, moved = [w0["centre_m"]], [0.0]
    if not yes(f"The robot will now drive BACKWARD {total:.2f} m in {len(steps)} steps. Clear behind it?"):
        return
    for step in steps:
        pi("move", op="reverse", value=step)
        moved.append(moved[-1] + step)
        readings.append(pi("wall")["centre_m"])
    d1 = tape_cm("again, from the wall to the front face of the ToF sensor") / 100
    f = pi("fit", kind="drive", d0_m=d0, d_end_m=d1, readings=readings, moved_cmd_m=moved)
    res.check("drive distance", abs(f["moved_m"] - total) < 0.02, f"told {total:.2f} m, drove {f['moved_m']:.3f} m")
    if abs(f["moved_m"] - total) >= 0.01:
        res.set("/drivetrain", "ticks_per_cm", f["ticks_per_cm"], f"drove {f['moved_m']:.3f} m for {total:.2f} m")
    res.check("ToF ranges", max(abs(e) for e in f["tof_error_cm"]) < 2, f"raw error {f['tof_error_cm']} cm")
    res.set("/explorer", "tof_range_scale", f["tof_range_scale"], "fit to the tape")
    res.set("/explorer", "tof_range_offset_m", f["tof_range_offset_m"], "fit to the tape")
    if not res.cur["drivetrain"].get("imu_fitted"):
        print("   no IMU (imu_fitted false): skipping the turn calibration")
        return
    if not yes(f"The robot will now turn {turn:.0f} deg RIGHT, then LEFT, on the spot. Clear around it?"):
        return
    turns, meas = [], []
    for op in ("right", "left"):
        print(f"   Mark where the robot's nose points now; it turns {turn:.0f} deg {op.upper()}.")
        t = pi("move", op=op, value=turn)
        turns.append(t)
        gyro = abs(t["yaw_deg"]) if t.get("yaw_deg") is not None else None
        a = input(f"   How far did it really turn, in degrees (asked {turn:.0f}; one full circle = 360"
                  f"{f'; Enter = the gyro: {gyro:.0f}' if gyro else ''})? ").strip()
        try:
            real = float(a) if a else gyro
        except ValueError:
            real = gyro
        if real:
            meas.append(real)
            if gyro and abs(gyro - real) > 15 * turn / 360.0:
                res.check(f"gyro on the {op} turn", False, f"gyro said {gyro:.0f}, you saw {real:.0f}: "
                          "the gyro's heading is off (imu_axes, or it was not zeroed: run 'calibrate.py imu')")
    if meas:
        tf = pi("fit", kind="turn", commanded=[turn] * len(meas), measured=meas)
        res.check("turns", all(abs(m - turn) < 5 for m in meas), f"told {turn:.0f}, turned {tf['turned_deg']}")
        if any(abs(m - turn) >= 3 for m in meas):
            res.set("/drivetrain", "ticks_per_degree", tf["ticks_per_degree"], f"turned {tf['turned_deg']} for {turn:.0f}")
    ratios = [t["fwd_bwd_ratio"] for t in turns if t.get("fwd_bwd_ratio")]
    cur_trim = res.cur["firmware"].get("REVERSE_TRIM_PCT")
    if ratios and cur_trim is not None:
        tr = pi("fit", kind="trim", ratios=ratios, current=cur_trim)
        res.check("track balance", abs(tr["forward_leads_by_pct"]) < 3,
                  f"the forward-running track led by {tr['forward_leads_by_pct']:+.1f} %")
        if tr["REVERSE_TRIM_PCT"] != cur_trim:
            res.set("firmware", "REVERSE_TRIM_PCT", tr["REVERSE_TRIM_PCT"], f"forward track led by "
                    f"{tr['forward_leads_by_pct']:+.1f} % in the turns")


def mode_door(pi: Pi, res: Results) -> None:
    """A doorway of known width, from 2 distances x centre / left edge / right edge, the robot still and only the
    head moving: how wide the robot's own ToF processing makes it (look by look, and as a map keeping every look's
    edge would hold it), and which error it is. Then turns, for the gyro's heading scale. Settings: door in
    .calibrate.json."""
    dc = CFG["door"]
    print("\n== door: what the ToF makes of a doorway of known width")
    a = input(f"   The doorway is {dc['width_m'] * 100:.1f} cm jamb to jamb (.calibrate.json door.width_m). "
              "Enter to keep it, or type the tape reading in cm: ").strip().replace(",", ".")
    try:
        if a:
            dc["width_m"] = round(float(a) / 100.0, 4)
            save_cfg()
    except ValueError:
        print("   not a number: keeping it")
    W, lo, hi = dc["width_m"], dc["pan_deg"][0], dc["pan_deg"][1]
    ex = res.cur.get("explorer", {})
    half_w, sensor_x = float(ex.get("robot_width_m", 0.25)) / 2.0, float(ex.get("pan_axis_x_m", 0.05))
    off = W / 2.0 - half_w - dc["edge_margin_m"]               # left / right: the robot's centre from the door's
    if off < 0:
        print(f"   the door is narrower than the robot plus {dc['edge_margin_m'] * 100:.0f} cm a side: centre only")
    where = {"centre": "CENTRED on the door",
             "left": f"with its LEFT side {dc['edge_margin_m'] * 100:.0f} cm inside the LEFT jamb (its centre "
                     f"{off * 100:.0f} cm left of the door's centre); driving straight it would still pass",
             "right": f"with its RIGHT side {dc['edge_margin_m'] * 100:.0f} cm inside the RIGHT jamb (its centre "
                      f"{off * 100:.0f} cm right of the door's centre)"}
    fit = lambda r, d: pi("fit", kind="door", looks=[lk["pts"] for lk in r["looks"]], half_w_m=r["half_w_m"],   # noqa: E731
                          expect_m=d + sensor_x)
    runs = []
    for d in dc["distances_m"]:
        for pos in dc["positions"]:
            if pos != "centre" and off < 0:
                continue
            countdown(dc["place_s"], f"Door, {pos} at {d:.2f} m: place the robot facing the door roughly square, "
                                     f"{d:.2f} m from the door's wall to the ToF sensor's front face, {where[pos]}. "
                                     f"Nothing in the doorway; stand aside")
            r = None
            try:
                r = pi("door_steps", lo=lo, hi=hi, step=dc["step_deg"], dwell_s=dc["dwell_s"], grids=dc["grids_per_step"])
                f = fit(r, d)
                sw = fit(pi("door_sweep", lo=lo, hi=hi, dps=dc["sweep_dps"]), d) if dc["sweep"] else None
            except RuntimeError as e:
                res.check(f"door {pos} {d:.2f} m", False, str(e))
                runs.append({"d": d, "pos": pos, "error": str(e), "looks": r["looks"] if r else []})   # kept: look later
                continue
            expect = {"centre": 0.0, "left": -off, "right": off}[pos]   # the door's centre from the robot's line
            run = {"d": d, "pos": pos, "fit": f, "sweep": sw, "expect_u": expect, "looks": r["looks"],
                   "wall_err_m": round(f["wall_m"] - sensor_x - d, 3)}
            runs.append(run)
            print(f"   width {f['width_m'] * 100:.1f} cm look by look, {f['width_worst_m'] * 100:.1f} cm as a map "
                  f"holds it (real {W * 100:.1f}); edges spread L {f['left']['spread_m']} R {f['right']['spread_m']} m; "
                  f"wall {run['wall_err_m'] * 100:+.1f} cm off, robot {f['wall_angle_deg']:+.1f} deg off square"
                  + (f"; sweeping: {sw['width_worst_m'] * 100:.1f} cm" if sw else ""))
    if dc["turn_deg"] and dc["turn_distance_m"] and res.cur.get("drivetrain", {}).get("imu_fitted"):
        countdown(dc["place_s"], f"Gyro check: the robot CENTRED on the door, {dc['turn_distance_m']:.2f} m away, "
                                 f"room to turn {dc['turn_deg']} deg each way")
        try:
            t = pi("door_turn", turn_deg=dc["turn_deg"], lo=-30, hi=30, step=dc["step_deg"], dwell_s=dc["dwell_s"])
            k = t["fit"]["gyro_scale"]
            res.check("gyro heading scale", abs(k - 1.0) < 0.05,
                      f"the door lines up after the turns with the gyro's yaw x {k:.2f} (the gyro reads "
                      f"{1 / k:.2f}x the real turn); door spread {t['fit']['spread_m'] * 100:.1f} cm at that scale, "
                      f"{(t['fit']['spread_at_1_m'] or 0) * 100:.1f} cm as read")
            runs.append({"turn": t})
        except RuntimeError as e:
            res.check("gyro heading scale", False, str(e))
    fits = [r for r in runs if "fit" in r]
    if not fits:
        return
    for r in fits:                                              # what the planner gets
        f = r["fit"]
        res.check(f"door {r['pos']} {r['d']:.2f} m: width as the map holds it", abs(f["width_worst_m"] - W) <= dc["width_tol_m"],
                  f"{f['width_worst_m'] * 100:.1f} cm for {W * 100:.1f} (look by look {f['width_m'] * 100:.1f})")
    spread = max((s for r in fits for s in (r["fit"]["left"]["spread_m"], r["fit"]["right"]["spread_m"]) if s is not None), default=0)
    res.check("door edges steady look to look", spread <= dc["edge_spread_tol_m"],
              f"one look puts an edge up to {spread * 100:.1f} cm from another (a map keeps the worst: the edge zones "
              "flicker between the jamb and the room beyond)" if spread > dc["edge_spread_tol_m"] else f"{spread * 100:.1f} cm")
    cen = sorted((r for r in fits if r["pos"] == "centre"), key=lambda r: r["d"])
    if len(cen) >= 2:                                           # what grows with distance and what does not
        ds = [r["d"] for r in cen]
        e = [r["fit"]["width_worst_m"] - W for r in cen]
        b = (e[-1] - e[0]) / (ds[-1] - ds[0])
        a0 = e[0] - b * ds[0]
        print(f"   width error = {a0 * 100:+.1f} cm {b * 100:+.1f} cm per metre away")
        if abs(b) > 0.03:
            res.check("door: error grows with distance", False, f"{b * 100:+.1f} cm a metre: the zones' 7.5 deg cones "
                      "(a zone straddling a jamb reads it, at the cone's centre). Fix in the ToF edge handling")
        if abs(a0) > 0.03:
            res.check("door: error the same at every distance", False, f"{a0 * 100:+.1f} cm: the angles are scaled "
                      "(the ToF's field of view, tof8x8.FOV_DEG 60) or the ranges (run 'drive')")
    # A ToF pointing off its head's axis and a robot placed a little off square turn the points alike: from the ToF
    # alone they cannot be told apart, so this is reported, not fitted. 'head' squares the robot on a straight edge.
    ang = statistics.median(r["fit"]["wall_angle_deg"] for r in fits)
    res.check("door: the wall seen square", abs(ang) < 2.0, f"the wall reads {ang:+.1f} deg off square: the robot was "
              "placed that much off, or the head/ToF points that much off (if it was square: run 'head')")
    werr = [r["wall_err_m"] for r in fits]
    res.check("door: ToF range to the wall", abs(statistics.mean(werr)) < 0.02,
              f"{statistics.mean(werr) * 100:+.1f} cm on average against your placement (over 2 cm: run 'drive')")
    sw = [(r["sweep"]["width_worst_m"] - r["fit"]["width_worst_m"]) for r in fits if r.get("sweep")]
    if sw:
        res.check("door: sweeping vs holding the head", min(sw) > -0.03, f"sweeping changes the width by "
                  f"{min(sw) * 100:+.1f} to {max(sw) * 100:+.1f} cm (narrower when sweeping: the head lags its command)")
    need = 0.177 + 0.04                                          # DWA's circle and margin, from the centreline
    for r in fits:
        if r["pos"] == "centre":
            continue
        f = r["fit"]
        near = f["clear_left_m"] if r["pos"] == "left" else f["clear_right_m"]
        jamb = f["left"]["median_m"] if r["pos"] == "left" else -f["right"]["median_m"]
        print(f"   {r['pos']} at {r['d']:.2f} m: the near jamb {near * 100:.1f} cm from the robot's side "
              f"(really {dc['edge_margin_m'] * 100:.0f}); its centreline {jamb * 100:.1f} cm from it: the planner's "
              f"circle needs {need * 100:.1f}, so driving straight it would {'' if jamb >= need else 'NOT '}go")
    out = HERE / "calibrations"
    out.mkdir(exist_ok=True)
    path = out / f"door-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"door_m": W, "settings": dc, "runs": runs}, indent=1))
    print(f"   every look saved to {path}")


def mode_service(url: str, res: Results) -> None:
    print(f"\n== service: {url}")
    try:
        t0 = time.time()
        with request.urlopen(url + "/healthz", timeout=5) as r:
            h = json.loads(r.read())
        res.check("service up", h.get("status") == "ok", f"backend {h.get('backend')}, "
                  f"{(time.time() - t0) * 1000:.0f} ms")
    except Exception as e:                                  # noqa: BLE001
        res.check("service up", False, f"{e}: is it running (make start in Vision-And-Planning)?")
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
    ap.add_argument("--apply", action="store_true", help="Apply new calibration parameters directly to robot.yaml")
    ap.add_argument("--dry-run", action="store_true", help="Preview calibration changes without writing to robot.yaml")
    ap.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")
    ap.add_argument("--target-width", type=float, help="Target width in inches (default: 38.0)")
    ap.add_argument("--target-height", type=float, help="Target height in inches (default: 56.0)")
    ap.add_argument("--target-depth", type=float, help="Target depth in inches (default: 30.0)")
    ap.add_argument("--target-distance", type=float, help="Target distance in metres (default: 1.00)")
    ap.add_argument("--target-config", help="Path to JSON config defining calibration target")
    a = ap.parse_args()

    # Load target configuration (config file -> env vars -> CLI args)
    target_cfg = TargetConfig.load(a.target_config or CONFIG)
    if a.target_width is not None:
        target_cfg.width_m = a.target_width * 0.0254
    if a.target_height is not None:
        target_cfg.height_m = a.target_height * 0.0254
    if a.target_depth is not None:
        target_cfg.depth_m = a.target_depth * 0.0254
    if a.target_distance is not None:
        target_cfg.distance_m = a.target_distance

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

    if mode == "apply":
        res = Results({})
        mode_apply(None, res, repo=Path(a.repo).expanduser() if a.repo else None, dry_run=a.dry_run, yes_flag=a.yes)
        return

    if not yes("Is the Pi switched on and on the same network as this Mac?"):
        print("Switch it on, wait about a minute for it to boot, then run this again.")
        return
    host = a.host or CFG.get("host") or input("Pi SSH address (user@hostname or user@ip): ").strip()
    repo = a.repo or CFG.get("repo") or "~/Robot-RPi-Core"
    CFG["host"], CFG["repo"] = host, repo
    CFG["target"] = {"width_in": round(target_cfg.width_in, 1), "height_in": round(target_cfg.height_in, 1),
                     "depth_in": round(target_cfg.depth_in, 1), "distance_m": round(target_cfg.distance_m, 2)}
    save_cfg()                                              # the full set: every setting shows in the file
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
                fn = globals()[f"mode_{m}"]
                if m == "align":
                    fn(pi, res, target=target_cfg)
                else:
                    fn(pi, res)
            except RuntimeError as e:
                res.check(m, False, str(e))
            except KeyboardInterrupt:
                print("\n   skipped")
    finally:
        pi.close()
        res.report()
        has_changes = any(res.new.get(s) for s in ("/explorer", "/drivetrain", "firmware"))
        if has_changes:
            repo_path = Path(repo).expanduser() if repo and Path(repo).expanduser().is_dir() else (HERE.parent / "Robot-RPi-Core")
            if a.apply or a.yes or yes("\nApply these changes to robot.yaml and sync to the Pi now?", True):
                apply_changes(res.new, repo_path, dry_run=a.dry_run, yes_flag=a.yes or a.apply, host=host)
        if stopped and yes("Start robot-voice again?"):
            subprocess.run(["ssh", "-t", host, "sudo systemctl start robot-voice"])
        res.report()
        if stopped and yes("Start robot-voice again?"):
            subprocess.run(["ssh", "-t", host, "sudo systemctl start robot-voice"])


if __name__ == "__main__":
    main()
