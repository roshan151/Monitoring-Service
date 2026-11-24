#!/usr/bin/env python3
"""Show what the robot's sensors saw: its camera frame in black and white, with the
8x8 ToF grid laid over it as coloured dots (red = close, green = far), each labelled
with its range in metres, so you can check that the numbers match the picture.

It reads the files Vision-And-Planning stores for the robot's debug mode
(robot.yaml /voice debug_mode: true); it makes no network calls and puts no load on
the robot. Default folder: ../Vision-And-Planning/debug, or VS_DEBUG_DIR, or --data.

    python navigation_debugging.py                          live: the newest upload, as it arrives
    python navigation_debugging.py --list                   the saved sessions
    python navigation_debugging.py --session 20261118-104200   step through a saved session
    python navigation_debugging.py --snapshot view.png      the newest (or --session [--n N]) to a PNG

A whole follow or explore run on one page, every problem found and marked (see "--report" below):
    python3 navigation_debugging.py --report --open                     the newest session
    python3 navigation_debugging.py --report --session 20261123-194111  -> reports/20261123-194111.html

Planning and obstacle avoidance (Robot-RPi-Core robot_core/nav, imported from ../Robot-RPi-Core):
    python3 navigation_debugging.py --nav                                live
    python3 navigation_debugging.py --nav --session 20261119-115626      saved sessions
    python navigation_debugging.py --sim box                             simulate the nav stack in a room
    python navigation_debugging.py --sim trap --snapshot trap.png        run it to the end, save the last view

The Mac's side (Vision-And-Planning records it in debug/<session>/nav/ while a debug session runs):
    --nav also adds a third panel when the session has it: the global map at that moment, the Mac's
    planned path, visual SLAM's base trajectory against raw and SLAM-corrected odometry, the odometry
    fit (scale, residual, its point pairs), tracking state and frame counts, camera-found objects

Keys: q / Esc quit · space pause / play · n or -> next · p or <- previous
      i image on / off · d range numbers on / off · m nav panel on / off · g SLAM panel on / off
      s save the view as PNG
      (--sim: n step · r restart · 1-6 scenario)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

NEAR_M, FAR_M = 0.3, 2.0        # colour scale: red at or under NEAR_M, green at or past FAR_M
HEADER, FOOTER = 54, 34         # px of text above and legend below the picture
DEFAULT_GEOMETRY = {"camera_hfov_deg": 80.0, "tof_fov_deg": 60.0, "tof_pan_offset_deg": 0.0, "tof_tilt_offset_deg": 0.0}
PLAY_S = 0.5                    # replay: seconds per frame (the robot sends at most 2 a second)


def default_data_dir() -> Path:
    env = os.environ.get("VS_DEBUG_DIR")
    if env:
        return Path(env).expanduser()
    here = Path(__file__).resolve().parent.parent
    new = here / "Vision-And-Planning" / "debug"                 # the service's folder was renamed
    return new if new.parent.exists() else None


# --------------------------------------------------------------------------- reading

def load(data: Path, rel_json: str) -> tuple[dict, Optional[np.ndarray]]:
    meta = json.loads((data / rel_json).read_text())
    img = None
    if meta.get("image"):
        img = cv2.imread(str((data / rel_json).parent / meta["image"]), cv2.IMREAD_GRAYSCALE)
    return meta, img


def session_frames(data: Path, session: str) -> list[str]:
    return [f"{session}/{p.name}" for p in sorted((data / session).glob("[0-9]*.json"))]


def latest(data: Path) -> Optional[dict]:
    try:
        return json.loads((data / "latest.json").read_text())
    except (OSError, ValueError):
        return None


def list_sessions(data: Path) -> None:
    rows = []
    for d in sorted(p for p in data.iterdir() if p.is_dir()) if data.exists() else []:
        frames = sorted(d.glob("[0-9]*.json"))
        if not frames:
            continue
        first, last = (json.loads(f.read_text()).get("ts", "?") for f in (frames[0], frames[-1]))
        rows.append(f"{d.name}  {len(frames):6d} uploads  {first[11:19]} - {last[11:19]} UTC")
    print("\n".join(rows) if rows else f"no debug sessions in {data}")


# --------------------------------------------------------------------------- drawing

def colour(d: float) -> tuple[int, int, int]:
    """BGR from red (near) through yellow to green (far)."""
    t = min(1.0, max(0.0, (d - NEAR_M) / (FAR_M - NEAR_M)))
    hsv = np.uint8([[[round(60 * t), 255, 230]]])          # OpenCV hue: 0 red, 30 yellow, 60 green
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def _text(img, s, org, scale=0.45, col=(255, 255, 255), thick=1, centre=False, outline=False):
    (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = (org[0] - w // 2, org[1] + h // 2) if centre else org
    if outline:                                         # readable over the picture
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def _ms_between(a: Optional[str], b: Optional[str]) -> Optional[int]:
    try:
        return round((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() * 1000)
    except (TypeError, ValueError):
        return None


SHOW_W = 960                       # every camera picture is drawn this wide
_LAST_HW = [540, SHOW_W]


def render(meta: dict, img: Optional[np.ndarray], show_image: bool = True, show_numbers: bool = True,
           mode: str = "") -> np.ndarray:
    geo = {**DEFAULT_GEOMETRY, **(meta.get("geometry") or {})}
    fr = meta.get("frame") or {}
    if img is not None and img.shape[1] != SHOW_W:     # one size on screen: follow sends 1920 x 1080 frames,
        img = cv2.resize(img, (SHOW_W, round(img.shape[0] * SHOW_W / img.shape[1])),   # the stream 640 x 360
                         interpolation=cv2.INTER_AREA)
    if img is not None:
        _LAST_HW[:] = img.shape[:2]                     # a ToF-only upload is drawn at the last picture's size
    h, w = img.shape[:2] if img is not None else (_LAST_HW[0], _LAST_HW[1])
    rows = (meta.get("tof") or {}).get("rows")
    n = len(rows) if rows else 8
    zone = geo["tof_fov_deg"] / n
    f = (w / 2) / math.tan(math.radians(geo["camera_hfov_deg"] / 2))       # pinhole, square pixels
    pan_offset = float(geo.get("tof_pan_offset_deg", 0.0) or 0.0)
    tilt_offset = float(geo.get("tof_tilt_offset_deg", 0.0) or 0.0)

    def px(r, c):                                       # row 0 = top, col 0 = left, as the camera sees
        az = math.radians((c - (n - 1) / 2) * zone - pan_offset)
        el = math.radians(((n - 1) / 2 - r) * zone + tilt_offset)
        return w / 2 + f * math.tan(az), h / 2 - f * math.tan(el)

    radius = max(6, int(0.3 * (px(0, n // 2)[0] - px(0, n // 2 - 1)[0])))
    xs, ys = zip(*(px(r, c) for r in range(n) for c in range(n)))
    pad = int(max(0, -min(xs), max(xs) - w, -min(ys), max(ys) - h)) + radius + 4   # zones off the picture
    W, H = w + 2 * pad, h + 2 * pad
    canvas = np.full((HEADER + H + FOOTER, max(W, 560), 3), 32, np.uint8)
    ox, oy = (canvas.shape[1] - W) // 2 + pad, HEADER + pad
    pic = canvas[oy:oy + h, ox:ox + w]
    if img is not None and show_image:
        pic[:] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        pic[:] = 0
        _text(canvas, "no camera frame (ToF grid only)" if img is None else "image hidden (i)",
              (ox + w // 2, oy + h // 2), 0.6, (180, 180, 180), centre=True)
    cv2.rectangle(canvas, (ox - 1, oy - 1), (ox + w, oy + h), (90, 90, 90), 1)

    if rows:
        dots = canvas.copy()
        for r in range(n):
            for c in range(n):
                x, y = (round(v) for v in px(r, c))
                d = rows[r][c]
                if d is None:
                    cv2.circle(canvas, (ox + x, oy + y), radius, (150, 150, 150), 1, cv2.LINE_AA)
                else:
                    cv2.circle(dots, (ox + x, oy + y), radius, colour(d), -1, cv2.LINE_AA)
        mask = np.any(dots != canvas, axis=2)
        canvas[mask] = cv2.addWeighted(dots, 0.55, canvas, 0.45, 0)[mask]   # see the picture through them
        if show_numbers:
            for r in range(n):
                for c in range(n):
                    x, y = (round(v) for v in px(r, c))
                    d = rows[r][c]
                    _text(canvas, "--" if d is None else f"{d:.2f}", (ox + x, oy + y), 0.38, centre=True,
                          outline=True)

    # header: what this is and how old
    tof = meta.get("tof") or {}
    up = _ms_between(meta.get("ts"), meta.get("received"))
    shown = _ms_between(meta.get("received"), datetime.now(timezone.utc).isoformat())
    line1 = f"{mode}  session {meta.get('session')}  #{meta.get('n')}" + \
            (f"  frame {fr.get('seq')}  pan {fr.get('pan_deg')}  tilt {fr.get('tilt_deg')}" if fr else "  (no frame)")
    line2 = (f"ToF {tof['age_ms']:+d} ms vs image" if tof.get("age_ms") is not None else "no ToF grid") + \
            (f"   robot->Mac {up} ms" if up is not None else "") + \
            (f"   on screen {shown} ms after it arrived" if shown is not None and mode == "LIVE" else "")
    note = meta.get("note") or {}
    if note.get("task") == "follow":                 # what follow made of this frame
        line1 += (f"   follow #{note.get('seq')}: {note.get('state')} {note.get('target') or ''}"
                  f" bearing {note.get('bearing_deg')} dist {note.get('distance_m')} ({note.get('distance_from')})"
                  f" -> turn {note.get('turn_deg')} drive {note.get('drive_m')}"
                  + ("" if note.get("pose") else "  NO POSE"))
    ex = meta.get("extras") or {}
    if ex.get("pitch_deg") is not None:
        line2 += f"   pitch {ex['pitch_deg']:.1f} deg"
    if ex.get("pose"):
        x_, y_, th = ex["pose"]
        line2 += f"   pose {x_:.2f}, {y_:.2f} m, {math.degrees(th):.0f} deg"
    _text(canvas, line1, (8, 20))
    _text(canvas, line2, (8, 42), col=(200, 200, 200))

    # legend: the colour scale
    y0, x0, bw = HEADER + H + 10, 8, 200
    for i in range(bw):
        cv2.line(canvas, (x0 + i, y0), (x0 + i, y0 + 12), colour(NEAR_M + (FAR_M - NEAR_M) * i / bw), 1)
    _text(canvas, f"{NEAR_M} m", (x0 + bw + 8, y0 + 11), 0.4)
    _text(canvas, f"{FAR_M} m+   grey ring = no reading   q quit  space pause  n/p step  i image  d numbers  s save",
          (x0 + bw + 52, y0 + 11), 0.4, (200, 200, 200))
    return canvas


# --------------------------------------------------------------------------- nav: planning + obstacle avoidance
#
# A top-down panel of what the robot's nav stack (Robot-RPi-Core robot_core/nav) knows and decides,
# beside the camera + ToF view. It imports the robot's own nav code from the Robot-RPi-Core folder
# next to this one (or --robot-core), so what it shows is what the robot runs, not a copy:
#
#   --nav          recorded or live debug sessions: the costmap rebuilt from every uploaded ToF grid
#                  (placed with the head's pan/tilt and the robot's pose), the reflex's zone, what DWA
#                  and the local Theta* make of it toward the goal follow recorded, and what the robot
#                  itself reported (nav state, command, waypoints)
#   --sim NAME     a simulated room (box, door, trap, lookaway, sudden, follow): a tank with the real
#                  LocalNav (reflex, costmap, DWA, Theta*, gimbal aim) driving on a simulated ToF; the
#                  left panel is what its ToF sees, as the camera view shows a real one

SIM_SCENARIOS = {
    #  name       boxes (x0, x1, y0, y1, height m)                                          goal / person   note
    "box":      ([(1.0, 1.3, -0.3, 0.3, 0.4), (0, 5, 1.0, 1.2, 1.0)], (3.0, 0.0), "a box between it and the goal, a wall to the left"),
    "door":     ([(1.5, 1.6, -3, -0.375, 1.0), (1.5, 1.6, 0.375, 3, 1.0)], (3.0, 0.4), "a 0.75 m doorway in a wall"),
    "trap":     ([(1.4, 1.5, -0.6, 0.6, 0.5), (0.7, 1.5, 0.5, 0.6, 0.5), (0.7, 1.5, -0.6, -0.5, 0.5)], (2.6, 0.0),
                 "a U-shaped trap opening toward it, the goal behind"),
    "lookaway": ([(1.2, 1.5, -0.25, 0.25, 0.3)], (3.0, 0.0), "the head held on a 'person' far left: it must glance ahead"),
    "sudden":   ([], (4.0, 0.0), "something appears 0.36 m in front at t=3 s: the reflex brakes"),
    "follow":   ([(1.2, 1.6, -0.35, 0.35, 0.4), (0, 5, 1.6, 1.75, 1.0), (0, 5, -1.75, -1.6, 1.0)], None,
                 "following a person who walks round a box (perception replaced by the true position)"),
    # Nov 19 incidents (test_scenarios.py runs these headless with pass rules):
    "doorway":  ([(1.6, 1.7, -3, 0.7, 2.0), (1.6, 1.7, 1.5, 3, 2.0)], None,
                 "the person walks through a doorway off to the left and stops just behind the wall (the Mac knows the wall)"),
    "doorway2": ([(1.6, 1.7, -3, 0.7, 2.0), (1.6, 1.7, 1.5, 3, 2.0)], None,
                 "the same doorway, but the Mac has never seen the wall: only the robot's own ToF finds it"),
    "doorleaf": ([(1.0, 1.04, -0.05, 0.75, 2.0)], None,
                 "an open door leaf stands in the room; the person walks past behind it"),
    "corridor": ([(0.5, 4.0, 0.45, 0.6, 1.0), (0.5, 4.0, -0.6, -0.45, 1.0)], (3.6, 0.0), "a 0.9 m corridor to drive down"),
    "nook":     ([(-0.25, 0.30, 0.18, 0.28, 1.0), (-0.25, 0.30, -0.28, -0.18, 1.0), (0.30, 0.40, -0.28, 0.28, 1.0)],
                 (-1.5, 0.5), "it drove into a 0.36 m recess (too tight to spin in); the goal is behind it: reverse out"),
    "behind":   ([], (-3.0, 0.0), "the goal is 3 m straight behind it in an open room: spin round and go"),
    "cable":    ([(3.0, 3.1, -1.5, 1.5, 1.0)], (2.4, 0.0),
                 "a cable across the whole room: head-on the tracks spin (odometry counts on); cross it at an angle"),
    "wedge":    ([], (-1.5, 0.15), "a chair leg the ToF misses, in the corners' sweep: back off from it, then turn"),
    "ledge":    ([], (2.5, 0.0), "a ledge the ToF misses: the nose rides up 8 deg and it stops; back off and go round"),
    "glass":    ([], (2.2, 0.0), "a glass panel the ToF cannot see across the way: never bump it the same way twice"),
    # Nov 20 follow (debug/20261120-114136), walls as the maps had them:
    "door20":   ([(0.0, 1.15, -0.17, -0.07, 1.0), (1.1, 1.2, 0.43, 1.6, 1.0)], None,
                 "Nov 20: the person walks out of a door that maps 0.5 m (really 0.75 m): the robot fits, follow them"),
    "corner20": ([(1.9, 2.47, -0.45, -0.33, 1.0)], None,
                 "Nov 20: the person is round a wall corner 0.15 m to its right, 1.5 m on: go, do not swing"),
    "chair":    ([(1.45, 1.85, -0.45, -0.05, 0.45)], None,
                 "the person stands still beside a chair: stop near them, do not hit the chair"),
}
# Scripted people (waypoints, m/s, start s) for the follow scenarios; "follow" keeps its original walk.
PERSON_PATHS = {
    "follow":   ([(1.0, 0.0), (1.0, 0.9), (2.2, 0.9), (2.6, 0.0), (3.2, 0.0)], 0.18, 2.0),
    "doorway":  ([(1.1, 0.0), (1.3, 1.1), (2.0, 1.1), (2.1, 0.1)], 0.3, 2.0),   # stops 0.4 m behind the wall
    "doorway2": ([(1.1, 0.0), (1.3, 1.1), (2.0, 1.1), (2.1, 0.1)], 0.3, 2.0),
    "doorleaf": ([(1.5, 0.4), (1.5, 1.1), (2.2, 1.1), (2.6, 0.2)], 0.25, 2.0),
    "chair":    ([(1.75, 0.25)], 0.0, 0.0),
    "door20":   ([(1.0, 0.3), (1.3, 0.15), (1.85, 0.05), (2.3, 0.0)], 0.4, 0.0),
    "corner20": ([(2.55, -1.6)], 0.0, 0.0),
}
START = {"door20": (0.48, 0.54, -0.94), "corner20": (2.59, -0.14, -1.50)}   # where the robot was (x, y, heading)
KNOWN_MAP = {"doorway", "door20", "corner20"}   # the Mac's global map already has these walls (seen on earlier passes)
# Terrain the ToF does not see, by room: ("slip", x0, x1, y0, y1, normal_deg) a cable or rug edge the tracks spin on
# unless crossing 30 deg or more off square to it; ("leg", x, y) a 2 cm chair leg; ("step", x0, x1, y0, y1) a ledge the
# nose rides up on (pitch 8 deg) and stops; ("glass", x0, x1, y0, y1) solid, but the ToF sees through it. The odometry (encoders + gyro) counts on; the true pose does not move.
TERRAIN: dict = {
    "cable": [("slip", 0.95, 1.05, -1.5, 1.5, 0.0)],
    "wedge": [("leg", 0.047, 0.174)],
    "ledge": [("step", 0.9, 1.1, -0.5, 0.5)],
    "glass": [("glass", 1.0, 1.02, -0.6, 0.6)],
}
SEEN_START = {"nook": ((-1.6, 0.0), (-0.16, 0.16))}   # x and y ranges seen free at the start (it drove in through them)
NAV_PX = 520                       # the top-down panel's size in pixels (4 x 4 m)


def robot_core(path: Optional[Path]):
    """Import Robot-RPi-Core's nav code; exit with a hint if it is not there."""
    root = Path(path or os.environ.get("ROBOT_CORE_DIR") or Path(__file__).resolve().parent.parent / "Robot-RPi-Core")
    if not (root / "robot_core" / "nav").is_dir():
        sys.exit(f"nav modes need Robot-RPi-Core (with robot_core/nav) at {root}: use --robot-core")
    sys.path.insert(0, str(root))
    import types as _t
    from robot_core import terrain
    from robot_core.nav import dwa, plan
    from robot_core.nav.controller import LocalNav
    from robot_core.nav.costmap import RollingCostmap
    from robot_core.nav.reflex import Reflex
    from robot_core.sensors.head import Head, HeadGeometry, parse_self_zones
    from robot_core.sensors.tof8x8 import Grid, SIZE, zone_angles
    return _t.SimpleNamespace(root=root, terrain=terrain, dwa=dwa, plan=plan, LocalNav=LocalNav, Reflex=Reflex,
                              RollingCostmap=RollingCostmap, Head=Head, HeadGeometry=HeadGeometry, Grid=Grid,
                              parse_self_zones=parse_self_zones,
                              SIZE=SIZE, zone_angles=zone_angles)


def robot_params(root: Path) -> dict:
    """The numbers the nav stack uses, from robot.yaml (first occurrence of each key)."""
    keys = {"head_height_m": 0.22, "hfov_deg": 80.0, "pan_axis_x_m": 0.05, "head_tilt_offset_deg": 21.0,
            "tof_from_right_m": 0.10, "robot_width_m": 0.255, "robot_height_m": 0.24, "h_max_m": 0.03,
            "flat_tol_m": 0.015, "drive_tilt_deg": -30.0, "stop_short_m": 0.25, "plan_clearance_m": 0.08,
            "tof_pan_offset_deg": 0.0, "tof_tilt_offset_deg": 0.0}
    out = dict(keys)
    try:
        text = (root / "ros2_ws/src/robot_bringup/config/robot.yaml").read_text()
        for k in keys:
            import re
            m = re.search(rf"^\s+{k}:\s*([-+0-9.]+)", text, re.M)
            if m:
                out[k] = float(m.group(1))
        m = re.search(r'^\s+tof_self_zones:\s*"([^"]*)"', text, re.M)     # zones that see the robot itself
        out["tof_self_zones"] = m.group(1) if m else ""
    except OSError:
        pass
    return out


def make_head(rc, prm: dict, gimbal=None, lidar=None, pitch=None):
    p = rc.terrain.Params(width_m=prm["robot_width_m"], corridor_m=prm["robot_width_m"] + 0.05,
                          clearance_m=prm["plan_clearance_m"], body_m=prm["robot_height_m"], h_max_m=prm["h_max_m"],
                          flat_tol_m=prm["flat_tol_m"], stop_short_m=max(0.25, prm["stop_short_m"]))
    geom = rc.HeadGeometry(height_m=prm["head_height_m"], hfov_deg=prm["hfov_deg"], pan_axis_x_m=prm["pan_axis_x_m"],
                           drive_tilt_deg=prm["drive_tilt_deg"], tilt_offset_deg=prm["head_tilt_offset_deg"],
                           pan_axis_y_m=prm["tof_from_right_m"] - prm["robot_width_m"] / 2.0,
                           tof_pan_offset_deg=prm.get("tof_pan_offset_deg", 0.0),
                           tof_tilt_offset_deg=prm.get("tof_tilt_offset_deg", 0.0),
                           self_zones=rc.parse_self_zones(prm.get("tof_self_zones", "")))
    return rc.Head(gimbal, lidar, None, geom, pitch=pitch or (lambda: 0.0), terrain_params=p)


# ---- drawing --------------------------------------------------------------

def draw_nav(st: dict, size: int = NAV_PX) -> np.ndarray:
    """The top-down panel from a state dict (see NavReplay.state / NavSim.state). North (+y) up,
    the 4 x 4 m costmap window, the robot at its pose."""
    cm, pose = st["costmap"], st.get("pose")
    k = size / (cm.n * cm.res)                                   # px per metre
    to_px = lambda x, y: (int((x - cm.ox) * k), int(size - (y - cm.oy) * k))   # noqa: E731
    img = np.full((size, size, 3), 40, np.uint8)                 # unknown
    lo = cm.lo
    cell = np.zeros((cm.n, cm.n, 3), np.uint8)
    cell[:] = 40
    cell[lo < -0.3] = (78, 78, 78)                               # seen free
    infl = cm.clearance(*np.meshgrid(cm.ox + (np.arange(cm.n) + 0.5) * cm.res,
                                     cm.oy + (np.arange(cm.n) + 0.5) * cm.res, indexing="ij"))
    keep = st.get("radius", 0.19) + st.get("margin", 0.12)
    cell[(infl < keep) & (lo <= cm.OCC)] = (90, 55, 40)          # within the robot's radius + margin: blue tint
    occ = lo > cm.OCC
    shade = np.clip(140 + lo * 30, 140, 255).astype(np.uint8)
    cell[occ] = np.stack([shade[occ]] * 3, axis=1)               # occupied: brighter = surer
    img[:] = cv2.resize(np.flipud(cell.transpose(1, 0, 2)), (size, size), interpolation=cv2.INTER_NEAREST)
    for i in range(0, int(cm.n * cm.res) + 1):                  # 1 m grid
        gx = math.floor(cm.ox) + i
        gy = math.floor(cm.oy) + i
        cv2.line(img, (to_px(gx, 0)[0], 0), (to_px(gx, 0)[0], size), (55, 55, 55), 1)
        cv2.line(img, (0, to_px(0, gy)[1]), (size, to_px(0, gy)[1]), (55, 55, 55), 1)
    for x0, x1, y0, y1, _ in st.get("boxes", []):               # simulation: the real obstacles
        cv2.rectangle(img, to_px(x0, y1), to_px(x1, y0), (0, 140, 255), 1)
    for (x, y, d) in st.get("hits", []):                         # this ToF grid's readings
        cv2.circle(img, to_px(x, y), 3, colour(d), -1, cv2.LINE_AA)
    for path, col in ((st.get("global"), (255, 120, 40)), (st.get("local"), (255, 255, 0))):
        if path and len(path) > 1:
            cv2.polylines(img, [np.array([to_px(*p[:2]) for p in path], np.int32)], False, col, 1, cv2.LINE_AA)
    for xs, ys, ok in st.get("fan", []):                         # sampled DWA arcs
        cv2.polylines(img, [np.array([to_px(x, y) for x, y in zip(xs, ys)], np.int32)], False,
                      (60, 140, 60) if ok else (50, 50, 140), 1, cv2.LINE_AA)
    if st.get("arc") is not None:                                # the arc DWA chose
        xs, ys = st["arc"]
        cv2.polylines(img, [np.array([to_px(x, y) for x, y in zip(xs, ys)], np.int32)], False, (80, 255, 80), 2, cv2.LINE_AA)
    if st.get("goal"):
        g = to_px(*st["goal"][:2])
        cv2.drawMarker(img, g, (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
        if st.get("keep"):
            cv2.circle(img, g, int(st["keep"] * k), (200, 0, 200), 1, cv2.LINE_AA)
    if st.get("person"):
        cv2.circle(img, to_px(*st["person"]), 6, (255, 0, 255), -1, cv2.LINE_AA)
    if pose:
        x, y, th = pose
        c, s = math.cos(th), math.sin(th)
        if st.get("pan") is not None:                            # the ToF's view, as the head points
            a = th + math.radians(st["pan"])
            sx, sy = x + 0.05 * c, y + 0.05 * s
            pts = [to_px(sx, sy)] + [to_px(sx + 1.6 * math.cos(a + math.radians(b)), sy + 1.6 * math.sin(a + math.radians(b)))
                                     for b in (30, -30)]
            over = img.copy()
            cv2.fillPoly(over, [np.array(pts, np.int32)], (0, 200, 220))
            img[:] = cv2.addWeighted(over, 0.18, img, 0.82, 0)
        zone = [(0.0, 0.155), (st.get("stop_m", 0.25), 0.155), (st.get("stop_m", 0.25), -0.155), (0.0, -0.155)]
        cv2.polylines(img, [np.array([to_px(x + c * a - s * b, y + s * a + c * b) for a, b in zone], np.int32)], True,
                      (0, 0, 255) if st.get("tripped") else (0, 0, 150), 2 if st.get("tripped") else 1)
        fp = np.array([to_px(x + c * a - s * b, y + s * a + c * b)
                       for a, b in [(0.125, 0.125), (0.125, -0.125), (-0.125, -0.125), (-0.125, 0.125)]], np.int32)
        cv2.fillPoly(img, [fp], (70, 70, 70))
        cv2.polylines(img, [fp], True, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.arrowedLine(img, to_px(x, y), to_px(x + 0.25 * c, y + 0.25 * s), (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)
    y0 = 16
    for line in st.get("text", []):
        _text(img, line, (6, y0), 0.42, outline=True)
        y0 += 16
    _text(img, "occupied white  margin blue tint  Mac path blue  Pi Theta* cyan  DWA green  reflex red  sim walls orange",
          (6, size - 8), 0.34, (190, 190, 190), outline=True)
    return img


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    pad = lambda im: np.vstack([im, np.full((h - im.shape[0], im.shape[1], 3), 32, np.uint8)])   # noqa: E731
    return np.hstack([pad(left), np.full((h, 6, 3), 32, np.uint8), pad(right)])


def dwa_fan(rc, pose, v0, costmap, prm, n: int = 9):
    """Arcs at top speed across the turn rates, each marked admissible (keeps radius + margin) or not:
    which ways are open from here."""
    half = prm["robot_width_m"] / 2.0
    p = rc.dwa.DwaParams(radius_m=math.hypot(half, half), half_l_m=half, half_w_m=half)   # as LocalNav
    vs = np.full(n, p.max_v)
    ws = np.linspace(-p.max_w, p.max_w, n)
    xs, ys, ths = rc.dwa.arcs(pose[0], pose[1], pose[2], vs, ws, p.horizon_s, p.dt)
    clear = rc.dwa.rect_clearance(xs, ys, ths, costmap.clearance, p.half_l_m, p.half_w_m, p.side_pad_m).min(axis=1)
    return [(xs[i], ys[i], clear[i] >= p.margin_m) for i in range(n)], p


# ---- recorded sessions -----------------------------------------------------

class NavReplay:
    """Rebuilds the nav stack's view from a debug session, frame by frame."""

    def __init__(self, rc, prm: dict) -> None:
        self.rc, self.prm = rc, prm
        self.head = make_head(rc, prm)
        self.reset(None)

    def reset(self, session) -> None:
        p = self.head.terrain
        self.cm = self.rc.RollingCostmap(flat_tol_m=max(p.h_max_m, p.flat_tol_m), body_m=p.body_m)
        self.session, self.done, self.pan, self.tilt = session, -1, 0.0, 0.0
        self.last = {}

    def upto(self, data: Path, rel: str) -> dict:
        session = rel.split("/")[0]
        frames = session_frames(data, session)
        i = frames.index(rel) if rel in frames else len(frames) - 1
        if session != self.session or i < self.done:
            self.reset(session)
        for j in range(self.done + 1, i + 1):
            try:
                self.feed(json.loads((data / frames[j]).read_text()))
            except (OSError, ValueError):
                break
            self.done = j
        return self.last

    def feed(self, meta: dict) -> None:
        fr, ex, note = meta.get("frame") or {}, meta.get("extras") or {}, meta.get("note") or {}
        if meta.get("gimbal"):                                   # the head's angle at upload (newer robots)
            self.pan, self.tilt = meta["gimbal"]
        elif fr.get("pan_deg") is not None:                      # else the frame's; ToF-only: the last known
            self.pan, self.tilt = fr["pan_deg"], fr["tilt_deg"]
        pose, rows = ex.get("pose"), (meta.get("tof") or {}).get("rows")
        hits, tripped = [], False
        if pose and rows:
            zones = self.head.project(rows, self.pan, self.tilt, pitch_deg=ex.get("pitch_deg") or 0.0)   # as the robot:
            self.cm.integrate(pose, zones, self.head.sensor_xy)                                   # its own body left out
            c, s = math.cos(pose[2]), math.sin(pose[2])
            hits = [(pose[0] + c * z[2] - s * z[3], pose[1] + s * z[2] + c * z[3], z[5]) for z in zones if z[4] is not None]
            cmd = note.get("cmd") or (0.15, 0.0)
            r = self.rc.Reflex(self.head, None, lambda: tuple(cmd), lambda: pose)
            tripped = bool(r.danger(zones))
        st = {"costmap": self.cm, "pose": pose, "pan": self.pan, "hits": hits, "tripped": tripped,
              "wall": _epoch(meta.get("received")),
              "goal": note.get("goal") or self.last.get("goal"), "keep": 0.5,
              "global": note.get("waypoints"), "text": []}
        if pose:
            v0 = (note.get("cmd") or (0.0, 0.0))[0]
            st["fan"], p = dwa_fan(self.rc, pose, v0, self.cm, self.prm)
            st["radius"], st["margin"] = p.radius_m, p.margin_m
            goal = st["goal"]
            if goal:
                res = self.rc.dwa.plan((pose[0], pose[1], pose[2], v0, 0.0), goal[:2], self.cm.clearance, p, stop_at_m=0.5)
                xs, ys, _ = self.rc.dwa.arcs(pose[0], pose[1], pose[2], np.array([res.v]), np.array([res.w]), p.horizon_s, p.dt)
                st["arc"] = (xs[0], ys[0])
                st["local"] = self.rc.plan.local_path(self.cm, pose[:2], goal[:2], p.radius_m, p.margin_m)
                st["text"].append(f"viewer DWA: v {res.v:.2f} w {res.w:+.2f}{' BLOCKED' if res.blocked else ''}")
        if note.get("nav"):
            st["text"].append(f"robot: nav {note['nav']} {note.get('nav_detail') or ''}"[:70])
            if note.get("cmd"):
                st["text"].append(f"robot cmd: v {note['cmd'][0]:.2f} w {note['cmd'][1]:+.2f}  reflex trips {note.get('reflex_trips')}")
        st["text"].append(f"head pan {self.pan:.0f} tilt {self.tilt:.0f}" + ("   REFLEX ZONE HIT" if tripped else ""))
        self.last = st


def _epoch(iso: Optional[str]) -> Optional[float]:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() if iso else None
    except ValueError:
        return None


def _compose(a, b) -> tuple:
    c, s = math.cos(a[2]), math.sin(a[2])
    return a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]


class SlamLog:
    """The Mac's recorded computations for a session (debug/<session>/nav/nav.jsonl, written by
    Vision-And-Planning's nav_record.py), looked up by the Mac-clock time of an upload."""

    def __init__(self) -> None:
        self.folder, self.size, self.recs, self.walls = None, -1, [], []

    def load(self, data: Path, session: str) -> bool:
        folder = data / session / "nav"
        f = folder / "nav.jsonl"
        try:
            size = f.stat().st_size
        except OSError:
            self.folder, self.recs, self.walls = folder, [], []
            return False
        if folder != self.folder or size != self.size:          # new session, or it grew (live)
            recs = []
            for line in f.read_text().splitlines():
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass                                         # the line being written
            recs.sort(key=lambda r: r.get("wall", 0.0))
            self.folder, self.size, self.recs, self.walls = folder, size, recs, [r.get("wall", 0.0) for r in recs]
        return bool(self.recs)

    def at(self, wall: Optional[float]) -> Optional[dict]:
        """Everything known by `wall` (None: the end of the log)."""
        if not self.recs:
            return None
        import bisect
        upto = self.recs[:bisect.bisect_right(self.walls, wall)] if wall is not None else self.recs
        last = {}
        for r in upto:
            last[r["kind"]] = r
        corr = [r for r in upto if r["kind"] == "slam_correction"]
        poses = [r for r in upto if r["kind"] == "slam_pose"]
        frames = [r for r in upto if r["kind"] == "slam_frame"]
        o2m = tuple(corr[-1]["after"]) if corr else (0.0, 0.0, 0.0)
        plan = last.get("plan")
        maps = [r for r in upto if r["kind"] == "plan" and r.get("map")]
        gmap = None
        if maps:
            try:
                z = np.load(self.folder / maps[-1]["map"])
                gmap = {k: z[k] for k in z.files}
            except (OSError, ValueError):
                gmap = None
        return {
            "status": last.get("status"), "fit": last.get("slam_fit"), "frame": last.get("slam_frame"),
            "tof": last.get("slam_tof_scale"),
            "plan": plan, "map": gmap, "o2m": o2m, "newmaps": sum(r["kind"] == "slam_newmap" for r in upto),
            "slam_track": [r["base_in_map"][:2] for r in poses if r.get("base_in_map")],
            "odom_raw": [r["odom"][:2] for r in frames] or [r["pose"][:2] for r in upto if r["kind"] == "plan"],
            "odom_fixed": [_compose(tuple(c["after"]), c["odom"])[:2] for c in corr],
            "semantic": [r for r in upto if r["kind"] == "semantic" and (wall is None or wall - r["wall"] < 120)],
        }


def draw_global(sl: Optional[dict], pose_odom=None, size: int = NAV_PX) -> np.ndarray:
    """The Mac's view at this upload: its global map, plan, and visual SLAM against odometry."""
    img = np.full((size, size, 3), 34, np.uint8)
    if sl is None:
        _text(img, "no Mac nav record for this session", (10, 24), 0.5, outline=True)
        _text(img, "(recorded by Vision-And-Planning since 2026-11-19, while debug_mode is on)", (10, 44), 0.38,
              (190, 190, 190), outline=True)
        return img
    o2m = sl["o2m"]
    here = _compose(o2m, pose_odom) if pose_odom else None
    pts = [p for p in sl["slam_track"] + sl["odom_raw"] + sl["odom_fixed"]] + ([here[:2]] if here else [])
    plan = sl.get("plan") or {}
    pts += [p[:2] for p in plan.get("path_map") or []]
    g = sl.get("map")
    if g is not None:                                            # the known part of the map
        x0 = float(g["origin"]) + int(g["ix0"]) * float(g["res"])
        y0 = float(g["origin"]) + int(g["iy0"]) * float(g["res"])
        pts += [(x0, y0), (x0 + g["lo"].shape[0] * float(g["res"]), y0 + g["lo"].shape[1] * float(g["res"]))]
    if not pts:
        pts = [(0.0, 0.0)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    span = max(4.0, max(xs) - min(xs) + 1.0, max(ys) - min(ys) + 1.0)
    cx, cy = (max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0
    k = size / span
    to_px = lambda x, y: (int((x - cx) * k + size / 2), int(size / 2 - (y - cy) * k))   # noqa: E731
    if g is not None:
        lo, res = g["lo"].astype(np.float32) / 40.0, float(g["res"])
        cell = np.full(lo.shape + (3,), 34, np.uint8)
        cell[lo < float(g["free_t"])] = (78, 78, 78)
        cell[lo > float(g["occ_t"])] = (235, 235, 235)
        cell[g["sem"]] = (0, 140, 255)
        x0 = float(g["origin"]) + int(g["ix0"]) * res
        y1 = float(g["origin"]) + (int(g["iy0"]) + lo.shape[1]) * res
        tl = to_px(x0, y1)
        w, h = max(1, int(round(lo.shape[0] * res * k))), max(1, int(round(lo.shape[1] * res * k)))
        pic = cv2.resize(np.flipud(cell.transpose(1, 0, 2)), (w, h), interpolation=cv2.INTER_NEAREST)
        ax0, ay0, ax1, ay1 = max(0, tl[0]), max(0, tl[1]), min(size, tl[0] + w), min(size, tl[1] + h)
        if ax1 > ax0 and ay1 > ay0:
            img[ay0:ay1, ax0:ax1] = pic[ay0 - tl[1]:ay1 - tl[1], ax0 - tl[0]:ax1 - tl[0]]
    step = 1.0 if span < 10 else 2.0                             # metre grid
    for i in range(int(-span), int(span) + 1):
        gx, gy = math.floor(cx) + i * step, math.floor(cy) + i * step
        cv2.line(img, (to_px(gx, 0)[0], 0), (to_px(gx, 0)[0], size), (52, 52, 52), 1)
        cv2.line(img, (0, to_px(0, gy)[1]), (size, to_px(0, gy)[1]), (52, 52, 52), 1)
    fit = sl.get("fit")
    if fit and fit.get("slam_base"):                             # the fit's pairs: SLAM (fitted) -> odometry
        R, t, sc = np.array(fit["R"]), np.array(fit["t"]), fit["scale"]
        for b, o in zip(fit["slam_base"][::3], fit["odom_xy"][::3]):
            m = sc * R @ np.array(b) + t
            cv2.line(img, to_px(m[0], m[1]), to_px(o[0], o[1]), (120, 60, 160), 1, cv2.LINE_AA)
            cv2.circle(img, to_px(m[0], m[1]), 2, (200, 100, 255), -1)
    for track, col in ((sl["odom_raw"], (60, 60, 220)), (sl["odom_fixed"], (0, 220, 255)), (sl["slam_track"], (60, 220, 60))):
        if len(track) > 1:
            cv2.polylines(img, [np.array([to_px(*p[:2]) for p in track], np.int32)], False, col, 1, cv2.LINE_AA)
    for r in sl["semantic"]:
        cv2.circle(img, to_px(*r["xy"]), max(2, int(r["radius"] * k)), (0, 140, 255), 1, cv2.LINE_AA)
        _text(img, r["cls"], to_px(*r["xy"]), 0.32, (0, 170, 255), outline=True)
    if plan.get("path_map") and len(plan["path_map"]) > 1:
        cv2.polylines(img, [np.array([to_px(*p[:2]) for p in plan["path_map"]], np.int32)], False, (255, 120, 40), 2, cv2.LINE_AA)
    if plan.get("goal_map"):
        cv2.drawMarker(img, to_px(*plan["goal_map"]), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
    if here:
        x, y, th = here
        cv2.circle(img, to_px(x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.arrowedLine(img, to_px(x, y), to_px(x + 0.3 * math.cos(th), y + 0.3 * math.sin(th)), (255, 255, 255), 2,
                        cv2.LINE_AA, tipLength=0.35)
    st = (sl.get("status") or {}).get("vslam") or {}
    fr = sl.get("frame") or {}
    lines = []
    if st.get("backend", "none") == "none" and not fr:
        lines.append(f"SLAM NOT RUNNING: {st.get('reason') or 'no backend'}")
        lines.append("global map = odometry frame (odom_to_map stays 0)")
    else:
        lines.append(f"SLAM {st.get('backend', '?')}: {'TRACKING' if fr.get('tracking') else 'LOST'} ({fr.get('state')})")
        lines.append(f"frames tracked {fr.get('tracked', 0)}/{fr.get('frames_in', 0)}, skipped {fr.get('skipped', 0)}, "
                     f"new maps {sl['newmaps']}")
        lines.append(f"fit: {fit['scale']:.3f} m/unit from {fit.get('scale_source', 'odometry')}, residual "
                     f"{fit['residual_mean']:.3f} (max {fit['residual_max']:.3f}) m, n {fit['n']}, spread {fit['spread_m']:.1f} m"
                     if fit else "fit: not yet (needs 30 frames over 0.8 m)")
        tof = sl.get("tof")
        if tof:
            run = (f"{tof['scale']:.3f} m/unit" if tof.get("scale") else "not agreed yet") + \
                  f" over {tof['frames']} frames" + (f" (spread {tof['spread']:.1%})" if tof.get("spread") is not None else "")
            lines.append(f"ToF scale: {run}; this view {tof['samples']} flat-zone pts"
                         + (f" -> {tof['frame_scale']:.3f}" if tof.get("frame_scale") else ""))
        else:
            lines.append("ToF scale: no ToF with the frames (older robot code, or none fitted)")
    lines.append("odom_to_map: x %+.2f y %+.2f th %+.1f deg" % (o2m[0], o2m[1], math.degrees(o2m[2])))
    if plan:
        lines.append(f"Mac plan: {'ok' if plan.get('ok') else 'NO PATH'}  {plan.get('ms', 0):.0f} ms  margin "
                     f"{plan.get('margin_m')}  r {plan.get('radius_m')}  {len(plan.get('path_map') or [])} pts")
    y0 = 16
    for line in lines:
        _text(img, line, (6, y0), 0.42, outline=True)
        y0 += 16
    _text(img, "map white/grey  objects orange  plan blue  SLAM green  odometry red  corrected yellow  fit pairs purple",
          (6, size - 8), 0.34, (190, 190, 190), outline=True)
    return img


# ---- simulation ------------------------------------------------------------

class NavSim:
    """A tank in a box world driven by the robot's own LocalNav, on a simulated clock."""

    def __init__(self, rc, prm: dict, name: str) -> None:
        self.rc, self.prm, self.name = rc, prm, name
        boxes, goal, self.about = SIM_SCENARIOS[name]
        self.boxes = list(boxes)
        self.t, self.p, self.v, self.w, self.cmd, self.hits = 0.0, [0.0, 0.0, 0.0], 0.0, 0.0, (0.0, 0.0), 0
        self.o, self.gyro_w, self.pitch = [0.0, 0.0, 0.0], 0.0, 0.0      # odometry pose, gyro rate, IMU pitch
        if name in START:
            self.p, self.o = list(START[name]), list(START[name])
        self.terrain = TERRAIN.get(name, [])
        self.pan_tilt, self.goal_pt = [0.0, -21.0], [0.0, -21.0]
        self.grid = None
        sim = self

        class Gimbal:
            def move_to(self, pan=None, tilt=None, speed_dps=None, wait=False, timeout=10):
                if pan is not None:
                    sim.goal_pt[0] = max(-90.0, min(81.0, pan))
                if tilt is not None:
                    sim.goal_pt[1] = max(-30.0, min(45.0, tilt))
                if wait:
                    sim.pan_tilt = list(sim.goal_pt)
                return tuple(sim.goal_pt)

            def angles(self):
                return tuple(sim.pan_tilt)

            def wait_settled(self, *a, **k):
                sim.pan_tilt = list(sim.goal_pt)
                return True

        class Lidar:
            def grid(self, newer_than=None, timeout=1.0):
                return sim.grid

        class Motion:
            def velocity(self, v, w):
                sim.cmd = (v, w)

            def stop(self):
                sim.cmd, sim.v, sim.w = (0.0, 0.0), 0.0, 0.0
                return {"ok": True}

        self.head = make_head(rc, prm, Gimbal(), Lidar(), pitch=lambda: sim.pitch)
        planner = None
        vp = rc.root.parent / "Vision-And-Planning"
        try:                                                     # the Mac's global planner, in-process
            sys.path.insert(0, str(vp))
            from vision_service.nav_map import GlobalMap
            gm = GlobalMap(24.0, 0.05)

            if name in KNOWN_MAP:                                # walls the Mac already knows
                for x0, x1, y0, y1, h in self.boxes:
                    if h > 0.06:
                        for x in np.arange(x0, x1, 0.025):
                            for y in np.arange(y0, y1, 0.025):
                                gm.lo[gm.cell((x, y))] = gm.TOP

            def _plan(pose, goal, local, target=None):
                if local:
                    gm.fold_local(local, (0.0, 0.0, 0.0))
                return gm.plan(tuple(pose[:2]), tuple(goal), getattr(self.nav, "plan_radius_m", self.nav.params.radius_m),
                               target=None if target is None else tuple(target[:2]))
            planner = _plan
            self.gm = gm
        except Exception:                                        # noqa: BLE001 — no Vision-And-Planning: the Pi's own Theta*
            pass
        self.bump_t, self._pressed = 0.0, 0                      # the drivetrain's stall check: pressed 0.3 s
        self.nav = rc.LocalNav(self.head, Motion(), lambda: tuple(self.o), twist=lambda: (self.v, self.gyro_w),
                               planner=planner, bumped=lambda: self.bump_t, clock=lambda: 1000.0 + self.t)
        self.goal = goal
        if name in SEEN_START:                 # it drove in: the costmap saw the way free and the walls beside it
            (x0, x1), (y0, y1) = SEEN_START[name]
            cm = self.nav.costmap
            for x in np.arange(x0, x1, cm.res / 2):
                for y in np.arange(y0, y1, cm.res / 2):
                    cm.lo[cm._cells(x, y)] = cm.FREE - 1.0
            for bx0, bx1, by0, by1, _ in self.boxes:
                for x in np.arange(bx0, bx1, cm.res / 2):
                    for y in np.arange(by0, by1, cm.res / 2):
                        cm.lo[cm._cells(x, y)] = cm.OCC + 1.0
            cm._field = None
        self._tick_goal()
        if name == "lookaway":
            self.nav.set_focus((2.0, 2.5))

    def person(self) -> Optional[tuple]:
        if self.name not in PERSON_PATHS:
            return None
        pts, speed, t0 = PERSON_PATHS[self.name]
        if len(pts) == 1:
            return pts[0]
        s, acc = max(0.0, (self.t - t0) * speed), 0.0
        for a, b in zip(pts, pts[1:]):
            L = math.dist(a, b)
            if s <= acc + L:
                f = (s - acc) / L
                return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
            acc += L
        return pts[-1]

    def _tick_goal(self) -> None:
        if self.name in PERSON_PATHS:
            pp = self.person()
            self.nav.set_goal(pp, keep_m=0.5)
            self.nav.set_focus(pp)
        else:
            self.nav.set_goal(self.goal, keep_m=0.1)

    def _solid(self, x, y, z):
        out = z <= 0.0
        for x0, x1, y0, y1, h in self.boxes:
            out = out | ((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (z <= h))
        pp = self.person()
        if pp is not None:
            out = out | ((np.hypot(x - pp[0], y - pp[1]) < 0.15) & (z < 1.7))
        return out

    def _collides(self, x, y, th) -> bool:
        pts = np.array([(0.14, 0.13), (0.14, -0.13), (-0.14, 0.13), (-0.14, -0.13), (0.14, 0.0), (0.0, 0.13), (0.0, -0.13)])
        c, s = math.cos(th), math.sin(th)
        px, py = x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]
        glass = any(((px >= t[1]) & (px <= t[2]) & (py >= t[3]) & (py <= t[4])).any() for t in self.terrain if t[0] == "glass")
        return glass or bool(self._solid(px, py, np.full(len(pts), 0.02)).any())

    def _terrain(self, x, y, th, v, w) -> tuple:
        """The true (v, w) given what the tracks are doing (v, w) over invisible terrain; sets the pitch."""
        nx, ny, pitch = x + 0.14 * math.cos(th), y + 0.14 * math.sin(th), 0.0
        for t in self.terrain:
            if t[0] == "slip" and t[1] <= nx <= t[2] and t[3] <= ny <= t[4] and v > 0:
                off = abs((math.degrees(th) - t[5] + 90.0) % 180.0 - 90.0)     # degrees off square to the line
                if off < 30.0:
                    v = 0.0
            elif t[0] == "step" and t[1] <= nx <= t[2] and t[3] <= ny <= t[4]:
                pitch = 8.0
                v = min(v, 0.0)
        self.pitch = pitch
        return v, w

    def _leg_hit(self, x, y, th) -> bool:
        c, s = math.cos(th), math.sin(th)
        for t in self.terrain:
            if t[0] == "leg":
                dx, dy = t[1] - x, t[2] - y
                if abs(c * dx + s * dy) <= 0.15 and abs(-s * dx + c * dy) <= 0.14:
                    return True
        return False

    def _scan(self) -> None:
        g, rc = self.head.geom, self.rc
        pan, tilt = math.radians(self.pan_tilt[0]), math.radians(self.pan_tilt[1] + g.tilt_offset_deg)
        dirs = []
        for r in range(rc.SIZE):
            for c in range(rc.SIZE):
                up, left = rc.zone_angles(r, c)
                f, l, u = 1.0, math.tan(math.radians(left)), math.tan(math.radians(up))
                f, u = f * math.cos(tilt) - u * math.sin(tilt), f * math.sin(tilt) + u * math.cos(tilt)
                f, l = f * math.cos(pan) - l * math.sin(pan), f * math.sin(pan) + l * math.cos(pan)
                dirs.append((f, l, u))
        d = np.array(dirs)
        depth = np.arange(0.02, 3.5, 0.01)
        x0, y0, th = self.p
        bx = g.pan_axis_x_m + d[:, 0:1] * depth
        by = g.pan_axis_y_m + d[:, 1:2] * depth
        bz = g.height_m + d[:, 2:3] * depth
        wx, wy = x0 + bx * math.cos(th) - by * math.sin(th), y0 + bx * math.sin(th) + by * math.cos(th)
        hit = self._solid(wx, wy, bz)
        first = hit.argmax(axis=1)
        vals = np.where(hit.any(axis=1), depth[first], np.nan)
        rows = tuple(tuple(None if np.isnan(v) else float(v) for v in vals[r * 8:(r + 1) * 8]) for r in range(8))
        self.grid = rc.Grid(1000.0 + self.t, rows)

    def step(self, dt: float = 0.1) -> None:
        for _ in range(int(round(dt / 0.02))):
            k = int(round(self.t / 0.02))
            if k % 10 == 0:
                self._scan()
            if self.name == "sudden" and abs(self.t - 3.0) < 0.011:
                x, y = self.p[0], self.p[1]
                self.boxes.append((x + 0.36, x + 0.5, y - 0.3, y + 0.3, 0.5))
            self.nav.reflex.tick()
            if k % 2 == 0:
                self._tick_goal()
                self.nav.tick()
            v, w = self.cmd                                      # tracks with the velocity loop's ramps
            self.v += max(-0.008, min(0.008, v - self.v))
            self.w += max(-0.04, min(0.04, w - self.w))
            x, y, th = self.p
            tv, tw = self._terrain(x, y, th, self.v, self.w)      # what the body really does
            n = (x + tv * math.cos(th) * 0.02, y + tv * math.sin(th) * 0.02, th + tw * 0.02)
            if self._leg_hit(*n):                                # a thin leg holds it: the tracks slip on, the gyro is still
                self.hits += 1
                tv = tw = 0.0
            elif self._collides(*n):
                self.hits += 1
                self.v = self.w = tv = tw = 0.0
                self._pressed += 1
                if self._pressed == 15:                          # pressed 0.3 s: the drivetrain reports a stall
                    self.bump_t = 1000.0 + self.t
            else:
                self.p = list(n)
                self._pressed = 0
            self.gyro_w = tw                                     # odometry: encoder distance, gyro heading
            ox, oy, oth = self.o
            self.o = [ox + self.v * math.cos(oth) * 0.02, oy + self.v * math.sin(oth) * 0.02, oth + tw * 0.02]
            for i in range(2):                                   # the head slews at 120 deg/s
                self.pan_tilt[i] += max(-2.4, min(2.4, self.goal_pt[i] - self.pan_tilt[i]))
            self.t += 0.02

    def finished(self) -> bool:
        return (self.nav.state == "arrived" and abs(self.v) < 0.01 and self.name not in PERSON_PATHS) or self.t > 60.0

    def camera_meta(self) -> dict:
        return {"session": f"sim:{self.name}", "n": int(self.t * 10), "geometry": {"camera_hfov_deg": self.prm["hfov_deg"],
                "tof_fov_deg": 60.0}, "frame": {"seq": int(self.t * 10), "pan_deg": round(self.pan_tilt[0], 1),
                "tilt_deg": round(self.pan_tilt[1], 1), "w": 640, "h": 480},
                "tof": {"age_ms": 0, "rows": self.grid.rows if self.grid else None},
                "extras": {"pose": self.p, "pitch_deg": 0.0}}

    def state(self) -> dict:
        nav, pose = self.nav, tuple(self.p)
        hits = []
        if self.grid is not None:
            c, s = math.cos(pose[2]), math.sin(pose[2])
            for z in self.head.project(self.grid.rows, *self.pan_tilt, pitch_deg=0.0, min_range_m=0.02):
                if z[4] is not None:
                    hits.append((pose[0] + c * z[2] - s * z[3], pose[1] + s * z[2] + c * z[3], z[5]))
        fan, p = dwa_fan(self.rc, pose, self.v, nav.costmap, self.prm)
        xs, ys, _ = self.rc.dwa.arcs(pose[0], pose[1], pose[2], np.array([nav.cmd[0]]), np.array([nav.cmd[1]]),
                                     p.horizon_s, p.dt)
        goal = nav.goal
        dist = math.dist(goal, pose[:2]) if goal else 0.0
        return {"costmap": nav.costmap, "pose": pose, "pan": self.pan_tilt[0], "hits": hits, "boxes": self.boxes,
                "tripped": nav.reflex.tripped(), "goal": goal, "keep": nav.keep_m, "person": self.person(),
                "global": nav.waypoints if nav.source == "global" else None,
                "local": nav.local_wps if nav.source == "local" else None,
                "fan": fan, "arc": (xs[0], ys[0]), "radius": p.radius_m, "margin": p.margin_m,
                "text": [f"SIM {self.name}  t {self.t:5.1f} s   {self.about}"[:78],
                         f"nav {nav.state}: {nav.detail}"[:70],
                         f"cmd v {nav.cmd[0]:.2f} m/s  w {nav.cmd[1]:+.2f} rad/s   path {nav.source}   goal {dist:.2f} m",
                         f"reflex trips {nav.reflex.trips}   collisions {self.hits}   head pan {self.pan_tilt[0]:.0f}"]}


def run_sim(rc, prm: dict, name: str, snapshot: Optional[Path] = None, speed: float = 1.0) -> None:
    names = list(SIM_SCENARIOS)
    sim = NavSim(rc, prm, name)
    if snapshot:
        while not sim.finished():
            sim.step()
        cv2.imwrite(str(snapshot), side_by_side(render(sim.camera_meta(), None, mode="SIM"), draw_nav(sim.state())))
        print(f"wrote {snapshot}: {sim.name} ended {sim.nav.state} at t={sim.t:.1f} s, "
              f"{sim.hits} collisions, {sim.nav.reflex.trips} reflex trips")
        return
    win = "Monitoring-Service: nav simulation"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    playing = True
    while True:
        t0 = time.monotonic()
        if playing and not sim.finished():
            sim.step()
        view = side_by_side(render(sim.camera_meta(), None, mode="SIM" if playing else "SIM (paused)"), draw_nav(sim.state()))
        _text(view, "q quit  space pause  n step  r restart  1-6 scenario  s save", (8, view.shape[0] - 6), 0.4, (200, 200, 200))
        cv2.imshow(win, view)
        k = cv2.waitKeyEx(max(1, int((0.1 / speed - (time.monotonic() - t0)) * 1000)))
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            playing = not playing
        elif k == ord("n"):
            playing = False
            sim.step()
        elif k == ord("r"):
            sim = NavSim(rc, prm, sim.name)
        elif ord("1") <= k <= ord(str(len(names))):
            sim = NavSim(rc, prm, names[k - ord("1")])
        elif k == ord("s"):
            out = Path(f"view-sim-{sim.name}-{sim.t:.1f}s.png")
            cv2.imwrite(str(out), view)
            print(f"saved {out}")
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------- the window

def with_nav(view: np.ndarray, nav: "NavReplay", data: Path, rel: str, slam: Optional[SlamLog], show_slam=True):
    st = nav.upto(data, rel)
    view = side_by_side(view, draw_nav(st))
    if slam is not None and show_slam:
        found = slam.load(data, rel.split("/")[0])
        view = side_by_side(view, draw_global(slam.at(st.get("wall")) if found else None, st.get("pose")))
    return view


def steady(view: np.ndarray, size: list) -> np.ndarray:
    """The view on a canvas that only ever grows: uploads with and without a camera frame render at
    different sizes, and the window rescaled with every one of them."""
    size[0], size[1] = max(size[0], view.shape[0]), max(size[1], view.shape[1])
    out = np.full((size[0], size[1], 3), 32, np.uint8)
    out[:view.shape[0], :view.shape[1]] = view
    return out


def run(data: Path, session: Optional[str], nav: Optional["NavReplay"] = None) -> None:
    win = "Monitoring-Service: camera + ToF"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    live = session is None
    frames = session_frames(data, session) if session else []
    if session and not frames:
        sys.exit(f"no uploads in session {session} under {data}")
    idx, playing, show_img, show_num, show_nav = 0, live, True, True, True   # live plays; a replay starts paused
    slam, show_slam = SlamLog(), True
    canvas = [0, 0]
    current, last_key, next_step, view = None, None, time.monotonic(), None
    while True:
        if live:
            lt = latest(data)
            if lt and (lt["session"], lt["n"]) != last_key and playing:
                last_key = (lt["session"], lt["n"])
                current = lt["json"]
        elif playing and time.monotonic() >= next_step and idx < len(frames) - 1:
            idx, next_step = idx + 1, time.monotonic() + PLAY_S
        if not live:
            current = frames[idx]
        if current:
            try:
                meta, img = load(data, current)
                mode = ("LIVE" if playing else "PAUSED") if live else \
                       f"REPLAY {idx + 1}/{len(frames)}{'' if playing else ' (paused)'}"
                view = render(meta, img, show_img, show_num, mode)
                if nav is not None and show_nav:
                    view = with_nav(view, nav, data, current, slam, show_slam)
                view = steady(view, canvas)
                cv2.imshow(win, view)
            except (OSError, ValueError):
                pass                                    # being written right now: next poll
        else:
            cv2.imshow(win, render({"session": "-", "n": "-"}, None, mode=f"waiting for uploads in {data}"))
        k = cv2.waitKeyEx(100)
        right, left = k in (ord("n"), 83, 63235, 2555904), k in (ord("p"), 81, 63234, 2424832)
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            playing = not playing
        elif k in (ord("i"),):
            show_img = not show_img
        elif k in (ord("d"),):
            show_num = not show_num
        elif k == ord("m"):
            show_nav, canvas[:] = not show_nav, [0, 0]   # a panel on or off: the canvas may shrink
        elif k == ord("g"):
            show_slam, canvas[:] = not show_slam, [0, 0]
        elif k == ord("s") and view is not None:
            out = Path(f"view-{current.replace('/', '-').replace('.json', '')}.png")
            cv2.imwrite(str(out), view)
            print(f"saved {out}")
        elif (right or left) and current:            # n / p or the arrow keys (Linux, macOS, Windows)
            if live:                                    # step within the live session: pause first
                live, playing = False, False
                frames = session_frames(data, current.split("/")[0])
                idx = frames.index(current) if current in frames else len(frames) - 1
            step = 1 if right else -1
            idx = min(len(frames) - 1, max(0, idx + step))
    cv2.destroyAllWindows()


# =========================================================================== --report: a whole run on one page
# Everything a follow or explore run recorded, as one HTML page with the problems found and marked, so a run
# can be checked by eye instead of by reading logs. Written to reports/<session>.html (its map pictures in
# reports/<session>/); the camera frames are read from the debug folder where they are.
#
# At any moment of the run (slider, play, click the timeline or a problem):
#   camera    the frame with the ToF grid of that moment over it: each zone coloured by distance, outlined red
#             where the Pi's costmap counts it as an obstacle, green where floor, and moved by however far the
#             head turned between the grid and the frame. Follow's person boxes (thick: the one it picked).
#   map       the Mac's global map of that moment (black: wall, orange: YOLO marks, MAGENTA: wall cells on
#             ground the robot had already driven over - a ghost: legs, a sensor error or odometry drift),
#             the plan (red) and the straight line to its goal (dashed), the robot (body circle, heading,
#             ToF field of view), its trail, the person's track, the ToF hits of that grid and the last 3 s.
#   timeline  bands for follow state, nav state and ORB-SLAM3, and a tick for every problem.
# Problems found: the floor read as obstacles (a ToF height / IMU pitch check over the whole run, the Pi's
# costmap rule applied to returns near the floor plane), no path, detours (path much longer than the
# straight line), the path swapping sides within seconds, reflex stops, nav blocked, motion health (stalls,
# slips, ToF gaps), the person lost, ORB-SLAM3 losing track or crashing, explore steps that failed.
# Explore runs have no Mac global map: the map shows the ToF hits only.

# ToF geometry as robot.yaml has it (head_height_m, 8x8 zones over 60 deg, tof_self_zones, self_range_m)
HEAD_H, MAX_R, ZONE_DEG, SELF_M = 0.22, 2.6, 7.5, 0.15
SELF_ZONES = {(7, 0): 0.22, (7, 1): 0.15, (7, 2): 0.15, (7, 3): 0.15, (6, 0): 0.30}
_up = np.radians((3.5 - np.arange(8))[:, None] * ZONE_DEG * np.ones((1, 8)))
_left = np.radians((3.5 - np.arange(8))[None, :] * ZONE_DEG * np.ones((8, 1)))
_near = np.full((8, 8), SELF_M)
for (_rr, _cc), _m in SELF_ZONES.items():
    _near[_rr, _cc] = max(SELF_M, _m)

R_BODY = 0.177                  # nav's robot circle, m
DETOUR_RATIO, DETOUR_M = 1.3, 0.3    # nav_map.plan: a path longer than ratio * straight + m is the long way
SWAP_S, SWAP_M2 = 3.0, 0.05     # the path passes the other side of the straight line within this many s
GHOST_M = 0.10                  # a wall cell centre this close to where the robot drove is a ghost
FLAT_TOL, LOW, TOP = 0.045, 0.065, 0.25   # the Pi costmap's rule (robot.yaml flat_tol_m, costmap.LOW, robot height + 5 cm)
EXPLORE_BAD = ("unreachable", "stuck", "fail", "error", "backing off", "blocked", "drop", "timeout", "gave up")


def _iso(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def _r(v, n=2):
    return None if v is None else round(float(v), n)


def _apply(o2m, pose):
    """odom pose / point -> map frame."""
    c, s = math.cos(o2m[2]), math.sin(o2m[2])
    x, y = o2m[0] + c * pose[0] - s * pose[1], o2m[1] + s * pose[0] + c * pose[1]
    return (x, y, pose[2] + o2m[2]) if len(pose) > 2 else (x, y)


def _zones(rows, pan, tilt, pitch, mount_y):
    """Per zone: class (-1 no return, 3 too near / self, 0 floor, 1 obstacle, 2 above) and the
    hit (x ahead, y left) in the robot frame - head.project's geometry, and the Pi costmap's rule for
    an obstacle (costmap.integrate: higher than flat_tol above where the zone's lower edge meets the floor)."""
    d = np.array([[np.nan if v is None else v for v in row] for row in rows], float)
    fwd, left, up = np.ones((8, 8)), np.tan(_left), np.tan(_up)
    along = np.sqrt(1 + left ** 2 + up ** 2)
    t, p, b = math.radians(tilt), math.radians(pan), math.radians(pitch)
    fwd, up = fwd * math.cos(t) - up * math.sin(t), fwd * math.sin(t) + up * math.cos(t)
    fwd, left = fwd * math.cos(p) - left * math.sin(p), fwd * math.sin(p) + left * math.cos(p)
    fwd, up = fwd * math.cos(b) - up * math.sin(b), fwd * math.sin(b) + up * math.cos(b)
    h = HEAD_H + d * up
    cls = np.where(h - LOW * d * along <= FLAT_TOL, 0, np.where(h < TOP, 1, 2))
    cls = np.where(d < _near, 3, cls)
    cls = np.where(~np.isfinite(d) | (d > MAX_R), -1, cls)
    return cls, d * fwd, mount_y + d * left


class RunReport:
    def __init__(self, session: str, imgdir: str = ""):
        self.session, self.imgdir = session, imgdir        # imgdir: the session's folder seen from the page
        self.up = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(session, "[0-9]*.json")))]
        self.nav = self._jsonl("nav/nav.jsonl")
        self.fol = self._jsonl("nav/follow.jsonl")
        st = next((r for r in self.nav if r["kind"] == "status"), {})
        self.mount_y = float((st.get("mount") or [0, -0.025, 0])[1])
        self.plans = [r for r in self.nav if r["kind"] == "plan"]
        self._o2m = [(r["wall"], r.get("odom_to_map") or [0, 0, 0]) for r in self.plans]

    def _jsonl(self, rel):
        p = os.path.join(self.session, rel)
        return [json.loads(line) for line in open(p)] if os.path.exists(p) else []

    def o2m(self, t):
        cur = [0.0, 0.0, 0.0]
        for tt, v in self._o2m:                     # plans are in time order
            if tt > t:
                break
            cur = v
        return cur

    # ------------------------------------------------------------------ the recording
    def build(self, out_dir: str) -> dict:
        frames, events, grids, track, problems = [], [], [], [], []
        hfov = 80.0
        for d in self.up:
            hfov = float((d.get("geometry") or {}).get("camera_hfov_deg") or hfov)
            ex, f, g = d.get("extras") or {}, d.get("frame") or {}, d.get("gimbal")
            pitch = ex.get("pitch_deg") or 0.0
            if d.get("kind") == "event":
                n = d.get("note") or {}
                t = _iso(d["ts"])
                pose = n.get("pose_now") or ex.get("pose")
                events.append(dict(t=t, n=n, img=d.get("image"), g=g, pose=pose))
                if d.get("image"):
                    frames.append(dict(t=t, img=d["image"], pan=g[0] if g else 0.0, tilt=g[1] if g else 0.0,
                                       seq=n.get("seq"), ev=len(events) - 1))
                if pose:
                    track.append((t, *pose))
                continue
            pan, tilt = (f.get("pan_deg"), f.get("tilt_deg")) if f.get("pan_deg") is not None else (g or (None, None))
            if d.get("image"):
                t = f.get("t") or _iso(d["ts"])
                frames.append(dict(t=t, img=d["image"], pan=pan or 0.0, tilt=tilt or 0.0, seq=f.get("seq"), ev=None))
                if f.get("pose") or ex.get("pose"):
                    track.append((t, *(f.get("pose") or ex.get("pose"))))
            seq = d.get("tof_seq")
            if seq is None and (d.get("tof") or {}).get("rows"):
                seq = [dict(t=_iso(d["ts"]) - (d["tof"].get("age_ms") or 0) / 1000, rows=d["tof"]["rows"])]
            for s in seq or []:
                gp = s.get("pose") or ex.get("pose")       # explore: no reflex history -> the upload's pose
                gpan = s.get("pan_deg") if s.get("pan_deg") is not None else pan
                gtilt = s.get("tilt_deg") if s.get("tilt_deg") is not None else tilt
                if gp is None or gpan is None or gtilt is None:
                    continue
                grids.append(dict(t=s["t"], pose=gp, pan=gpan, tilt=gtilt, pitch=pitch, rows=s["rows"]))
                track.append((s["t"], *gp))
        times = [x["t"] for x in frames + events + grids] + [r["wall"] for r in self.nav]
        t0 = min(times) if times else 0.0
        T = lambda t: round(t - t0, 3)

        # robot track, in the map frame
        track.sort()
        trk = []
        for t, x, y, th in track:
            mx, my, mth = _apply(self.o2m(t), (x, y, th))
            trk.append([T(t), _r(mx, 3), _r(my, 3), _r(mth, 3)])

        # ToF grids: zone classes for the camera, hits in the map frame for the map
        G = []
        for gr in sorted(grids, key=lambda g: g["t"]):
            cls, x, y = _zones(gr["rows"], gr["pan"], gr["tilt"], gr["pitch"], self.mount_y)
            ok = (cls == 0) | (cls == 1)
            o2m = self.o2m(gr["t"])
            pose = _apply(o2m, gr["pose"])
            c, s = math.cos(pose[2]), math.sin(pose[2])
            wx, wy = pose[0] + c * x[ok] - s * y[ok], pose[1] + s * x[ok] + c * y[ok]
            G.append(dict(t=T(gr["t"]), pan=_r(gr["pan"], 1), tilt=_r(gr["tilt"], 1),
                          d=[_r(v) if v is not None else None for row in gr["rows"] for v in row],
                          c=cls.ravel().tolist(),
                          h=[v for a, b, k in zip(wx, wy, cls[ok]) for v in (_r(a), _r(b), int(k))]))

        calib = self._calib(grids)
        if calib.get("warn"):
            problems.append(("calib", 0.0, calib["warn"]))

        # follow's person boxes, by step seq
        boxes = {}
        for r in self.fol:
            if r.get("kind") == "follow_step" and r.get("persons") is not None:
                boxes[r["seq"]] = dict(wh=r.get("wh"), p=[[p["box"], p.get("face"), i == r.get("pick")]
                                                          for i, p in enumerate(r["persons"])])
        F = []
        for f in sorted(frames, key=lambda f: f["t"]):
            F.append(dict(t=T(f["t"]), img=f["img"], pan=_r(f["pan"], 1), tilt=_r(f["tilt"], 1),
                          ev=f["ev"], b=boxes.get(f["seq"]) if f["ev"] is not None else None))
        E = []
        for e in events:
            n = e["n"]
            tr = (n.get("tracker") or {}).get("xy") if isinstance(n.get("tracker"), dict) else None
            E.append(dict(t=T(e["t"]), n=n, img=e["img"],
                          person=list(map(_r, _apply(self.o2m(e["t"]), tr))) if tr else None))

        # plans, maps and their problems
        P, maps, prev, mp = [], {}, None, None
        ghosts_at = self._ghosts(trk, out_dir, maps, t0)
        for r in self.plans:
            path = r.get("path_map") or []
            st, gl = r.get("start_map"), r.get("goal_map")
            length = sum(math.dist(a, b) for a, b in zip(path, path[1:])) if len(path) > 1 else None
            straight = math.dist(st[:2], gl) if st and gl else None
            side = self._side(path, st, gl)
            mp = r.get("map") or mp                    # plans without a new snapshot use the last one
            p = dict(t=T(r["wall"]), path=[[_r(a), _r(b)] for a, b in path], start=st and [_r(v) for v in st[:2]],
                     goal=gl and [_r(v) for v in gl], target=r.get("target_map"), map=mp,
                     len=_r(length), straight=_r(straight), margin=r.get("margin_m"), door=r.get("door"))
            P.append(p)
            if not r.get("ok", True) or length is None:
                problems.append(("nopath", p["t"], f"no path ({_r(straight)} m to the goal)"))
            elif straight and length > straight * DETOUR_RATIO + DETOUR_M:
                g = ghosts_at.get(mp)
                near = self._near_line(g, st, gl) if g is not None else 0
                problems.append(("detour", p["t"], f"detour: {length:.1f} m path for {straight:.1f} m straight"
                                 + (f"; {near} ghost wall cells by the straight line" if near else "")))
            if prev and side and prev[1] and side * prev[1] < 0 and p["t"] - prev[0] < SWAP_S:
                problems.append(("swap", p["t"], "path swapped sides within %.1f s" % (p["t"] - prev[0])))
            if side:
                prev = (p["t"], side)

        # events: follow and explore
        last = {}
        for e in E:
            n, t = e["n"], e["t"]
            if n.get("task") == "explore":
                if any(k in (n.get("msg") or "").lower() for k in EXPLORE_BAD):
                    problems.append(("explore", t, n["msg"]))
                continue
            trips = n.get("reflex_trips") or 0
            if trips > last.get("trips", trips):
                problems.append(("reflex", t, f"reflex stop ({n.get('nav_detail') or n.get('nav')})"))
            last["trips"] = trips
            if n.get("nav") in ("blocked", "tripped") and last.get("nav") != n.get("nav"):
                problems.append(("blocked", t, f"nav {n['nav']}: {n.get('nav_detail') or ''}"))
            last["nav"] = n.get("nav")
            h = n.get("health") or ["normal", []]
            hs = h[0] + ": " + ", ".join(h[1]) if h and h[0] != "normal" else None
            if hs and (last.get("health") or "").split(" for ")[0] != hs.split(" for ")[0]:
                problems.append(("health", t, hs))
            last["health"] = hs
            if n.get("state") == "searching" and last.get("state") in ("tracking", "holding"):
                problems.append(("lost", t, f"person lost -> {n['state']}"))
            last["state"] = n.get("state")

        # ORB-SLAM3
        S, was = [], None
        for r in self.nav:
            if r["kind"] == "slam_frame":
                S.append([T(r["wall"]), 1 if r.get("tracking") else 0])
                if was and not r.get("tracking"):
                    problems.append(("orb", T(r["wall"]), "ORB-SLAM3 lost track"))
                was = r.get("tracking")
            elif r["kind"] == "slam_stopped":
                S.append([T(r["wall"]), -1])
                problems.append(("orb", T(r["wall"]), r.get("reason") or "ORB-SLAM3 stopped"))
        sem = [dict(t=T(r["wall"]), cls=r["cls"], xy=r["xy"], r=r.get("radius")) for r in self.nav if r["kind"] == "semantic"]

        problems.sort(key=lambda p: p[1])
        merged = []                                    # the same kind again within 2 s: one entry, counted
        for k, t, txt in problems:
            m = next((m for m in reversed(merged[-6:]) if m["k"] == k and t - m["t1"] < 2.0), None)
            if m:
                m["t1"], m["n"] = t, m["n"] + 1
            else:
                merged.append(dict(k=k, t=t, t1=t, n=1, txt=txt))
        return dict(session=os.path.basename(os.path.normpath(self.session)), imgdir=self.imgdir, hfov=hfov, mount_y=self.mount_y, calib=calib,
                    R=R_BODY, F=F, E=E, G=G, P=P, maps=maps, trk=trk, S=S, sem=sem, problems=merged,
                    dur=max([x["t"] for x in F + E + G + P] + [1.0]))

    @staticmethod
    def _calib(grids):
        """Is the floor where the ToF geometry says? Downward returns that lie near the floor plane:
        their height with the IMU pitch (what the Pi uses) and without, per column, and how many the Pi
        counts as obstacles. A floor that reads high turns into phantom obstacles in front of the robot."""
        rows = []
        for g in grids:
            d = np.array([[np.nan if v is None else v for v in r] for r in g["rows"]], float)
            a = np.degrees(_up) + g["tilt"]
            ok = np.isfinite(d) & (d >= _near) & (a < -12)
            if ok.any():
                rr, cc = np.nonzero(ok)
                rows += [(d[r, c], a[r, c], c, g["pitch"], math.sqrt(1 + math.tan(_left[r, c]) ** 2 + math.tan(_up[r, c]) ** 2))
                         for r, c in zip(rr, cc)]
        if len(rows) < 200:
            return {}
        d, a, c, p, al = np.array(rows).T
        h0 = HEAD_H + d * np.tan(np.radians(a))
        fl = np.abs(h0 - 0.03) < 0.05                       # near the floor plane, pitch left out
        h1 = HEAD_H + d * np.tan(np.radians(a + p))
        hit = (h1 - LOW * d * al > FLAT_TOL) & fl
        out = dict(pitch=_r(np.median(p), 1), floor_cm=_r(100 * np.median(h0[fl]), 1),
                   floor_pitch_cm=_r(100 * np.median(h1[fl]), 1), phantom_pct=_r(100 * hit.sum() / max(fl.sum(), 1), 0),
                   by_col=[_r(100 * hit[c == k].sum() / max((fl & (c == k)).sum(), 1), 0) for k in range(8)],
                   pitch_corr=_r(np.corrcoef(h1[fl], p[fl])[0, 1]))
        if out["phantom_pct"] >= 5:
            out["warn"] = (f"floor read as obstacle: {out['phantom_pct']:.0f}% of floor returns (by column, left to right: "
                           f"{', '.join('%.0f' % v for v in out['by_col'])}%). IMU pitch median {out['pitch']} deg on this "
                           f"floor; the floor reads {out['floor_pitch_cm']} cm high with it, {out['floor_cm']} cm without")
        return out

    @staticmethod
    def _side(path, st, gl):
        """Signed area between the path and the straight line: + left of it, - right, 0 straight."""
        if not st or not gl or len(path) < 3:
            return 0
        ax, ay = st[0], st[1]
        dx, dy = gl[0] - ax, gl[1] - ay
        a = sum(dx * (p[1] - ay) - dy * (p[0] - ax) for p in path) / max(math.hypot(dx, dy), 1e-6) * 0.05
        return 0 if abs(a) < SWAP_M2 else (1 if a > 0 else -1)

    @staticmethod
    def _near_line(cells, st, gl, within=0.35):
        if cells is None or not len(cells):
            return 0
        s, g = np.array(st[:2]), np.array(gl)
        dd = g - s
        tt = np.clip(((cells - s) @ dd) / max(dd @ dd, 1e-9), 0, 1)
        return int((np.linalg.norm(cells - (s + tt[:, None] * dd), axis=1) < within).sum())

    def _ghosts(self, trk, out_dir, maps, t0):
        """Each map snapshot as a PNG (1 px per cell) with its ghost cells in magenta; returns the ghost
        cell centres per map name, for the detour notes."""
        out = {}
        tr = np.array([[t + t0, x, y] for t, x, y, _ in trk]) if trk else np.zeros((0, 3))
        when = {}
        for r in self.plans:
            if r.get("map"):
                when.setdefault(r["map"], r["wall"])
        for name, t in when.items():
            p = os.path.join(self.session, "nav", name)
            if not os.path.exists(p):
                continue
            z = np.load(p)
            lo = z["lo"] / 40.0
            occ_t = float(z["occ_t"]) if "occ_t" in z.files else 0.6
            ix0, iy0, res, org = int(z["ix0"]), int(z["iy0"]), float(z["res"]), float(z["origin"])
            nx, ny = lo.shape
            occ = lo > occ_t
            driven = np.zeros_like(occ)
            past = tr[tr[:, 0] < t - 2.0] if len(tr) else tr
            k = int(math.ceil(GHOST_M / res))
            ii = np.floor((past[:, 1] - org) / res).astype(int) - ix0
            jj = np.floor((past[:, 2] - org) / res).astype(int) - iy0
            for a in range(-k, k + 1):
                for b in range(-k, k + 1):
                    if (a * a + b * b) * res * res <= GHOST_M ** 2:
                        ok = (ii + a >= 0) & (ii + a < nx) & (jj + b >= 0) & (jj + b < ny)
                        driven[ii[ok] + a, jj[ok] + b] = True
            ghost = occ & driven
            img = np.full((nx, ny, 3), 160, np.uint8)
            img[lo < -0.3] = (245, 245, 245)
            img[occ] = (0, 0, 0)
            img[z["sem"]] = (235, 140, 0)
            img[ghost] = (255, 0, 255)
            png = name.replace(".npz", ".png")
            cv2.imwrite(os.path.join(out_dir, png), cv2.cvtColor(np.ascontiguousarray(np.transpose(img, (1, 0, 2))[::-1]), cv2.COLOR_RGB2BGR))
            maps[name] = dict(png=os.path.basename(out_dir) + "/" + png, x0=org + ix0 * res, y0=org + iy0 * res, nx=nx, ny=ny, res=res)
            gi = np.argwhere(ghost)
            out[name] = org + (gi + [ix0, iy0] + 0.5) * res
        return out


PAGE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>navigation debugging</title>
<style>
body{font:13px -apple-system,system-ui,sans-serif;margin:0;background:#1d1f22;color:#ddd}
#top{display:flex;gap:8px;align-items:center;padding:6px 10px;background:#26292d}
button{background:#3a3f45;color:#eee;border:0;border-radius:4px;padding:4px 9px;cursor:pointer}
#slider{flex:1}
#tl{display:block;width:100%;height:74px;cursor:pointer;background:#222}
#main{display:flex;gap:8px;padding:8px}
.panel{background:#000;border-radius:4px;position:relative}
#hud{padding:6px 10px;font:12px ui-monospace,Menlo,monospace;white-space:pre;line-height:1.45}
#bottom{display:flex;gap:8px;padding:0 8px 8px}
#probs{flex:1;max-height:280px;overflow:auto;background:#26292d;border-radius:4px}
#probs div{padding:3px 8px;cursor:pointer;border-left:4px solid}
#probs div:hover,#probs div.on{background:#3a3f45}
#note{flex:1;max-height:280px;overflow:auto;background:#26292d;border-radius:4px;font:11px ui-monospace,Menlo,monospace;white-space:pre;padding:6px}
.lg{font-size:11px;color:#aaa;padding:2px 10px}
.lg b{display:inline-block;width:10px;height:10px;margin:0 3px 0 10px;vertical-align:-1px}
</style></head><body>
<div id="top"><b id="title"></b><button id="play">▶ play</button><select id="speed"><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select>
<button id="prev">◀ problem</button><button id="next">problem ▶</button><input id="slider" type="range" min="0" step="0.05"><span id="clock"></span></div>
<canvas id="tl"></canvas>
<div class="lg">bands: follow <b style="background:#2e8b57"></b>tracking<b style="background:#c9a227"></b>searching<b style="background:#7b5ea7"></b>holding ·
nav <b style="background:#2e8b57"></b>driving<b style="background:#d9534f"></b>tripped/blocked<b style="background:#3a7bd5"></b>arrived<b style="background:#555"></b>idle ·
ORB <b style="background:#2e8b57"></b>tracking<b style="background:#555"></b>not<b style="background:#d9534f"></b>dead · ticks: problems (colours below)</div>
<div id="main">
 <div><canvas id="cam" class="panel" width="720" height="405"></canvas><div id="hud"></div></div>
 <div><canvas id="map" class="panel" width="620" height="520"></canvas>
 <div class="lg">wheel: zoom · drag: pan · double-click: follow the robot again ·
 <b style="background:#000;border:1px solid #888"></b>wall<b style="background:#eb8c00"></b>YOLO mark<b style="background:#f0f"></b>ghost (on driven ground)
 <b style="background:#e33"></b>plan<b style="background:#2c2"></b>trail<b style="background:#39f"></b>person<b style="background:#f44"></b>ToF obstacle<b style="background:#4c4"></b>ToF floor</div></div>
</div>
<div id="bottom"><div id="probs"></div><div id="note"></div></div>
<script>
const D=/*DATA*/;
const KC={calib:'#ffffff',nopath:'#ff3b30',detour:'#ff9500',swap:'#ffcc00',reflex:'#ff2d92',blocked:'#d9534f',health:'#af52de',lost:'#5ac8fa',orb:'#8e8e93',explore:'#ff9500'};
const $=id=>document.getElementById(id);
let T=0,playing=false,view=null,imgs={},mapImgs={},probOn=-1;
$('title').textContent=D.session; $('slider').max=D.dur;
function last(a,t){let lo=0,hi=a.length-1,r=-1;while(lo<=hi){const m=(lo+hi)>>1;if(a[m].t<=t){r=m;lo=m+1}else hi=m-1}return r}
function lastArr(a,t){let lo=0,hi=a.length-1,r=-1;while(lo<=hi){const m=(lo+hi)>>1;if(a[m][0]<=t){r=m;lo=m+1}else hi=m-1}return r}
function nearest(a,t){const i=last(a,t);if(i<0)return a.length?0:-1;if(i+1<a.length&&a[i+1].t-t<t-a[i].t)return i+1;return i}
function img(src,cache){if(!cache[src]){const im=new Image();im.onload=()=>draw();im.src=src;cache[src]=im}return cache[src]}
function col(d){if(d==null)return null;const x=Math.max(0,Math.min(1,(d-0.15)/2.2));const h=x*230;return `hsla(${h},90%,50%,0.38)`}
function pose(t){const i=lastArr(D.trk,t);return i<0?null:D.trk[i]}
// ---------------------------------------------------------------- timeline
function drawTL(){const c=$('tl'),g=c.getContext('2d');c.width=c.clientWidth;const W=c.width,X=t=>t/D.dur*W;
 g.fillStyle='#222';g.fillRect(0,0,W,74);
 const band=(y,pts,cf)=>{for(let i=0;i<pts.length;i++){const a=pts[i],b=pts[i+1];const c2=cf(a);if(!c2)continue;g.fillStyle=c2;g.fillRect(X(a.t),y,Math.max(1,X(b?b.t:D.dur)-X(a.t)),10)}};
 band(4,D.E.filter(e=>e.n.task!='explore'),e=>({tracking:'#2e8b57',searching:'#c9a227',holding:'#7b5ea7'})[e.n.state]);
 band(16,D.E.filter(e=>e.n.task!='explore'),e=>({driving:'#2e8b57',tripped:'#d9534f',blocked:'#d9534f',arrived:'#3a7bd5',idle:'#555'})[e.n.nav]);
 band(28,D.S.map(s=>({t:s[0],v:s[1]})),s=>s.v>0?'#2e8b57':s.v<0?'#d9534f':'#555');
 g.fillStyle='#888';g.font='10px sans-serif';g.fillText('follow',2,12);g.fillText('nav',2,24);g.fillText('ORB',2,36);
 for(const e of D.E.filter(e=>e.n.task=='explore')){g.fillStyle='#bbb';g.fillRect(X(e.t),4,1,34)}
 for(const p of D.problems){g.fillStyle=KC[p.k]||'#fff';g.fillRect(X(p.t)-1,42,Math.max(3,X(p.t1)-X(p.t)+2),28)}
 g.fillStyle='#fff';g.fillRect(X(T)-1,0,2,74)}
$('tl').onclick=e=>seek(e.offsetX/$('tl').clientWidth*D.dur);
// ---------------------------------------------------------------- camera
function drawCam(){const c=$('cam'),g=c.getContext('2d');g.fillStyle='#000';g.fillRect(0,0,c.width,c.height);
 const fi=last(D.F,T);let hud=[];if(fi<0){$('hud').textContent='no frames yet';return}
 const F=D.F[fi],im=img(D.imgdir+F.img,imgs);
 if(im.complete&&im.naturalWidth){const k=Math.min(c.width/im.naturalWidth,c.height/im.naturalHeight);const w=im.naturalWidth*k,h=im.naturalHeight*k,ox=(c.width-w)/2,oy=(c.height-h)/2;
  g.drawImage(im,ox,oy,w,h);const f=(w/2)/Math.tan(D.hfov/2*Math.PI/180);
  const gi=nearest(D.G,F.t);
  if(gi>=0&&Math.abs(D.G[gi].t-F.t)<0.6){const G=D.G[gi],dp=G.pan-F.pan,dt=G.tilt-F.tilt;
   for(let r=0;r<8;r++)for(let cc=0;cc<8;cc++){const k2=r*8+cc,d=G.d[k2],cl=G.c[k2];
    const L=(3.5-cc)*7.5+dp,U=(3.5-r)*7.5+dt,R=Math.PI/180;
    const u0=ox+w/2-f*Math.tan((L+3.75)*R),u1=ox+w/2-f*Math.tan((L-3.75)*R),v0=oy+h/2-f*Math.tan((U+3.75)*R),v1=oy+h/2-f*Math.tan((U-3.75)*R);
    if(d!=null&&cl>=0){g.fillStyle=cl==3?'rgba(128,128,128,0.45)':col(d);g.fillRect(u0,v0,u1-u0,v1-v0)}
    g.strokeStyle=cl==1?'#ff3030':cl==0?'#30d030':cl==2?'#6af':'rgba(255,255,255,0.25)';g.lineWidth=cl==1||cl==0?2:1;g.strokeRect(u0+1,v0+1,u1-u0-2,v1-v0-2);
    g.fillStyle='#fff';g.font='11px Menlo,monospace';g.fillText(d==null?'·':d.toFixed(2),u0+3,v0+13)}
   hud.push(`ToF grid ${(G.t-F.t>=0?'+':'')+(G.t-F.t).toFixed(2)} s from the frame, head moved ${dp.toFixed(1)}° pan ${dt.toFixed(1)}° tilt between them`)}
  else hud.push('no ToF grid within 0.6 s of this frame');
  if(F.b&&F.b.wh){const sx=w/F.b.wh[0],sy=h/F.b.wh[1];for(const [b,face,pick] of F.b.p){g.strokeStyle=pick?'#39f':'#9cf';g.lineWidth=pick?4:1.5;g.strokeRect(ox+b[0]*sx,oy+b[1]*sy,(b[2]-b[0])*sx,(b[3]-b[1])*sy);if(face){g.fillStyle='#39f';g.fillText(face,ox+b[0]*sx+3,oy+b[1]*sy+14)}}}}
 const ei=last(D.E,T);
 hud.unshift(`t ${T.toFixed(2)} s   frame ${F.img}   head pan ${F.pan}° tilt ${F.tilt}°`);
 if(ei>=0){const E=D.E[ei],n=E.n;
  if(n.task=='explore')hud.push(`explore: ${n.msg}`);
  else{hud.push(`follow ${n.state}  distance ${n.distance_m==null?'-':n.distance_m.toFixed(2)} m (${n.distance_from||'-'})  bearing ${n.bearing_deg==null?'-':n.bearing_deg.toFixed(0)}°`);
   hud.push(`nav ${n.nav} ${n.nav_detail||''}  via ${n.nav_source}  cmd ${JSON.stringify(n.cmd)}  reflex trips ${n.reflex_trips}  health ${JSON.stringify(n.health)}`);
   const ex=D.E.slice(0,ei+1).reverse().find(e=>e.n.task=='explore');if(ex)hud.push(`explore: ${ex.n.msg}`)}
  $('note').textContent=`event at ${E.t.toFixed(2)} s\n`+JSON.stringify(n,null,1)}
 const pi=last(D.P,T);if(pi>=0){const P=D.P[pi];hud.push(`plan at ${P.t.toFixed(1)} s: ${P.len==null?'NO PATH':P.len.toFixed(2)+' m'} for ${P.straight} m straight, margin ${P.margin}${P.door?'  door '+JSON.stringify(P.door):''}`)}
 $('hud').textContent=hud.join('\n')}
// ---------------------------------------------------------------- map
function drawMap(){const c=$('map'),g=c.getContext('2d');g.fillStyle='#777';g.fillRect(0,0,c.width,c.height);
 const p=pose(T);if(!view)view={s:90,follow:true,cx:0,cy:0};if(view.follow&&p){view.cx=p[1];view.cy=p[2]}
 const X=x=>c.width/2+(x-view.cx)*view.s,Y=y=>c.height/2-(y-view.cy)*view.s;
 const pi=last(D.P,T),P=pi>=0?D.P[pi]:null,M=P&&D.maps[P.map];
 if(M){const im=img(M.png,mapImgs);if(im.complete){g.imageSmoothingEnabled=false;g.drawImage(im,X(M.x0),Y(M.y0+M.ny*M.res),M.nx*M.res*view.s,M.ny*M.res*view.s)}}
 // ToF hits: all so far faint (the only map in explore), last 3 s stronger, this grid bold
 const gi=last(D.G,T);
 for(let i=M?Math.max(0,gi-150):0;i<=gi;i++){const G=D.G[i],age=T-G.t,bold=i==gi;if(M&&age>3&&!bold)continue;
  for(let k=0;k<G.h.length;k+=3){const ob=G.h[k+2]==1;if(!M&&!ob&&age>3)continue;g.fillStyle=bold?(ob?'#f22':'#3c3'):age<=3?(ob?'rgba(255,60,60,0.45)':'rgba(60,200,60,0.35)'):'rgba(0,0,0,0.25)';const r=bold?3:2;g.fillRect(X(G.h[k])-r/2,Y(G.h[k+1])-r/2,r,r)}}
 for(const s of D.sem)if(s.t<=T&&T-s.t<120){g.strokeStyle='#eb8c00';g.beginPath();g.arc(X(s.xy[0]),Y(s.xy[1]),Math.max(3,(s.r||0.1)*view.s),0,7);g.stroke();g.fillStyle='#eb8c00';g.fillText(s.cls,X(s.xy[0])+5,Y(s.xy[1])-5)}
 // trail and person track
 g.fillStyle='#2c2';const ti=lastArr(D.trk,T);for(let i=Math.max(0,ti-6000);i<=ti;i+=3){const q=D.trk[i];g.fillRect(X(q[1])-1,Y(q[2])-1,2,2)}
 for(const e of D.E)if(e.person&&e.t<=T&&T-e.t<30){g.fillStyle=T-e.t<1?'#39f':'rgba(50,150,255,0.5)';g.beginPath();g.arc(X(e.person[0]),Y(e.person[1]),T-e.t<1?5:2.5,0,7);g.fill()}
 if(P){if(P.start&&P.goal){g.setLineDash([5,5]);g.strokeStyle='#fff';g.lineWidth=1;g.beginPath();g.moveTo(X(P.start[0]),Y(P.start[1]));g.lineTo(X(P.goal[0]),Y(P.goal[1]));g.stroke();g.setLineDash([])}
  if(P.path.length){g.strokeStyle='#e33';g.lineWidth=2.5;g.beginPath();P.path.forEach((q,i)=>i?g.lineTo(X(q[0]),Y(q[1])):g.moveTo(X(q[0]),Y(q[1])));g.stroke()}
  if(P.goal){g.strokeStyle='#f0f';g.lineWidth=2;g.beginPath();g.arc(X(P.goal[0]),Y(P.goal[1]),7,0,7);g.stroke()}
  if(P.target){g.strokeStyle='#39f';g.lineWidth=2;const u=X(P.target[0]),v=Y(P.target[1]);g.beginPath();g.moveTo(u-6,v-6);g.lineTo(u+6,v+6);g.moveTo(u+6,v-6);g.lineTo(u-6,v+6);g.stroke()}}
 // robot: body, heading, ToF field of view (from the head's pan axis)
 if(p){const [_,x,y,th]=p,G=gi>=0?D.G[gi]:null,pan=(G?G.pan:0)*Math.PI/180,R=D.R;
  const sx=x-Math.sin(th)*D.mount_y,sy=y+Math.cos(th)*D.mount_y;
  g.fillStyle='rgba(255,255,0,0.12)';g.beginPath();g.moveTo(X(sx),Y(sy));for(let a=-30;a<=30;a+=5){const b=th+pan+a*Math.PI/180;g.lineTo(X(sx+2.4*Math.cos(b)),Y(sy+2.4*Math.sin(b)))}g.fill();
  g.strokeStyle='#ff0';g.lineWidth=2;g.beginPath();g.arc(X(x),Y(y),R*view.s,0,7);g.stroke();
  g.beginPath();g.moveTo(X(x),Y(y));g.lineTo(X(x+R*1.4*Math.cos(th)),Y(y+R*1.4*Math.sin(th)));g.stroke()}
 g.fillStyle='#fff';g.fillText(`${(1/view.s*100).toFixed(1)} cm/px${M?'':'  (no Mac map in this run: ToF hits only)'}`,6,c.height-6)}
(()=>{const c=$('map');let drag=null;
 c.onwheel=e=>{e.preventDefault();view.s*=e.deltaY<0?1.15:1/1.15;draw()};
 c.onmousedown=e=>drag=[e.offsetX,e.offsetY,view.cx,view.cy];window.onmouseup=()=>drag=null;
 c.onmousemove=e=>{if(!drag)return;view.follow=false;view.cx=drag[2]-(e.offsetX-drag[0])/view.s;view.cy=drag[3]+(e.offsetY-drag[1])/view.s;draw()};
 c.ondblclick=()=>{view.follow=true;draw()}})();
// ---------------------------------------------------------------- problems, controls
const pl=$('probs');D.problems.forEach((p,i)=>{const d=document.createElement('div');d.style.borderColor=KC[p.k]||'#fff';
 d.textContent=`${p.t.toFixed(1)} s  ${p.k}  ${p.txt}${p.n>1?`  (×${p.n}, to ${p.t1.toFixed(1)} s)`:''}`;d.onclick=()=>{seek(Math.max(0,p.t-1.5));probOn=i};pl.appendChild(d)});
if(!D.problems.length)pl.textContent='no problems found';
function draw(){drawTL();drawCam();drawMap();$('slider').value=T;$('clock').textContent=`${T.toFixed(1)} / ${D.dur.toFixed(0)} s`;
 [...pl.children].forEach((d,i)=>d.classList.toggle('on',D.problems[i]&&D.problems[i].t<=T+2&&D.problems[i].t1>=T-2))}
function seek(t){T=Math.max(0,Math.min(D.dur,t));draw()}
$('slider').oninput=e=>{seek(+e.target.value)};
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'❚❚ pause':'▶ play';if(playing){let w=performance.now();const tick=n=>{if(!playing)return;T+=(n-w)/1000*+$('speed').value;w=n;if(T>=D.dur){T=D.dur;playing=false;$('play').textContent='▶ play'}draw();requestAnimationFrame(tick)};requestAnimationFrame(tick)}};
const jump=dir=>{const ps=D.problems;const i=dir>0?ps.findIndex(p=>p.t>T+1.6):ps.map(p=>p.t<T+1.4).lastIndexOf(true)-1;const p=ps[Math.max(0,i)];if(p)seek(Math.max(0,p.t-1.5))};
$('next').onclick=()=>jump(1);$('prev').onclick=()=>jump(-1);
document.onkeydown=e=>{if(e.key==' '){e.preventDefault();$('play').click()}else if(e.key=='ArrowRight'){const i=last(D.F,T);if(i+1<D.F.length)seek(D.F[i+1].t)}
 else if(e.key=='ArrowLeft'){const i=last(D.F,T-1e-3);if(i>=0)seek(D.F[Math.max(0,i)].t)}else if(e.key=='n')jump(1);else if(e.key=='p')jump(-1)};
window.onresize=draw;seek(0);
</script></body></html>
"""


def report(data: Path, session: Optional[str], open_it: bool = False) -> Path:
    """reports/<session>.html for a saved session (default: the newest); prints the problems found."""
    import webbrowser
    session = session or max((p.name for p in data.glob("2*") if p.is_dir()), default=None)
    if not session:
        sys.exit(f"no sessions in {data}")
    out = Path(__file__).resolve().parent / "reports"
    (out / session).mkdir(parents=True, exist_ok=True)
    src = data / session
    run = RunReport(str(src), os.path.relpath(src, out) + "/").build(str(out / session))
    page = out / f"{session}.html"
    page.write_text(PAGE.replace("/*DATA*/", json.dumps(run, separators=(",", ":"), default=str)))
    print(f"{page}: {len(run['F'])} frames, {len(run['G'])} ToF grids, {len(run['P'])} plans, {len(run['problems'])} problems")
    for p in run["problems"]:
        print(f"  {p['t']:7.1f} s  {p['k']:8s} {p['txt']}" + (f"  (x{p['n']} to {p['t1']:.1f} s)" if p["n"] > 1 else ""))
    if open_it:
        webbrowser.open(page.resolve().as_uri())
    return page


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=default_data_dir(), help="Vision-And-Planning debug folder")
    ap.add_argument("--session", help="replay a saved session (see --list)")
    ap.add_argument("--n", type=int, help="with --snapshot and --session: which upload")
    ap.add_argument("--list", action="store_true", help="list saved sessions and exit")
    ap.add_argument("--snapshot", type=Path, help="write the view to this PNG instead of opening a window")
    ap.add_argument("--nav", action="store_true", help="add the nav panel: costmap, reflex, DWA, Theta*")
    ap.add_argument("--sim", choices=list(SIM_SCENARIOS), help="simulate the nav stack in a room instead")
    ap.add_argument("--speed", type=float, default=1.0, help="--sim: playback speed")
    ap.add_argument("--robot-core", type=Path, help="Robot-RPi-Core folder (default: next to this one)")
    ap.add_argument("--report", action="store_true",
                    help="the whole session (--session, default the newest) on one HTML page with its problems marked")
    ap.add_argument("--open", action="store_true", help="--report: open the page in the browser")
    a = ap.parse_args()
    if a.list:
        return list_sessions(a.data)
    if a.report:
        return report(a.data, a.session, a.open)
    if a.sim or a.nav:
        rc = robot_core(a.robot_core)
        prm = robot_params(rc.root)
        if a.sim:
            return run_sim(rc, prm, a.sim, a.snapshot, a.speed)
        nav = NavReplay(rc, prm)
    else:
        nav = None
    if a.snapshot:
        if a.session:
            frames = session_frames(a.data, a.session)
            rel = next((f for f in frames if f.endswith(f"/{a.n:06d}.json")), None) if a.n else \
                (frames[-1] if frames else None)
        else:
            rel = (latest(a.data) or {}).get("json")
        if not rel:
            sys.exit(f"nothing to show in {a.data}")
        meta, img = load(a.data, rel)
        view = render(meta, img, mode="SNAPSHOT")
        if nav is not None:
            view = with_nav(view, nav, a.data, rel, SlamLog())
        cv2.imwrite(str(a.snapshot), view)
        print(f"wrote {a.snapshot}")
        return
    run(a.data, a.session, nav)


if __name__ == "__main__":
    main()
