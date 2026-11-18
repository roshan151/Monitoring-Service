#!/usr/bin/env python3
"""Show what the robot's sensors saw: its camera frame in black and white, with the
8x8 ToF grid laid over it as coloured dots (red = close, green = far), each labelled
with its range in metres, so you can check that the numbers match the picture.

It reads the files Vision-Microservice stores for the robot's debug mode
(robot.yaml /voice debug_mode: true); it makes no network calls and puts no load on
the robot. Default folder: ../Vision-Microservice/debug, or VS_DEBUG_DIR, or --data.

    python sensor_view.py                          live: the newest upload, as it arrives
    python sensor_view.py --list                   the saved sessions
    python sensor_view.py --session 20261118-104200   step through a saved session
    python sensor_view.py --snapshot view.png      the newest (or --session [--n N]) to a PNG

Keys: q / Esc quit · space pause / play · n or -> next · p or <- previous
      i image on / off · d range numbers on / off · s save the view as PNG
"""

from __future__ import annotations

import argparse
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
DEFAULT_GEOMETRY = {"camera_hfov_deg": 66.0, "tof_fov_deg": 60.0}
PLAY_S = 0.5                    # replay: seconds per frame (the robot sends at most 2 a second)


def default_data_dir() -> Path:
    env = os.environ.get("VS_DEBUG_DIR")
    return Path(env).expanduser() if env else Path(__file__).resolve().parent.parent / "Vision-Microservice" / "debug"


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


def render(meta: dict, img: Optional[np.ndarray], show_image: bool = True, show_numbers: bool = True,
           mode: str = "") -> np.ndarray:
    geo = {**DEFAULT_GEOMETRY, **(meta.get("geometry") or {})}
    fr = meta.get("frame") or {}
    h, w = (img.shape[:2] if img is not None else (fr.get("h", 480), fr.get("w", 640)))
    rows = (meta.get("tof") or {}).get("rows")
    n = len(rows) if rows else 8
    zone = geo["tof_fov_deg"] / n
    f = (w / 2) / math.tan(math.radians(geo["camera_hfov_deg"] / 2))       # pinhole, square pixels

    def px(r, c):                                       # row 0 = top, col 0 = left, as the camera sees
        az, el = math.radians((c - (n - 1) / 2) * zone), math.radians(((n - 1) / 2 - r) * zone)
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


# --------------------------------------------------------------------------- the window

def run(data: Path, session: Optional[str]) -> None:
    win = "Monitoring-Sensors: camera + ToF"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    live = session is None
    frames = session_frames(data, session) if session else []
    if session and not frames:
        sys.exit(f"no uploads in session {session} under {data}")
    idx, playing, show_img, show_num = 0, live, True, True      # live plays; a replay starts paused
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=default_data_dir(), help="Vision-Microservice debug folder")
    ap.add_argument("--session", help="replay a saved session (see --list)")
    ap.add_argument("--n", type=int, help="with --snapshot and --session: which upload")
    ap.add_argument("--list", action="store_true", help="list saved sessions and exit")
    ap.add_argument("--snapshot", type=Path, help="write the view to this PNG instead of opening a window")
    a = ap.parse_args()
    if a.list:
        return list_sessions(a.data)
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
        cv2.imwrite(str(a.snapshot), render(meta, img, mode="SNAPSHOT"))
        print(f"wrote {a.snapshot}")
        return
    run(a.data, a.session)


if __name__ == "__main__":
    main()
