#!/usr/bin/env python3
"""Replay recorded robot sessions through TODAY's code: a bug found once stays fixed.

Each incident becomes a saved scenario (a debug session, frozen) plus the rules it must keep,
instead of a dated comment in the code. No robot and no network: what the sensors and the Mac
said is taken from the recording, and the current decision code is run on it again:

  1. Mac range rules (Vision-And-Planning follow.FollowSession._refuse) on every recorded person:
     a ToF range from edge zones only that is nearer than their feet (a door frame), or one under
     0.35 m while their feet are further, must be refused.
  2. Pi tracker (Robot-RPi-Core follow.PersonTracker + refuse_range) fed the recorded answers, with
     the Mac's re-decided distances: the tracked person may not jump further than anyone can walk,
     nor be placed within 0.35 m of the robot while their feet said they were over 0.8 m away.
  3. Planning and steering: today's Mac planner re-run on the map it had at each recorded plan (with
     the person as target), then target_on_path on the costmap rebuilt from the recorded ToF grids:
     turning on the spot may not go left-right-left within 2 s (the Nov 19 doorway). Open loop: the
     recorded pose, not where today's code would have driven. The recorded behaviour is measured the
     same way for comparison.

    python3 test_replay.py                                      every saved scenario (scenarios/replay/)
    python3 test_replay.py --session 20261119-182630            one session straight from the debug folder
    python3 test_replay.py --save doorway --session 20261119-182630 --about "lost him through a doorway"

--save copies the session's uploads (JSON only, no pictures), nav/follow.jsonl, nav/nav.jsonl and the Mac's
saved maps (nav/map-*.npz) into
scenarios/replay/<name>/ with a scenario.json (what happened, the session, the limits). Results go to
test-runs/<time>.html like the other test scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

from session_checks import HERE, default_data_dir, load_session
from test_hardware import Results

SCEN = HERE / "scenarios" / "replay"
ROBOT = HERE.parent / "Robot-RPi-Core"
VISION = HERE.parent / "Vision-And-Planning"
DEFAULT_LIMITS = {"mac_bad_accepted": 0, "pi_goal_jumps": 0, "pi_placed_too_near": 0, "pivot_flip_episodes": 0}


def _imports():
    for p in (ROBOT, VISION):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


# --------------------------------------------------------------------------- 1. the Mac's range rules

def replay_mac(session) -> dict:
    """Re-decide every recorded ToF range on a person with today's FollowSession._refuse."""
    from vision_service.follow import FollowSession, Geometry
    fs = FollowSession()
    out = {"persons": 0, "ranges": 0, "refused_now": 0, "refused_then": 0, "mac_bad_accepted": 0, "bad": [],
           "redecided": {}}                                       # (run, seq) -> (distance, source) for the pick
    for ri, run in enumerate(session.runs):
        for r in run.steps:
            W, H = r.get("wh") or ((1920, 1080) if any(max(p["box"]) > 640 for p in r["persons"]) else (640, 480))
            geo = Geometry(tilt_deg=r.get("tilt") or 0.0, camera_height_m=r.get("cam_h") or 0.22,
                           hfov_deg=r.get("hfov") or 80.0)
            for i, p in enumerate(r.get("persons", [])):
                out["persons"] += 1
                if p.get("tof") is None:
                    continue
                out["ranges"] += 1
                zones = p.get("zones") or [None]
                why = fs._refuse(p["tof"], p["box"], H, geo, p.get("feet_m"), edge_only=zones[0] == 0)
                out["refused_now"] += why is not None
                out["refused_then"] += bool(p.get("tof_why"))
                d = p["tof"] * math.cos(math.radians(geo.tilt_deg))
                feet = p.get("feet_m")
                bad = feet is not None and why is None and (
                    (zones[0] == 0 and d < feet - 0.3) or (d < 0.35 and feet > 0.8))
                if bad:
                    out["mac_bad_accepted"] += 1
                    out["bad"].append({"run": ri + 1, "seq": r["seq"], "tof": round(d, 2), "feet": round(feet, 2)})
                if i == r.get("pick"):
                    if why is None:
                        out["redecided"][(ri, r["seq"])] = (round(d, 3), "tof")
                    elif feet is not None:
                        out["redecided"][(ri, r["seq"])] = (round(feet, 3), "floor")
                    else:
                        out["redecided"][(ri, r["seq"])] = (None, None)
    return out


# --------------------------------------------------------------------------- 2. the Pi's tracker

def replay_pi(session, redecided: dict) -> dict:
    from robot_core.follow import WALK_MAX_MPS, PersonTracker, refuse_range
    out = {"updates": 0, "refused": 0, "pi_goal_jumps": 0, "pi_placed_too_near": 0, "jumps": []}
    for ri, run in enumerate(session.runs):
        feet_by_seq = {}
        for r in run.steps:
            p = r["persons"][r["pick"]] if r.get("pick") is not None and r["pick"] < len(r["persons"]) else None
            feet_by_seq[r["seq"]] = p.get("feet_m") if p else None
        tr = PersonTracker()
        for wall, n in run.events:
            here = n.get("pose")
            if n.get("state") != "tracking" or here is None or n.get("bearing_deg") is None or wall is None:
                continue
            d, src = n.get("distance_m"), n.get("distance_from")
            if (ri, n["seq"]) in redecided:                          # today's Mac would have said this instead
                d, src = redecided[(ri, n["seq"])]
            if not d or src == "near":
                continue
            if refuse_range(tr, here, d, src):
                out["refused"] += 1
                continue
            before, tb = tr.xy, tr.t
            tr.update(here, n["bearing_deg"], d, None, wall)
            out["updates"] += 1
            if before is not None:
                jump, dt = math.dist(before, tr.xy), max(0.0, tr.t - tb)
                if jump > WALK_MAX_MPS * dt + 0.5:
                    out["pi_goal_jumps"] += 1
                    out["jumps"].append({"run": ri + 1, "seq": n["seq"], "jump_m": round(jump, 2), "dt_s": round(dt, 2)})
            feet = feet_by_seq.get(n["seq"])
            if feet and feet > 0.8 and math.dist(here[:2], tr.xy) < 0.35:
                out["pi_placed_too_near"] += 1
    return out


# --------------------------------------------------------------------------- 3. planning and steering

def _load_map(path: Path):
    """A GlobalMap as Vision-And-Planning saved it at plan time (nav_record.save_map)."""
    import time as _t
    import numpy as np
    from vision_service.nav_map import GlobalMap
    z = np.load(path)
    gm = GlobalMap(float(z["n"]) * float(z["res"]), float(z["res"]))
    x0, y0 = int(z["ix0"]), int(z["iy0"])
    lo = z["lo"].astype(np.float32) / 40.0
    gm.lo[x0:x0 + lo.shape[0], y0:y0 + lo.shape[1]] = lo
    gm.sem_t[x0:x0 + lo.shape[0], y0:y0 + lo.shape[1]] = np.where(z["sem"], _t.monotonic(), 0.0)
    return gm


def _flip_episodes(seq) -> int:
    """(wall, turn direction 0/-1/+1) in order -> left-right-left turns on the spot within 2 s."""
    flips, sign, episodes = [], 0, 0
    for wall, s in seq:
        if not s or wall is None:
            continue
        if sign and s != sign:
            flips = [t for t in flips if wall - t <= 2.0] + [wall]
            if len(flips) >= 2:
                episodes, flips = episodes + 1, []
        sign = s
    return episodes


def replay_target(session, data: Path) -> dict:
    """The whole chain at every recorded step, open loop (the robot's recorded pose and goal):

      then  the recorded Mac plan and the steering target as it was picked then (unknown cells free),
            as it was.
      now   today's Mac planner re-run on the map it had at that moment (nav/map-*.npz) with the
            person as `target` and the robot's plan radius, then target_on_path with seen_fn on the
            costmap rebuilt from the recorded ToF grids.
    Counts left-right-left turns on the spot for each."""
    import navigation_debugging as T
    from robot_core.nav.controller import PLAN_EXTRA_M
    from vision_service.nav_map import apply, inverse
    rc = T.robot_core(ROBOT)
    prm = T.robot_params(rc.root)
    nr = T.NavReplay(rc, prm)
    radius = rc.dwa.DwaParams().radius_m + rc.dwa.DwaParams().margin_m
    plan_radius = radius + PLAN_EXTRA_M                      # what today's robot asks the planner for (LocalNav.plan_radius_m)
    plans = [r for r in session.nav if r.get("kind") == "plan"]
    maps = {r["wall"]: session.folder / "nav" / r["map"] for r in plans if r.get("map")}
    events = []                                              # (wall, meta) of every upload, in order
    for f in sorted(session.folder.glob("[0-9]*.json")):
        try:
            meta = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        events.append((T._epoch(meta.get("received")), meta))
    out = {"steps": 0, "replanned": 0, "plan_changed_end": 0}
    then, now = [], []
    cur, cur_goal, pi, gm_cache = None, None, 0, {}
    last_goal = None
    for wall, meta in events:
        nr.feed(meta)
        n = meta.get("note") if isinstance(meta.get("note"), dict) else None
        if n and n.get("task") == "follow" and n.get("goal"):
            last_goal = n["goal"]
        while pi < len(plans) and wall is not None and plans[pi]["wall"] <= wall:   # plans made before this upload
            r = plans[pi]
            pi += 1
            mk = max((w for w in maps if w <= r["wall"]), default=None)
            if mk is None or last_goal is None:
                continue
            if mk not in gm_cache:
                gm_cache = {mk: _load_map(maps[mk])}
            gm = gm_cache[mk]
            o2m = r.get("odom_to_map") or [0.0, 0.0, 0.0]
            tgt_map = apply(o2m, [last_goal[:2]])[0]
            path = gm.plan(tuple(r["start_map"][:2]), tuple(r["goal_map"][:2]), plan_radius,
                           0.05, 2.0, (float(tgt_map[0]), float(tgt_map[1])))
            out["replanned"] += 1
            if not path:
                continue
            path = [tuple(p) for p in apply(inverse(o2m), path).tolist()]
            if r.get("path_odom") and math.dist(path[-1], r["path_odom"][-1]) > 0.3:
                out["plan_changed_end"] += 1
            cur, cur_goal = path, tuple(last_goal[:2])
        if not n or n.get("task") != "follow":
            continue
        here, goal = n.get("pose_now") or n.get("pose"), n.get("goal")
        if not (here and goal):
            continue
        out["steps"] += 1
        for key, path, seen in (("then", n.get("waypoints"), None),
                                ("now", cur, lambda a, b: nr.cm.known_free_line(a, b))):
            if not path:
                continue
            tgt = rc.dwa.target_on_path(here, path, nr.cm.clearance, radius, seen_fn=seen)
            if tgt is None or math.dist(tgt, goal) < 0.1:
                tgt = goal
            tb = math.atan2(tgt[1] - here[1], tgt[0] - here[0]) - here[2]
            tb = math.atan2(math.sin(tb), math.cos(tb))
            if abs(tb) <= math.radians(28.0):
                (then if key == "then" else now).append((wall, 0))
                continue
            (then if key == "then" else now).append((wall, 1 if tb > 0 else -1))
    out["pivot_flip_episodes_then"] = _flip_episodes(then)
    out["pivot_flip_episodes"] = _flip_episodes(now)
    return out


# --------------------------------------------------------------------------- running

def run_one(res: Results, name: str, data: Path, session: str, limits: dict, about: str = "",
            known: dict | None = None) -> dict:
    res.mode = name
    res.info[name] = (f"Replay: {name}", about or f"debug session {session}, run again through today's code")
    known = known or {}
    s = load_session(data, session)
    check = res.check

    def note(key):
        return f" | known: {known[key]}" if key in known else ""
    if not s.runs:
        res.skipped[name] = "no follow steps in this session"
        return {}
    _imports()
    found = {}
    try:
        mac = replay_mac(s)
        found["mac"] = {k: v for k, v in mac.items() if k != "redecided"}
        check("the Mac refuses door-frame and too-near ToF ranges on a person",
                  mac["mac_bad_accepted"] <= limits["mac_bad_accepted"],
                  f"{mac['ranges']} ranges re-decided: {mac['refused_now']} refused now (then {mac['refused_then']}); "
                  f"bad ones still taken: {mac['mac_bad_accepted']}" + (f" e.g. {mac['bad'][:3]}" if mac["bad"] else "")
                  + note("mac_bad_accepted"))
    except Exception as exc:                                       # noqa: BLE001
        mac = {"redecided": {}}
        res.check("the Mac refuses door-frame and too-near ToF ranges on a person", False,
                  f"could not run Vision-And-Planning's follow code: {type(exc).__name__}: {exc}")
    try:
        pi = replay_pi(s, mac.get("redecided", {}))
        found["pi"] = pi
        res.check("the tracked person never jumps further than anyone can walk", pi["pi_goal_jumps"] <= limits["pi_goal_jumps"],
                  f"{pi['updates']} tracker updates, {pi['refused']} ranges refused by the Pi; jumps: {pi['pi_goal_jumps']}"
                  + (f" e.g. {pi['jumps'][:3]}" if pi["jumps"] else "") + note("pi_goal_jumps"))
        res.check("the person is never placed within 0.35 m while their feet are over 0.8 m away",
                  pi["pi_placed_too_near"] <= limits["pi_placed_too_near"], f"{pi['pi_placed_too_near']} times"
                  + note("pi_placed_too_near"))
    except Exception as exc:                                       # noqa: BLE001
        res.check("the Pi's tracker replays", False, f"{type(exc).__name__}: {exc}")
    try:
        tg = replay_target(s, data)
        found["target"] = tg
        res.check("planning and steering never make it turn left-right-left on the spot",
                  tg["pivot_flip_episodes"] <= limits["pivot_flip_episodes"],
                  f"{tg['steps']} steps, {tg['replanned']} plans re-made on the recorded maps "
                  f"({tg['plan_changed_end']} now end elsewhere): "
                  f"{tg['pivot_flip_episodes']} flip episodes now, {tg['pivot_flip_episodes_then']} as recorded"
                  + note("pivot_flip_episodes"))
    except Exception as exc:                                       # noqa: BLE001
        res.check("the steering target replays", False, f"{type(exc).__name__}: {exc}")
    return found


def save(data: Path, session: str, name: str, about: str) -> Path:
    src, dst = data / session, SCEN / name
    if not src.is_dir():
        raise SystemExit(f"no session {session} in {data}")
    (dst / session / "nav").mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.glob("[0-9]*.json")):
        shutil.copy2(f, dst / session / f.name)
        n += 1
    for f in ("follow.jsonl", "nav.jsonl"):
        if (src / "nav" / f).exists():
            shutil.copy2(src / "nav" / f, dst / session / "nav" / f)
    for f in sorted((src / "nav").glob("map-*.npz")):          # the Mac's map at plan time: the planner replays on it
        shutil.copy2(f, dst / session / "nav" / f.name)
    (dst / "scenario.json").write_text(json.dumps({
        "about": about, "session": session, "saved": datetime.now().isoformat(timespec="seconds"),
        "limits": DEFAULT_LIMITS}, indent=2))
    print(f"saved {n} uploads of {session} to {dst} (no pictures); edit {dst / 'scenario.json'} to change its limits")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=default_data_dir())
    ap.add_argument("--session", help="replay this debug session instead of the saved scenarios")
    ap.add_argument("--save", metavar="NAME", help="freeze --session as scenarios/replay/NAME")
    ap.add_argument("--about", default="", help="with --save: what happened in it, in a sentence")
    a = ap.parse_args()
    if a.save:
        if not a.session:
            raise SystemExit("--save needs --session")
        save(a.data, a.session, a.save, a.about)
        return
    res = Results({}, "Replay test: recorded sessions through today's code")
    found = {}
    if a.session:
        found[a.session] = run_one(res, a.session, a.data, a.session, DEFAULT_LIMITS)
    else:
        scen = sorted(SCEN.glob("*/scenario.json")) if SCEN.exists() else []
        if not scen:
            raise SystemExit(f"no saved scenarios in {SCEN}: save one with --save NAME --session S")
        for sj in scen:
            sc = json.loads(sj.read_text())
            found[sj.parent.name] = run_one(res, sj.parent.name, sj.parent, sc["session"],
                                            {**DEFAULT_LIMITS, **sc.get("limits", {})}, sc.get("about", ""),
                                            sc.get("known"))
    res.data = found
    res.report()


if __name__ == "__main__":
    main()
