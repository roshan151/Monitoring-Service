#!/usr/bin/env python3
"""Measure a recorded robot session: the numbers scorecard.py shows and test_replay.py checks.

A session is what Vision-And-Planning stores while the robot runs with debug_mode on:

    debug/<session>/000001.json ...     the robot's uploads; follow steps are "kind": "event" with a
                                        "note" (state, distances, nav state, command, tracker, rules...)
    debug/<session>/nav/follow.jsonl    the Mac's working per follow step (every person in view, their
                                        ToF range and why it was refused, who was picked and why)
    debug/<session>/nav/nav.jsonl       the Mac's plans and visual SLAM records

Everything is measured from those files alone (no robot, no network), so an old session and a new
one are measured the same way and can be compared. Each measure is reconstructed from raw fields
where it can be (a goal jump from the tracked positions, a pivot flip from the commands), so it
works on sessions recorded before the robot counted its own rule breaks; when the robot did count
them (note["rules"], Nov 19 evening on) those counts are reported too.

    from session_checks import load_session, measure
    s = load_session(data_dir, "20261119-182630")
    m = measure(s)          # {"follow": {...}, "nav": {...}, "sensors": {...}, "mac": {...}, "runs": [...]}

python3 session_checks.py [--session S] prints the numbers as JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
KEEP_M = 0.5                    # follow's standoff (Robot-RPi-Core settings.FOLLOW_KEEP_M)
WALK_MAX_MPS = 1.5              # as robot_core.follow: a tracked jump beyond this x dt + 0.5 m is a goal jump


def default_data_dir() -> Path:
    env = os.environ.get("VS_DEBUG_DIR")
    if env:
        return Path(env).expanduser()
    return HERE.parent / "Vision-And-Planning" / "debug"


def sessions(data: Path) -> list[str]:
    return sorted(p.name for p in data.iterdir() if p.is_dir() and p.name[:2] == "20")


def _epoch(iso: Optional[str]) -> Optional[float]:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


@dataclass
class Run:
    """One follow task: the robot's events and the Mac's steps for it."""
    events: list = field(default_factory=list)        # (wall, note)
    steps: list = field(default_factory=list)         # follow.jsonl follow_step records


@dataclass
class Session:
    name: str
    folder: Path
    runs: list = field(default_factory=list)
    nav: list = field(default_factory=list)           # nav.jsonl records
    uploads: int = 0


def load_session(data: Path, name: str) -> Session:
    folder = Path(data) / name
    s = Session(name, folder)
    run: Optional[Run] = None
    for f in sorted(folder.glob("[0-9]*.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        s.uploads += 1
        note = d.get("note")
        if d.get("kind") != "event" or not isinstance(note, dict) or note.get("task") != "follow":
            continue
        if run is None or (note.get("seq") == 1 and run.events):
            run = Run()
            s.runs.append(run)
        run.events.append((_epoch(d.get("received")), note))
    fl = folder / "nav" / "follow.jsonl"
    if fl.exists():
        mac_runs, cur = [], None
        for line in fl.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("kind") == "follow_start" or (r.get("kind") == "follow_step" and cur is None):
                cur = []
                mac_runs.append(cur)
            if r.get("kind") == "follow_step":
                cur.append(r)
        for i, steps in enumerate(mac_runs):            # pair them in order with the robot's runs
            if i < len(s.runs):
                s.runs[i].steps = steps
            else:
                s.runs.append(Run(steps=steps))
    nj = folder / "nav" / "nav.jsonl"
    if nj.exists():
        for line in nj.read_text().splitlines():
            try:
                s.nav.append(json.loads(line))
            except ValueError:
                pass
    return s


# --------------------------------------------------------------------------- measures

def _pivot_flip_episodes(events) -> int:
    """Turning on the spot (v < 0.02, |w| > 0.25) going left-right-left (2 direction changes) within 2 s, once each."""
    flips, sign, episodes = [], 0, 0
    for wall, n in events:
        cmd = n.get("cmd")
        if not cmd or wall is None or cmd[0] >= 0.02 or abs(cmd[1]) <= 0.25:
            continue
        s = 1 if cmd[1] > 0 else -1
        if sign and s != sign:
            flips = [t for t in flips if wall - t <= 2.0] + [wall]
            if len(flips) >= 2:
                episodes, flips = episodes + 1, []
        sign = s
    return episodes


def _goal_jumps(events) -> int:
    last, n_jumps = None, 0
    for wall, n in events:
        trk = (n.get("tracker") or {}).get("xy")
        if not trk or wall is None:
            continue
        if last is not None and trk != last[1]:
            dt = max(0.0, wall - last[0])
            if math.dist(trk, last[1]) > WALK_MAX_MPS * dt + 0.5:
                n_jumps += 1
        if last is None or trk != last[1]:
            last = (wall, trk)
    return n_jumps


def measure_run(run: Run) -> dict:
    ev = run.events
    notes = [n for _, n in ev]
    walls = [w for w, _ in ev if w is not None]
    states = [n.get("state") for n in notes] or [r["out"]["state"] for r in run.steps]
    n_steps = len(states)
    locked_from = next((i for i, st in enumerate(states) if st == "tracking"), None)
    after = states[locked_from:] if locked_from is not None else []
    lost = sum(1 for a, b in zip(states, states[1:]) if a in ("tracking", "holding") and b == "searching")
    rules: dict = {}
    for n in notes:                                     # cumulative counters: keep the last value of each
        rules.update(n.get("rules") or {})
    trips = max([n.get("reflex_trips") or 0 for n in notes] or [0])
    marked = {tuple(m) for n in notes for m in (n.get("marked") or [])}
    mac_ref = edge_ok = near_ok = 0
    for r in run.steps:
        if r.get("pick") is None or r["pick"] >= len(r.get("persons", [])):
            continue
        p = r["persons"][r["pick"]]
        if p.get("tof_why"):
            mac_ref += 1
        elif r["out"].get("distance_from") == "tof" and p.get("feet_m") and p.get("tof"):
            d = p["tof"] * math.cos(math.radians(r.get("tilt") or 0.0))
            if (p.get("zones") or [1])[0] == 0 and d < p["feet_m"] - 0.3:
                edge_ok += 1                            # edge zones only, nearer than the feet: likely a door frame
            if d < 0.35 and p["feet_m"] > 0.8:
                near_ok += 1
    mac_s = [n["mac_s"] for n in notes if n.get("mac_s") is not None]
    if not mac_s and len(run.steps) > 1:
        mac_s = [b["wall"] - a["wall"] for a, b in zip(run.steps, run.steps[1:]) if 0 < b["wall"] - a["wall"] < 10]
    return {
        "steps": n_steps,
        "seconds": round(walls[-1] - walls[0], 1) if len(walls) > 1 else None,
        "locked": locked_from is not None,
        "tracking_pct": round(100 * after.count("tracking") / len(after)) if after else 0,
        "holding_pct": round(100 * after.count("holding") / len(after)) if after else 0,
        "lost": lost,
        "ended": states[-1] if states else None,
        "no_pose_steps": sum(1 for n in notes if n.get("pose") is None),
        "goal_jumps": _goal_jumps(ev),
        "pivot_flip_episodes": _pivot_flip_episodes(ev),
        "reflex_trips": trips,
        "marked_obstacles": len(marked),
        "pi_refused": sum(1 for n in notes if n.get("gated")),
        "mac_refused": mac_ref,
        "edge_ranges_used": edge_ok,
        "near_ranges_used": near_ok,
        "stood_way_round_long": sum(1 for n in notes if "way round is long" in (n.get("nav_detail") or "")),
        "degraded_steps": sum(1 for n in notes if (n.get("health") or ["normal"])[0] == "degraded"),
        "mac_step_s_median": round(statistics.median(mac_s), 3) if mac_s else None,
        "rules": rules,
    }


def measure(s: Session) -> dict:
    runs = [measure_run(r) for r in s.runs]

    def tot(k):
        return sum(r[k] or 0 for r in runs)
    secs = sum(r["seconds"] or 0 for r in runs)
    locked = [r for r in runs if r["locked"]]
    slam_status = next((r for r in s.nav if r.get("kind") == "status"), {})
    slam_frames = sum(1 for r in s.nav if r.get("kind") == "slam_frame")
    rules: dict = {}
    for r in runs:
        for k, v in r["rules"].items():
            rules[k] = rules.get(k, 0) + v
    meds = [r["mac_step_s_median"] for r in runs if r["mac_step_s_median"] is not None]
    return {
        "session": s.name,
        "runs": runs,
        "follow": {
            "runs": len(runs), "locked_runs": len(locked), "minutes": round(secs / 60.0, 2),
            "tracking_pct": round(statistics.mean([r["tracking_pct"] for r in locked])) if locked else 0,
            "lost_per_min": round(tot("lost") / (secs / 60.0), 2) if secs else 0.0,
            "goal_jumps": tot("goal_jumps"), "pi_refused": tot("pi_refused"), "mac_refused": tot("mac_refused"),
            "edge_ranges_used": tot("edge_ranges_used"), "near_ranges_used": tot("near_ranges_used"),
        },
        "nav": {
            "pivot_flip_episodes": tot("pivot_flip_episodes"),
            "reflex_trips_per_min": round(tot("reflex_trips") / (secs / 60.0), 2) if secs else 0.0,
            "marked_obstacles": tot("marked_obstacles"),
            "stood_way_round_long": tot("stood_way_round_long"),
            "plans": sum(1 for r in s.nav if r.get("kind") == "plan"),
        },
        "sensors": {
            "no_pose_steps": tot("no_pose_steps"), "degraded_steps": tot("degraded_steps"),
            "uploads": s.uploads,
        },
        "mac": {
            "step_s_median": round(statistics.median(meds), 3) if meds else None,
            "slam_tracked_frames": slam_frames,
            "slam_status": {k: v for k, v in slam_status.items() if k not in ("wall", "kind")},
        },
        "rules": rules,
    }


# --------------------------------------------------------------------------- what good looks like

# (area, key, test, limit, plain words). One place for the limits, used by the scorecard (a red or
# green line each) and by test_replay.py (a frozen session may not get worse than its own numbers).
LIMITS = [
    ("follow", "tracking_pct", ">=", 70, "kept the person in view at least 70% of the time once locked"),
    ("follow", "lost_per_min", "<=", 2.0, "lost the person at most twice a minute"),
    ("follow", "goal_jumps", "<=", 1, "the tracked person never jumped further than anyone can walk"),
    ("follow", "edge_ranges_used", "<=", 0, "never took a door-frame range (edge zones nearer than the feet) as the person"),
    ("follow", "near_ranges_used", "<=", 0, "never put the person under 0.35 m away while their feet were further"),
    ("nav", "pivot_flip_episodes", "<=", 0, "never turned on the spot left-right-left (steering target flipping)"),
    ("nav", "reflex_trips_per_min", "<=", 2.0, "the emergency brake fired at most twice a minute"),
    ("nav", "stood_way_round_long", "<=", 2, "rarely stood still because the way round looked long"),
    ("sensors", "no_pose_steps", "<=", 0, "had a fresh odometry pose on every step"),
    ("sensors", "degraded_steps", "<=", 5, "rarely ran degraded (ToF quiet, odometry late, Mac slow)"),
    ("mac", "step_s_median", "<=", 0.5, "the Mac answered a follow step in half a second (median)"),
]


def judge(m: dict, key_area: str, key: str, test: str, limit) -> tuple[bool, Optional[float]]:
    v = m.get(key_area, {}).get(key)
    if v is None:
        return True, None
    return (v >= limit if test == ">=" else v <= limit), v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=default_data_dir())
    ap.add_argument("--session", help="default: the newest")
    a = ap.parse_args()
    name = a.session or sessions(a.data)[-1]
    print(json.dumps(measure(load_session(a.data, name)), indent=2, default=str))


if __name__ == "__main__":
    main()
