#!/usr/bin/env python3
"""Drive the robot's real nav stack through simulated rooms, headless, and check it did right.

The same simulator as `navigation_debugging.py --sim` (Robot-RPi-Core's real LocalNav, reflex, DWA,
costmap and gimbal aim on a simulated tank, head and 8x8 ToF; the Mac's real GlobalMap planner from
../Vision-And-Planning), run faster than real time with no window, planning made deterministic
(LocalNav's planning threads run inline), and judged by plain rules per scenario: did it get there,
did it touch anything, did it turn left-right-left on the spot, did it brake when it had to. A
picture of each scenario's end goes next to the report.

    python3 test_scenarios.py                 every scenario
    python3 test_scenarios.py doorway chair   some
    python3 test_scenarios.py --list

Scenarios that fail today for a reason already known are listed in KNOWN: they are reported in their
own section, "Known problems", so a NEW failure stands out, and a known one that starts passing is
noticed too (then delete it from KNOWN).
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import types
from datetime import datetime
from pathlib import Path

import cv2

import navigation_debugging as T
from test_hardware import HERE, Results

# name: (rules, plain words). Rules: arrive (nav arrived by the end), near_person (ends within m of the
# person), through_x (gets past this x), no_contact, no_flip, brakes (the reflex fired), max_s, true_goal (the TRUE
# pose ends within m of the goal, not the odometry), contact_s (touching at most this long), same_approach (most times a
# failure repeated one at the same place and heading), near_by_s (within near_person of them by then).
RULES = {
    "box":      ({"arrive": True, "no_contact": True, "no_flip": True, "max_s": 45},
                 "drives round a box to the goal without touching it"),
    "door":     ({"arrive": True, "no_contact": True, "no_flip": True, "max_s": 40},
                 "goes through a 0.75 m doorway to the goal"),
    "trap":     ({"arrive": True, "no_contact": True, "max_s": 28},
                 "gets out of a U-shaped trap and round to the goal behind it"),
    "lookaway": ({"arrive": True, "no_contact": True, "max_s": 60},
                 "with the head held on someone far left, still glances ahead and misses a box"),
    "sudden":   ({"brakes": True, "no_contact": True},
                 "brakes for something that appears 0.36 m in front, and does not hit it"),
    "follow":   ({"near_person": 1.0, "no_contact": True, "no_flip": True},
                 "follows a person round a box between two walls"),
    "doorway":  ({"through_x": 1.9, "near_person": 1.3, "no_contact": True, "no_flip": True},
                 "follows the person through a doorway off to the left into the next room (Nov 19)"),
    "doorway2": ({"through_x": 1.9, "near_person": 1.3, "no_contact": True, "no_flip": True},
                 "the same, but the Mac has never seen the wall: only the robot's own ToF finds it (Nov 19)"),
    "doorleaf": ({"near_person": 1.3, "no_contact": True, "no_flip": True},
                 "follows the person past an open door leaf without touching it (Nov 19)"),
    "corridor": ({"arrive": True, "no_contact": True, "no_flip": True, "max_s": 40},
                 "drives down a 0.9 m corridor"),
    "nook":     ({"arrive": True, "no_contact": True, "max_s": 25},
                 "reverses out of a recess too tight to spin in, to a goal behind it"),
    "cable":    ({"true_goal": 0.3, "no_contact": True, "max_s": 40},
                 "notices the tracks spinning on a cable (odometry moving, it is not) and crosses it at an angle"),
    "wedge":    ({"true_goal": 0.3, "contact_s": 1.0, "max_s": 25},
                 "notices a turn held by a leg it cannot see (gyro still, tracks turning), backs off and turns clear of it"),
    "ledge":    ({"true_goal": 0.3, "max_s": 40},
                 "notices the nose riding up on a ledge it cannot see, backs off and goes round"),
    "glass":    ({"true_goal": 0.3, "same_approach": 0, "contact_s": 2.0, "max_s": 45},
                 "bumps a glass panel it cannot see, and never bumps it the same way twice"),
    "door20":   ({"through_x": 1.5, "near_person": 1.3, "near_by_s": 15, "no_contact": True, "no_flip": True},
                 "follows the person out of a corner door the Mac's map has shut, not round the long way (Nov 20)"),
    "corner20": ({"near_person": 1.3, "near_by_s": 10, "no_contact": True, "no_flip": True},
                 "goes round a wall corner close on its right to the person, without swinging left-right (Nov 20)"),
    "behind":   ({"arrive": True, "no_contact": True, "no_flip": True, "max_s": 16.5},
                 "spins round to a goal straight behind and drives there"),
    "chair":    ({"near_person": 1.0, "no_contact": True, "no_flip": True},
                 "stops near a person standing beside a chair, without hitting the chair (Nov 19)"),
}

# Failing today for a reason already understood (what to fix next): name -> why. Remove an entry once it
# passes. Empty since Nov 19 evening: trap, lookaway and sudden passed once the planner stopped relaxing
# its clearance rule across the whole map (and only ends a path where the full margin fits), the robot asked
# it for room beyond DWA's own margin, the fallback turn followed the target's side, a wedged robot backs off
# (no_progress), and skipping ahead along a path keeps clear of corners. pivot_commit and keep_plan were tried
# and removed: no scenario needed them and doorway2 and the replay were better without.
KNOWN: dict = {}


def _sync_planning(rc) -> None:
    """LocalNav plans in threads (it must not wait on the Mac); here they run inline so each run is the same."""
    from robot_core.nav import controller as C

    class SyncThread:
        def __init__(self, target=None, args=(), kwargs=None, name=None, daemon=None):
            self.t, self.a, self.k = target, args, kwargs or {}

        def start(self):
            self.t(*self.a, **self.k)

        def join(self, timeout=None):
            pass

    C.threading = types.SimpleNamespace(Thread=SyncThread, Lock=threading.Lock, Event=threading.Event)


def run_scenario(rc, prm, name: str, pictures: Path) -> dict:
    from robot_core.rules import rules as rule_log
    rule_log.reset()
    sim = T.NavSim(rc, prm, name)
    trace = []                                            # (t, v cmd, w cmd)
    contact_steps, near_t = 0, None
    near = RULES[name][0].get("near_person")
    while not sim.finished():
        before = sim.hits
        sim.step()
        pp = sim.person()
        if near_t is None and near and pp and math.dist(pp, sim.p[:2]) <= near:
            near_t = round(sim.t, 1)
        contact_steps += sim.hits - before
        trace.append((sim.t, sim.nav.cmd[0], sim.nav.cmd[1]))
    flips, sign, episodes = [], 0, 0                     # left-right-left turns on the spot within 2 s
    for t, v, w in trace:
        if abs(v) >= 0.02 or abs(w) <= 0.25:              # driving (either way) is not a turn on the spot
            continue
        s = 1 if w > 0 else -1
        if sign and s != sign:
            flips = [x for x in flips if t - x <= 2.0] + [t]
            if len(flips) >= 2:
                episodes, flips = episodes + 1, []
        sign = s
    person = sim.person()
    pic = pictures / f"{name}.png"
    try:
        cv2.imwrite(str(pic), T.side_by_side(T.render(sim.camera_meta(), None, mode="SIM"), T.draw_nav(sim.state())))
    except Exception:                                    # noqa: BLE001 — the picture is a bonus
        pic = None
    return {"t": round(sim.t, 1), "state": sim.nav.state, "detail": sim.nav.detail, "pose": [round(v, 2) for v in sim.p],
            "contact_s": round(contact_steps * 0.02, 2), "reflex_trips": sim.nav.reflex.trips,
            "flip_episodes": episodes, "person": person,
            "to_person": round(math.dist(person, sim.p[:2]), 2) if person else None,
            "marked": len(getattr(sim.nav, "marked", [])), "picture": str(pic) if pic else None,
            "to_goal_true": round(math.dist(sim.goal, sim.p[:2]), 2) if sim.goal else None,
            "to_goal_odom": round(math.dist(sim.goal, sim.o[:2]), 2) if sim.goal else None,
            "rules": rule_log.snapshot(), "near_t": near_t}


def judge(name: str, r: dict) -> tuple[bool, list[str]]:
    rules, _ = RULES[name]
    why = []
    if rules.get("true_goal") is not None and (r["to_goal_true"] is None or r["to_goal_true"] > rules["true_goal"]):
        why.append(f"really ended {r['to_goal_true']} m from the goal (it thought {r['to_goal_odom']} m)")
    if rules.get("near_by_s") and (r["near_t"] is None or r["near_t"] > rules["near_by_s"]):
        why.append(f"reached the person at {r['near_t']} s (limit {rules['near_by_s']} s)")
    if rules.get("contact_s") is not None and r["contact_s"] > rules["contact_s"]:
        why.append(f"touched something for {r['contact_s']} s (limit {rules['contact_s']} s)")
    if rules.get("same_approach") is not None and r["rules"].get("same_approach", 0) > rules["same_approach"]:
        why.append(f"failed the same way twice {r['rules']['same_approach']} time(s)")
    if rules.get("arrive") and r["state"] != "arrived":
        why.append(f"did not arrive (ended {r['state']}: {r['detail']})")
    if rules.get("max_s") and r["state"] == "arrived" and r["t"] > rules["max_s"]:
        why.append(f"took {r['t']} s (limit {rules['max_s']} s)")
    if rules.get("near_person") and (r["to_person"] is None or r["to_person"] > rules["near_person"]):
        why.append(f"ended {r['to_person']} m from the person (limit {rules['near_person']} m)")
    if rules.get("through_x") and r["pose"][0] < rules["through_x"]:
        why.append(f"did not get through (x {r['pose'][0]} < {rules['through_x']})")
    if rules.get("no_contact") and r["contact_s"] > 0:
        why.append(f"touched something for {r['contact_s']} s")
    if rules.get("no_flip") and r["flip_episodes"]:
        why.append(f"turned left-right-left on the spot {r['flip_episodes']} time(s)")
    if rules.get("brakes") and not r["reflex_trips"]:
        why.append("never braked")
    return not why, why


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="scenarios (default: all)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--robot-core", type=Path)
    a = ap.parse_args()
    if a.list:
        for n, (_, words) in RULES.items():
            print(f"{n:9} {words}" + ("   [known problem]" if n in KNOWN else ""))
        return
    names = a.names or list(RULES)
    bad = [n for n in names if n not in RULES or n not in T.SIM_SCENARIOS]
    if bad:
        raise SystemExit(f"unknown scenario(s): {bad}; --list shows them")
    rc = T.robot_core(a.robot_core)
    prm = T.robot_params(rc.root)
    _sync_planning(rc)
    pictures = HERE / "test-runs" / f"sim-{datetime.now():%Y%m%d-%H%M%S}"
    pictures.mkdir(parents=True, exist_ok=True)
    info = {"sim": ("Simulated rooms", "The robot's real navigation code driving a simulated tank: each room is one "
                                       "situation it must handle (pictures of how each ended are saved next to this report)."),
            "known": ("Known problems (not fixed yet)", "Situations it still fails for a reason already understood. "
                                                        "They are what to fix next; if one starts passing, take it off the list.")}
    res = Results(info, "Navigation scenarios (simulated)")
    found = {}
    for n in names:
        print(f"-- {n}: {RULES[n][1]}", flush=True)
        r = run_scenario(rc, prm, n, pictures)
        ok, why = judge(n, r)
        found[n] = {**r, "ok": ok, "why": why}
        res.mode = "known" if n in KNOWN else "sim"
        detail = (f"{'; '.join(why)}" if why else f"done in {r['t']} s") + \
            f" | trips {r['reflex_trips']}, marked {r['marked']}, picture {Path(r['picture']).name if r['picture'] else '-'}"
        if n in KNOWN:
            detail += f" | known: {KNOWN[n]}" if not ok else " | NOW PASSING: take it off KNOWN"
            res.check(RULES[n][1], True, ("still failing: " if not ok else "") + detail)
        else:
            res.check(RULES[n][1], ok, detail)
    res.data = {"scenarios": found, "pictures": str(pictures)}
    print(f"pictures in {pictures}")
    res.report()


if __name__ == "__main__":
    main()
