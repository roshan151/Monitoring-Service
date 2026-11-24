#!/usr/bin/env python3
"""Score a recorded robot session in plain words: what went well, what broke, against fixed limits.

Reads only what Vision-And-Planning stored for the session (debug_mode on) and measures it with
session_checks.py. Each limit is a green or red line on the same kind of page as the test scripts
(test-runs/<time>.html, opened in the browser), so two sessions can be compared by their numbers
instead of by how a run felt.

    python3 scorecard.py                                   the newest session
    python3 scorecard.py --session 20261119-182630         one session
    python3 scorecard.py --session NEW --compare OLD       each line also says what OLD scored
    python3 scorecard.py --list                            the sessions there are

The limits live in session_checks.LIMITS (one place, also used by test_replay.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from session_checks import LIMITS, default_data_dir, judge, load_session, measure, sessions
from test_hardware import Results

INFO = {
    "follow": ("Following the person", "Did it keep them in view, and believe only sensible distances?"),
    "nav": ("Driving and obstacles", "Did it steer steadily, and brake rarely?"),
    "sensors": ("Sensors and health", "Did the odometry, ToF and head keep working?"),
    "mac": ("Mac: vision and SLAM", "Did the Mac answer quickly, and did visual SLAM track?"),
    "rules": ("Rule breaks the robot counted", "Things that should never happen (robot_core/rules.py), as the robot counted them"),
}


def _fmt(v) -> str:
    return "-" if v is None else f"{v:g}" if isinstance(v, (int, float)) else str(v)


def score(res: Results, m: dict, old: dict | None = None) -> None:
    for area, key, test, limit, words in LIMITS:
        res.mode = area
        ok, v = judge(m, area, key, test, limit)
        was = ""
        if old is not None:
            _, ov = judge(old, area, key, test, limit)
            was = f" (was {_fmt(ov)} in {old['session']})"
        if v is None:
            res.check(words, True, f"not measured in this session{was}")
        else:
            res.check(words, ok, f"{key} = {_fmt(v)}, limit {test} {_fmt(limit)}{was}")
    res.mode = "mac"
    vs = (m["mac"].get("slam_status") or {}).get("vslam") or {}
    fin, ftr = vs.get("frames_in") or 0, m["mac"].get("slam_tracked_frames") or 0
    if vs:
        res.check("visual SLAM tracked at least half the frames it was given", fin == 0 or ftr >= 0.5 * fin,
                  f"{ftr} tracked of {fin} given" + (f"; not running: {vs.get('reason')}" if vs.get("reason") else ""))
    res.mode = "rules"
    if m["rules"]:
        for rule, n in sorted(m["rules"].items()):
            res.check(f"rule {rule} never broken", n == 0, f"broken {n} time{'s' if n != 1 else ''}")
    else:
        res.skipped["rules"] = "this session's robot code did not count rule breaks yet (Nov 19 evening on)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=default_data_dir())
    ap.add_argument("--session", help="default: the newest")
    ap.add_argument("--compare", help="a second session to compare with")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    names = sessions(a.data)
    if a.list or not names:
        print("\n".join(names) or f"no sessions in {a.data}")
        return
    name = a.session or names[-1]
    m = measure(load_session(a.data, name))
    old = measure(load_session(a.data, a.compare)) if a.compare else None
    f = m["follow"]
    print(f"Session {name}: {f['runs']} follow run(s), {f['minutes']} min")
    res = Results(INFO, f"Session scorecard {name}" + (f" vs {a.compare}" if a.compare else ""))
    res.data = {"measures": m, "compare": old}
    score(res, m, old)
    res.report()


if __name__ == "__main__":
    main()
