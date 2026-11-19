#!/usr/bin/env python3
"""Write a calibrate.py run into Robot-RPi-Core: the robot.yaml values (and a firmware
constant, if the run found one), after showing every change and asking.

    python apply_calibration.py                    the newest run in calibrations/
    python apply_calibration.py calibrations/20261118-143012.json
    python apply_calibration.py --dry-run          show the changes, write nothing
    python apply_calibration.py --yes              no question

Only the changed lines are edited, so comments and layout in robot.yaml stay as they
are; a key the section does not have yet is added under its `ros__parameters:` line.
Robot-RPi-Core is ../Robot-RPi-Core next to this folder (or --repo). Afterwards it
offers to copy robot.yaml to the Pi (the address calibrate.py remembered); a firmware
change still needs ./flash.sh from this Mac.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
YAML_REL = Path("ros2_ws/src/robot_bringup/config/robot.yaml")
FIRMWARE_REL = Path("firmware/drivetrain/drivetrain.ino")


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
            if comment:                                  # keep the comment in its column (one space if no room)
                col = len(m.group(1)) + len(m.group(2)) + len(comment) - len(comment.lstrip())
                text = (text.ljust(col) if len(text) < col else text + " ") + comment.lstrip()
            lines[i] = text + "\n"
            return old, i + 1
    params = next((i for i in range(start, end) if lines[i].strip() == "ros__parameters:"), start)
    indent = re.match(r"^(\s*)", lines[params + 1] if params + 1 < end else "    ").group(1) or "    "
    lines.insert(params + 1, f"{indent}{key}: {fmt(value)}    # added by apply_calibration.py\n")
    return None, params + 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", nargs="?", type=Path, help="a calibrations/*.json (default: the newest)")
    ap.add_argument("--repo", type=Path, default=HERE.parent / "Robot-RPi-Core")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    run = a.run
    if run is None:
        runs = sorted((HERE / "calibrations").glob("2*.json"))
        if not runs:
            sys.exit("no runs in calibrations/: run calibrate.py first")
        run = runs[-1]
    new = json.loads(run.read_text()).get("new", {})
    yaml_path, fw_path = a.repo / YAML_REL, a.repo / FIRMWARE_REL
    if not yaml_path.exists():
        sys.exit(f"no robot.yaml at {yaml_path}: pass --repo <Robot-RPi-Core>")

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
            print(f"  (firmware: no #define {key} found, skipped)")
            continue
        if m.group(2) != str(value):
            changes.append(f"  drivetrain.ino {key}: {m.group(2)} -> {value}   ({why})")
            fw_text = fw_text[:m.start(2)] + str(value) + fw_text[m.end(2):]

    print(f"From {run.name}:")
    if not changes:
        print("  nothing to change: robot.yaml already has these values")
        return
    print("\n".join(changes))
    if a.dry_run:
        print("(dry run: nothing written)")
        return
    if not a.yes and input("Write these? [y/N] ").strip().lower()[:1] != "y":
        print("nothing written")
        return
    yaml_path.write_text("".join(lines))
    if new.get("firmware") and fw_text:
        fw_path.write_text(fw_text)
    print(f"written: {yaml_path}" + (f" and {fw_path}" if new.get("firmware") else ""))

    cfg = HERE / ".calibrate.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        dest = f"{c['host']}:{c.get('repo', '~/Robot-RPi-Core')}/{YAML_REL}"
        if a.yes or input(f"Copy robot.yaml to the Pi ({dest})? [Y/n] ").strip().lower()[:1] != "n":
            ok = subprocess.run(["scp", str(yaml_path), dest]).returncode == 0
            print("copied; restart the robot: ssh " + c["host"] + " sudo systemctl restart robot-voice"
                  if ok else "copy failed: copy robot.yaml to the Pi yourself")
    else:
        print("copy robot.yaml to the Pi (or pull there), then: sudo systemctl restart robot-voice")
    if new.get("firmware"):
        print("firmware changed: flash it from this Mac (cd ../Robot-RPi-Core && ./flash.sh)")


if __name__ == "__main__":
    main()
