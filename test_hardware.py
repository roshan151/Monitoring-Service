#!/usr/bin/env python3
"""Test the running robot from the Mac: one script, one SSH session, robot-voice left running.

calibrate.py stops the robot and measures its parts; this tests the whole robot as it
runs: the ROS nodes, /odom and /motion_health, moves through the same /turn and
/drive actions the voice node uses (does the result come back? does odometry agree?),
the e-stop, the link to this Mac, a real follow ("follow me" by voice) and how fast
the voice session comes back after it. Nothing is installed on the Pi: the Pi-side
half below is sent over SSH and runs with the robot's ROS environment. Each run is saved to
test-runs/<time>.json and as a plain-English page, test-runs/<time>.html, opened in the browser.
test_software.py is its partner: the software end to end, nothing moves.

    python test_hardware.py              pick a mode from a menu
    python test_hardware.py all          stack, topics, mac, moves, follow, logs
    python test_hardware.py stack        robot-voice running, every node up, firmware (TF-Luna), no fatal since start
    python test_hardware.py topics       /odom and /motion_health arrive, robot idle and level
    python test_hardware.py mac          the Pi reaches the Vision-And-Planning; debug uploads arriving
    python test_hardware.py moves        turn 90 right/left, drive 30 cm out/back, e-stop mid-drive   (floor, 1 m clear)
    python test_hardware.py follow       you say "follow me": frames keep coming, it drives to you; stop; voice back
    python test_hardware.py logs         warnings and errors in logs.json since robot-voice started

Options: --host user@pi and --repo (shared with calibrate.py in .calibrate.json),
--service http://127.0.0.1:8080 (this Mac's Vision-And-Planning), --domain 42 (ROS_DOMAIN_ID)
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import shlex
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from urllib import request

from calibrate_and_apply import CONFIG, Pi as _CalibratePi, countdown, yes

HERE = Path(__file__).resolve().parent
MODES = ["stack", "topics", "mac", "moves", "follow", "logs"]

# What each mode means, for the report page (written for someone who doesn't know the code).
MODE_INFO = {
    "stack": ("Robot software is running", "The robot's main program is up, all its parts are talking to each "
              "other, and its motor board (Arduino) answers."),
    "topics": ("Robot knows where it is", "The robot keeps reporting its position and balance, and sits still and "
               "level when parked."),
    "mac": ("Robot can reach this Mac", "The robot can talk to the vision service over Wi-Fi, and its pictures arrive."),
    "moves": ("Robot moves correctly", "It turns and drives the amounts asked, and the emergency stop works."),
    "follow": ("\"Follow me\" works", "Ask it to follow you: it keeps watching, drives after you, then stops on "
               "command and is ready to listen again."),
    "logs": ("No errors in the robot's diary", "The robot's log file since it last started has no errors."),
}

# --------------------------------------------------------------------------- the Pi side (sent over SSH)

PI_SIDE = r'''
import json, math, os, subprocess, sys, threading, time
from datetime import datetime, timezone
from urllib import request

def say(**m):
    print(json.dumps(m, default=str), flush=True)

try:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from std_srvs.srv import Trigger
    from robot_interfaces.action import Drive, Turn
    from robot_interfaces.msg import MotionHealth
except Exception as e:
    say(ready=False, error=f"ROS environment: {type(e).__name__}: {e}")
    sys.exit(1)

LOG = os.path.join(os.getcwd(), os.environ.get("ROBOT_LOG_PATH", "logs.json"))
rclpy.init()
node = Node("system_test")
lock = threading.Lock()
st = {"odom": 0, "health": 0, "pose": None, "health_msg": None}

def on_odom(m):
    q = m.pose.pose.orientation
    with lock:
        st["odom"] += 1
        st["pose"] = (m.pose.pose.position.x, m.pose.pose.position.y, math.degrees(2.0 * math.atan2(q.z, q.w)))

def on_health(m):
    with lock:
        st["health"] += 1
        st["health_msg"] = {"state": m.state, "reason": m.reason, "imu_known": m.imu_known,
                            "pitch_deg": round(m.pitch_deg, 1), "roll_deg": round(m.roll_deg, 1)}

node.create_subscription(Odometry, "odom", on_odom, 10)
node.create_subscription(MotionHealth, "motion_health", on_health, 10)
clients = {"drive": ActionClient(node, Drive, "drive"), "turn": ActionClient(node, Turn, "turn")}
estop = node.create_client(Trigger, "estop")
task_stop = node.create_client(Trigger, "task/stop")
task_start = {t: node.create_client(Trigger, f"task/{t}") for t in ("follow", "explore")}
ex = MultiThreadedExecutor(num_threads=2)
ex.add_node(node)
threading.Thread(target=ex.spin, daemon=True).start()

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

def wait(fut, timeout):
    end = time.monotonic() + timeout
    while not fut.done() and time.monotonic() < end:
        time.sleep(0.02)
    return fut.result() if fut.done() else None

def pose():
    with lock:
        return st["pose"]

def read_log(since):
    out = []
    try:
        with open(LOG, errors="replace") as f:
            for line in f:
                if line.startswith('{"ts":"') and line[7:26] < since[:19]:     # cheap skip before parsing
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("ts", "") >= since:
                    out.append(e)
    except FileNotFoundError:
        pass
    return out

def cmd_hello(a):
    return {"host": sh("hostname"), "repo": os.getcwd(), "now": now(), "log": LOG}

def cmd_stack(a):
    props = dict(l.split("=", 1) for l in sh(
        "systemctl show robot-voice -p ActiveState -p ActiveEnterTimestamp -p NRestarts").splitlines() if "=" in l)
    ts = props.get("ActiveEnterTimestamp", "")
    since = sh(f'date -u -d "{ts}" +%Y-%m-%dT%H:%M:%S') if ts else ""
    for k in clients.values():
        k.wait_for_server(timeout_sec=3.0)
    estop.wait_for_service(timeout_sec=1.0)
    task_stop.wait_for_service(timeout_sec=1.0)
    evs = read_log(since) if since else []
    fw = [e.get("msg") for e in evs if "Arduino firmware" in (e.get("msg") or "")]
    return {"service": props, "since": since, "nodes": sorted(set(node.get_node_names())),
            "servers": {k: c.server_is_ready() for k, c in clients.items()},
            "estop": estop.service_is_ready(), "task_stop": task_stop.service_is_ready(),
            "firmware": fw[-1] if fw else None,
            "fatal": [e for e in evs if e.get("evt") == "fatal"][-3:]}

def cmd_topics(a):
    with lock:
        o0, h0 = st["odom"], st["health"]
    secs = float(a.get("seconds", 5))
    time.sleep(secs)
    with lock:
        return {"odom_hz": round((st["odom"] - o0) / secs, 1), "health_hz": round((st["health"] - h0) / secs, 1),
                "pose": st["pose"], "health": st["health_msg"]}

def cmd_move(a):
    op, value = a["op"], float(a["value"])
    c = clients[op]
    if not c.wait_for_server(timeout_sec=3.0):
        raise RuntimeError(f"/{op} action server not found (is the drivetrain node up?)")
    goal = Turn.Goal(degrees=value) if op == "turn" else Drive.Goal(meters=value)
    before, t0 = pose(), time.monotonic()
    gh = wait(c.send_goal_async(goal), 5.0)
    out = {"op": op, "value": value, "accepted": None if gh is None else gh.accepted}
    if not gh or not gh.accepted:
        return out
    out["accept_s"] = round(time.monotonic() - t0, 2)
    if a.get("estop_after_s"):
        time.sleep(float(a["estop_after_s"]))
        out["estop"] = bool(wait(estop.call_async(Trigger.Request()), 3.0))
    r = wait(gh.get_result_async(), float(a.get("timeout_s", 20)))
    out["result_s"] = round(time.monotonic() - t0, 2) if r else None
    out["status"] = r.result.status if r else None
    out["duration_s"] = round(r.result.duration_s, 2) if r else None
    time.sleep(0.6)                                   # odometry settles
    after = pose()
    out["odom_before"], out["odom_after"] = before, after
    if before and after:
        out["moved_m"] = round(math.hypot(after[0] - before[0], after[1] - before[1]), 3)
        out["turned_deg"] = round((after[2] - before[2] + 180.0) % 360.0 - 180.0, 1)   # CCW (left) positive
    return out

def cmd_brake(a):
    return bool(wait(estop.call_async(Trigger.Request()), 3.0))

def cmd_task_stop(a):
    if not task_stop.wait_for_service(timeout_sec=2.0):
        raise RuntimeError("/task/stop not found (voice node down?)")
    r = wait(task_stop.call_async(Trigger.Request()), 3.0)
    return {"message": r.message if r else None, "now": now()}

def cmd_task_start(a):
    c = task_start[a["task"]]
    if not c.wait_for_service(timeout_sec=3.0):
        raise RuntimeError(f"/task/{a['task']} not found: pull Robot-RPi-Core on the robot and restart robot-voice")
    r = wait(c.call_async(Trigger.Request()), 3.0)
    return {"ok": bool(r and r.success), "message": r.message if r else "no answer in 3 s", "now": now()}

def cmd_mac(a):
    ip = (os.environ.get("SSH_CLIENT") or "").split(" ")[0]
    url = f"http://{ip}:{a.get('port', 8080)}/healthz"
    t0 = time.monotonic()
    try:
        with request.urlopen(url, timeout=3) as r:
            h = json.loads(r.read())
        return {"url": url, "ok": h.get("status") == "ok", "ms": round((time.monotonic() - t0) * 1000)}
    except Exception as e:
        return {"url": url, "ok": False, "error": f"{type(e).__name__}: {e}"}

def cmd_logs(a):
    return read_log(a["since"])

def short(e, t):
    keys = {k: v for k, v in e.items() if k not in ("ts", "lvl", "evt", "msg", "src", "_unregistered", "_t")}
    return (f"{t:5.1f}s {e.get('evt')} " + (e.get("msg") or "") + " " + json.dumps(keys, default=str))[:170]

def cmd_watch(a):
    """Follow logs.json live for up to `seconds`, stop early at one of the `until` events."""
    secs, until = float(a.get("seconds", 30)), set(a.get("until") or [])
    with lock:
        o0 = st["odom"]
    f = open(LOG, errors="replace")
    f.seek(0, 2)
    t0, evs, hit, buf = time.monotonic(), [], None, ""
    while time.monotonic() - t0 < secs and hit is None:
        chunk = f.read()
        if chunk:
            buf += chunk
            *lines, buf = buf.split("\n")
            for line in lines:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                t = round(time.monotonic() - t0, 2)
                e["_t"] = t
                evs.append(e)
                msg = e.get("msg") or ""
                if e.get("evt") != "log" or e.get("lvl") != "info" or msg.startswith("follow"):
                    say(progress=short(e, t))
                if e.get("evt") in until and hit is None:
                    hit = t
        else:
            time.sleep(0.25)
    f.close()
    with lock:
        odom = st["odom"] - o0
    return {"events": evs, "odom_msgs": odom, "seconds": round(time.monotonic() - t0, 1), "until_s": hit}

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
ex.shutdown()
node.destroy_node()
rclpy.shutdown()
'''


def _remote(repo: str, domain: int, code: str) -> str:
    """The shell line run on the Pi: robot-voice's checkout and ROS environment, then `code`."""
    repo = "$HOME/" + repo[2:] if repo.startswith("~/") else repo
    b64 = base64.b64encode(code.encode()).decode()
    line = ('R=$(systemctl show -p WorkingDirectory --value robot-voice 2>/dev/null); '
            f'cd "${{R:-{repo}}}" && set +u && source /opt/ros/jazzy/setup.bash && '
            'source ros2_ws/install/setup.bash && { [ -f .venv/bin/activate ] && source .venv/bin/activate || true; } && '
            'export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp} '
            f'ROS_AUTOMATIC_DISCOVERY_RANGE=${{ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}} ROS_DOMAIN_ID={domain} && '
            f"exec python3 -u -c \"import base64;exec(base64.b64decode('{b64}').decode())\"")
    return "bash -c " + shlex.quote(line)


class Pi(_CalibratePi):
    """calibrate.py's JSON-lines session, running PI_SIDE instead of calibrate_pi."""

    def __init__(self, host: str, repo: str, domain: int, code: str = PI_SIDE) -> None:
        self.p = subprocess.Popen(["ssh", "-T", host, _remote(repo, domain, code)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        self.owed = 0
        first = self._line()
        if not first.get("ready"):
            raise SystemExit(f"the Pi side did not start: {first}")


# --------------------------------------------------------------------------- results

_TTY = sys.stdout.isatty()


def _paint(ok: bool, text: str) -> str:
    return f"\033[{'32' if ok else '31'}m{text}\033[0m" if _TTY else text


class Results:
    def __init__(self, info: dict | None = None, title: str = "Robot hardware test") -> None:
        self.info, self.title = MODE_INFO if info is None else info, title
        self.checks = []                                   # {mode, name, ok, detail}
        self.data = {}
        self.mode = ""
        self.skipped = {}                                  # mode -> why it did not run

    def check(self, name: str, ok, detail: str = "") -> None:
        self.checks.append({"mode": self.mode, "name": name, "ok": bool(ok), "detail": detail})
        print(f"   {_paint(bool(ok), 'PASS' if ok else 'FAIL')} {name}" + (f": {detail}" if detail else ""))

    def report(self) -> None:
        print("\n" + "=" * 72)
        for c in self.checks:
            print(f"  {_paint(c['ok'], 'PASS' if c['ok'] else 'FAIL')} {c['name']}"
                  + (f": {c['detail']}" if c["detail"] else ""))
        for m, why in self.skipped.items():
            print(f"  SKIP {m}: {why}")
        bad = sum(not c["ok"] for c in self.checks)
        print(f"\n{len(self.checks) - bad} passed, {bad} failed")
        out = HERE / "test-runs"
        out.mkdir(exist_ok=True)
        stamp = datetime.now()
        path = out / f"{stamp:%Y%m%d-%H%M%S}.json"
        path.write_text(json.dumps({"checks": self.checks, "skipped": self.skipped, "data": self.data},
                                   indent=2, default=str))
        page = path.with_suffix(".html")
        page.write_text(self.html(stamp))
        print(f"Saved {path}\nReport  {page}")
        try:
            webbrowser.open(page.as_uri())
        except Exception:                                   # noqa: BLE001  the page is saved either way
            pass

    def html(self, stamp: datetime) -> str:
        """One self-contained page: a big verdict, a tile per area, then every check in plain words."""
        def e(t):
            return html.escape(str(t))

        def cap(t):
            return t[:1].upper() + t[1:]
        bad = [c for c in self.checks if not c["ok"]]
        total = len(self.checks)
        if not total:
            verdict, tone = "Nothing was tested", "grey"
        elif not bad:
            verdict, tone = f"All good: {total} of {total} checks passed", "green"
        else:
            verdict, tone = f"{len(bad)} problem{'s' if len(bad) > 1 else ''} found ({total - len(bad)} of {total} " \
                            "checks passed)", "red"
        modes = list(dict.fromkeys([c["mode"] for c in self.checks] + list(self.skipped)))
        tiles, sections = [], []
        for m in modes:
            title, about = self.info.get(m, (m, ""))
            cs = [c for c in self.checks if c["mode"] == m]
            n_bad = sum(not c["ok"] for c in cs)
            if m in self.skipped and not cs:
                state, label, note = "grey", "Not tested", self.skipped[m]
            elif n_bad:
                state, label, note = "red", f"{n_bad} problem{'s' if n_bad > 1 else ''}", f"{len(cs) - n_bad} of {len(cs)} passed"
            else:
                state, label, note = "green", "Working", f"{len(cs)} of {len(cs)} passed"
            tiles.append(f'<a class="tile {state}" href="#{e(m)}"><div class="icon">'
                         f'{"&#10004;" if state == "green" else "&#10008;" if state == "red" else "&ndash;"}</div>'
                         f'<div><b>{e(title)}</b><span>{e(label)} &middot; {e(note)}</span></div></a>')
            rows = "".join(
                f'<li class="{"ok" if c["ok"] else "bad"}"><span class="mark">{"&#10004;" if c["ok"] else "&#10008;"}'
                f'</span><div><b>{e(cap(c["name"]))}</b>' + (f'<small>{e(c["detail"])}</small>' if c["detail"] else "")
                + "</div></li>" for c in sorted(cs, key=lambda c: c["ok"]))       # problems first
            if m in self.skipped and not cs:
                rows = f'<li class="skip"><span class="mark">&ndash;</span><div><b>Not tested</b><small>{e(self.skipped[m])}</small></div></li>'
            sections.append(f'<section id="{e(m)}"><h2 class="{state}">{e(title)} <em>{e(label)}</em></h2>'
                            f'<p>{e(about)}</p><ul>{rows}</ul></section>')
        fix = ""
        if bad:
            fix = "<div class='fix'><h3>What needs attention</h3><ol>" + "".join(
                f"<li><b>{e(self.info.get(c['mode'], (c['mode'],))[0])}: {e(cap(c['name']))}</b>"
                f"<small>{e(c['detail'])}</small></li>" for c in bad) + "</ol></div>"
        pct = round(100 * (total - len(bad)) / total) if total else 0
        return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{e(self.title)} {stamp:%d %b %H:%M}</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{{font:16px/1.45 -apple-system,system-ui,sans-serif;margin:0;background:#f4f5f7;color:#1d2330}}
main{{max-width:900px;margin:0 auto;padding:28px 18px 60px}}
.verdict{{border-radius:16px;padding:26px 28px;color:#fff;margin-bottom:22px}}
.verdict.green{{background:#1f9d55}}.verdict.red{{background:#d64545}}.verdict.grey{{background:#7a8291}}
.verdict h1{{margin:0;font-size:28px}}.verdict p{{margin:6px 0 14px;opacity:.9}}
.bar{{height:10px;background:rgba(255,255,255,.3);border-radius:5px;overflow:hidden}}
.bar i{{display:block;height:100%;background:#fff;width:{pct}%}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;margin-bottom:22px}}
.tile{{display:flex;gap:12px;align-items:center;background:#fff;border-radius:12px;padding:14px;text-decoration:none;
color:inherit;border-left:6px solid #ccc;box-shadow:0 1px 2px rgba(0,0,0,.06)}}
.tile span{{display:block;font-size:14px;color:#5b6270}}.tile .icon{{font-size:22px;width:28px;text-align:center}}
.tile.green{{border-color:#1f9d55}}.tile.green .icon{{color:#1f9d55}}
.tile.red{{border-color:#d64545}}.tile.red .icon{{color:#d64545}}.tile.grey .icon{{color:#7a8291}}
.fix{{background:#fff4f4;border:1px solid #f2c4c4;border-radius:12px;padding:6px 22px 14px;margin-bottom:22px}}
.fix li{{display:list-item;margin:8px 0;border:0;padding:0}}small{{display:block;color:#5b6270;font-size:13px;word-break:break-word}}
section{{background:#fff;border-radius:12px;padding:6px 22px 14px;margin-bottom:14px;box-shadow:0 1px 2px rgba(0,0,0,.06)}}
h2{{font-size:19px;margin:14px 0 2px}}h2 em{{font-style:normal;font-size:13px;padding:2px 9px;border-radius:10px;
margin-left:8px;vertical-align:middle;color:#fff;background:#7a8291}}
h2.green em{{background:#1f9d55}}h2.red em{{background:#d64545}}section>p{{margin:2px 0 8px;color:#5b6270}}
ul{{list-style:none;padding:0;margin:0}}li{{display:flex;gap:10px;padding:8px 0;border-top:1px solid #eef0f3}}
.mark{{font-weight:700;width:18px}}li.ok .mark{{color:#1f9d55}}li.bad .mark{{color:#d64545}}li.skip .mark{{color:#7a8291}}
footer{{color:#7a8291;font-size:13px;text-align:center}}
</style></head><body><main>
<div class="verdict {tone}"><h1>{e(verdict)}</h1><p>{e(self.title)}, {stamp:%A %d %B %Y, %H:%M}</p>
<div class="bar"><i></i></div></div>
<div class="tiles">{''.join(tiles)}</div>{fix}{''.join(sections)}
<footer>Monitoring-Service/{e(Path(sys.argv[0]).name)} &middot; raw data in the .json file of the same name</footer>
</main></body></html>"""


def _get(url: str):
    try:
        with request.urlopen(url, timeout=3) as r:
            return json.loads(r.read())
    except Exception:                                       # noqa: BLE001
        return None


# --------------------------------------------------------------------------- modes

def mode_stack(pi: Pi, res: Results, a) -> None:
    print("\n== stack: service, nodes, firmware")
    s = pi("stack")
    res.data["stack"] = s
    svc = s["service"]
    res.check("robot-voice running", svc.get("ActiveState") == "active",
              f"{svc.get('ActiveState')} since {svc.get('ActiveEnterTimestamp')}")
    res.check("no restarts", svc.get("NRestarts", "0") == "0", f"{svc.get('NRestarts')} restarts since boot "
              "(journalctl -u robot-voice -b)")
    missing = [n for n in NODES if n not in s["nodes"]]
    res.check("ROS nodes", not missing, f"missing {missing} (seen {s['nodes']}; wrong --domain?)" if missing
              else ", ".join(s["nodes"]))
    res.check("/turn and /drive servers", all(s["servers"].values()), str(s["servers"]))
    res.check("/estop and /task/stop", s["estop"] and s["task_stop"], f"estop {s['estop']}, task/stop {s['task_stop']}")
    fw = s["firmware"] or ""
    res.check("Arduino firmware", "v9" in fw, fw or "no 'Arduino firmware' line since start")
    res.check("TF-Luna answering", "-luna" in fw,
              "noluna: check its SDA/SCL wiring and 3.3 V pull-ups (drop guard is off)" if "noluna" in fw
              else fw or "unknown")
    res.check("no fatal since start", not s["fatal"], "; ".join(str(f.get("cause")) for f in s["fatal"]))


def mode_topics(pi: Pi, res: Results, a) -> None:
    print("\n== topics: /odom and /motion_health (5 s, robot still)")
    t = pi("topics", seconds=5)
    res.data["topics"] = t
    res.check("/odom arriving", t["odom_hz"] >= 2, f"{t['odom_hz']} Hz (follow and explore steer by it)")
    res.check("/motion_health arriving", t["health_hz"] >= 2, f"{t['health_hz']} Hz")
    h = t["health"] or {}
    res.check("idle while still", h.get("state") == "idle", f"{h.get('state')} {h.get('reason') or ''}".strip())
    res.check("IMU level", h.get("imu_known") and abs(h.get("pitch_deg", 99)) < 5 and abs(h.get("roll_deg", 99)) < 5,
              f"pitch {h.get('pitch_deg')}, roll {h.get('roll_deg')}" if h.get("imu_known") else "no IMU reading")


def mode_mac(pi: Pi, res: Results, a) -> None:
    print("\n== mac: Pi -> Vision-And-Planning")
    port = int(a.service.rsplit(":", 1)[-1]) if a.service.count(":") > 1 else 8080
    m = pi("mac", port=port)
    res.data["mac"] = m
    res.check("Pi reaches the service", m["ok"], f"{m['url']} in {m.get('ms')} ms" if m["ok"]
              else f"{m['url']}: {m.get('error')} (service running? firewall? same network?)")
    first = _get(a.service + "/v1/debug/latest")
    time.sleep(4)
    second = _get(a.service + "/v1/debug/latest")
    n0, n1 = (first or {}).get("n"), (second or {}).get("n")
    res.check("debug uploads arriving", n0 is not None and n1 is not None and n1 > n0,
              f"session {(second or {}).get('session')}: #{n0} -> #{n1} in 4 s" if second
              else "none: debug_mode false in robot.yaml, or the Pi cannot reach this Mac")


def _move(pi: Pi, res: Results, name: str, ask: str, **kw):
    r = pi("move", **kw)
    res.data.setdefault("moves", []).append(r)
    if not r.get("accepted"):
        res.check(name, False, "goal refused" if r.get("accepted") is False else "no answer to the goal")
        return r
    res.check(f"{name}: result came back", r["result_s"] is not None,
              f"{r['status']} after {r['result_s']} s" if r["result_s"] else
              "never: the action result is lost (the bug that froze follow)")
    if ask:
        res.check(f"{name}: you saw it", yes(f"   {ask}"))
    return r


def mode_moves(pi: Pi, res: Results, a) -> None:
    print("\n== moves: /turn, /drive, /estop")
    countdown(15, "Robot on the floor with 1 m clear all round. Mark where its nose points")
    for deg, way in ((90, "RIGHT"), (-90, "LEFT")):
        r = _move(pi, res, f"turn {way.lower()} 90", f"Did it turn about a quarter turn {way}?", op="turn", value=deg)
        if r.get("turned_deg") is not None:
            res.check(f"turn {way.lower()} 90: odometry", abs(abs(r["turned_deg"]) - 90) <= 15
                      and (r["turned_deg"] < 0) == (deg > 0),
                      f"{r['turned_deg']:+.0f} deg (left +): should be {-deg:+d}")
    for m, way in ((0.3, "FORWARD"), (-0.3, "BACK")):
        r = _move(pi, res, f"drive {m:+.1f} m", f"Did it drive about 30 cm {way}?", op="drive", value=m)
        if r.get("moved_m") is not None:
            res.check(f"drive {m:+.1f} m: odometry", abs(r["moved_m"] - 0.3) <= 0.08, f"{r['moved_m']:.2f} m")
    countdown(10, "E-stop: it drives forward 1 m and should brake after about 0.8 s. Keep 1 m clear ahead")
    r = _move(pi, res, "e-stop mid-drive", "", op="drive", value=1.0, estop_after_s=0.8)
    if r.get("accepted"):
        res.check("e-stop stops the drive", r.get("status") != "done" and (r.get("moved_m") or 0) < 0.6,
                  f"status {r.get('status')}, moved {r.get('moved_m')} m")


def _frames(evs):
    """Follow's per-frame lines (debug mode): [(t, frame number, drive m)]."""
    out = []
    for e in evs:
        m = re.match(r"follow (\d+):", e.get("msg") or "")
        if m:
            d = re.search(r"drive (-?\d+(?:\.\d+)?)", e["msg"])
            out.append((e["_t"], int(m.group(1)), float(d.group(1)) if d else 0.0))
    return out


def mode_follow(pi: Pi, res: Results, a) -> None:
    print("\n== follow: a real follow, by voice")
    countdown(20, 'Stand about 2 m in front of the robot, a little to one side, open floor between you. '
                  'When it says go, say "follow me"')
    w = pi("watch", seconds=45, until=["task.start"])
    if w["until_s"] is None:
        res.check("follow started", False, "no task.start in 45 s: was the voice session up and did it hear you?")
        return
    res.check("follow started", True, f"{w['until_s']:.0f} s after go")
    latest0 = _get(a.service + "/v1/debug/latest")
    print("   Now walk slowly backwards 2-3 m and stop; it should keep after you (30 s).")
    w = pi("watch", seconds=30, until=["task.end"])
    res.data["follow"] = w
    evs, fr = w["events"], _frames(w["events"])
    if not fr and not any(e.get("lvl") == "info" for e in evs):
        res.check("follow frames", False, "no info lines in logs.json: set debug_mode: true under /voice in robot.yaml")
    else:
        nums = sorted({n for _, n, _ in fr})
        res.check("frames keep coming", len(nums) >= 10, f"{len(nums)} frames in {w['seconds']:.0f} s")
        ts = [0.0] + [t for t, _, _ in fr] + [w["seconds"]]
        gap = max(b - t for t, b in zip(ts, ts[1:]))
        res.check("no stall", gap < 6, f"longest gap without a frame {gap:.1f} s")
        res.check("decided to drive", any(d > 0 for _, _, d in fr),
                  f"largest drive {max((d for _, _, d in fr), default=0):.2f} m")
    lost = [e for e in evs if "no result for" in (e.get("msg") or "")]
    res.check("move results arrive", not lost, f"{len(lost)} lost: " + "; ".join(e["msg"] for e in lost[:2]))
    res.check("/odom while following", w["odom_msgs"] > 0 and not any("no /odom" in str(e.get("err")) for e in evs),
              f"{w['odom_msgs']} messages")
    res.check("came toward you", yes("   Did it come toward you (not just turn)?"))
    latest1 = _get(a.service + "/v1/debug/latest")
    if latest0 and latest1:
        print(f"   debug uploads this follow: {latest1['n'] - latest0['n']} in debug/{latest1['session']}/")

    print("   Stopping follow through /task/stop; timing the voice session's return.")
    pi("task_stop")
    w = pi("watch", seconds=30, until=["voice.connected"])
    res.data["reconnect"] = w
    res.check("task ended", any(e.get("evt") == "task.end" for e in w["events"]) or
              any(e.get("evt") == "task.end" for e in evs), "")
    res.check("voice back quickly", w["until_s"] is not None and w["until_s"] <= 8,
              f"voice.connected {w['until_s']} s after the stop" if w["until_s"] is not None else "not within 30 s")


def mode_logs(pi: Pi, res: Results, a) -> None:
    print("\n== logs: since robot-voice started")
    since = (res.data.get("stack") or pi("stack"))["since"]
    evs = pi("logs", since=since)
    bad = [e for e in evs if e.get("lvl") in ("error", "critical", "fatal")]
    warn = [e for e in evs if e.get("lvl") == "warn"]
    counts = {}
    for e in warn + bad:
        k = f"{e.get('lvl')} {e.get('evt')}: {(e.get('msg') or e.get('err') or e.get('cause') or '')[:70]}"
        counts[k] = counts.get(k, 0) + 1
    for k, n in sorted(counts.items(), key=lambda x: -x[1])[:15]:
        print(f"   {n:4d} x {k}")
    res.data["logs"] = counts
    res.check("no errors since start", not bad, f"{len(bad)} errors, {len(warn)} warnings since {since} UTC")


# --------------------------------------------------------------------------- main

def connect(a, code: str = PI_SIDE):
    """Ask the Pi's address once (kept in .calibrate.json with calibrate.py), start `code` on it over SSH.
    Returns (pi, host, repo)."""
    cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    host = a.host or cfg.get("host") or input("Pi SSH address (user@hostname or user@ip): ").strip()
    repo = a.repo or cfg.get("repo") or "~/Robot-RPi-Core"
    try:
        CONFIG.write_text(json.dumps({"host": host, "repo": repo}))
    except OSError as e:                                    # only a convenience: never stop for it
        print(f"(could not remember the address in {CONFIG}: {e}; pass --host next time)")
    return Pi(host, repo, a.domain, code), host, repo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", nargs="?", choices=MODES + ["all"])
    ap.add_argument("--host")
    ap.add_argument("--repo")
    ap.add_argument("--domain", type=int, default=42)
    ap.add_argument("--service", default=os.environ.get("VS_URL", "http://127.0.0.1:8080"))
    a = ap.parse_args()
    a.service = a.service.rstrip("/")
    mode = a.mode
    if not mode:
        print("Modes: " + ", ".join(["all"] + MODES))
        mode = input("Which? [all] ").strip() or "all"
        if mode not in MODES + ["all"]:
            raise SystemExit(f"unknown mode {mode}")
    if not yes("Is the Pi on, robot-voice started (the robot greeted you), on the same network as this Mac?"):
        print("Switch it on and wait for the greeting, then run this again.")
        return
    pi, host, repo = connect(a)
    res = Results()
    try:
        hello = pi("hello")
        res.data["hello"] = hello
        print(f"Connected to {hello['host']} ({hello['repo']}).")
        for m in (MODES if mode == "all" else [mode]):
            res.mode = m
            try:
                globals()[f"mode_{m}"](pi, res, a)
            except RuntimeError as e:
                res.check(m, False, str(e))
            except KeyboardInterrupt:
                print("\n   skipped; braking")
                res.skipped[m] = "skipped (Ctrl-C)"
                subprocess.run(["ssh", "-T", host, _remote(repo, a.domain,
                               "import subprocess; subprocess.run(['ros2','service','call','/estop',"
                               "'std_srvs/srv/Trigger'], timeout=10)")])
    finally:
        pi.close()
        res.report()


if __name__ == "__main__":
    main()
