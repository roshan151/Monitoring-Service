import sys, os, json, glob
sys.argv=['x']; sys.path.insert(0, os.path.expanduser('~/mnt/Monitoring-Service'))
import navigation_debugging as T
rc = T.robot_core(None); prm = T.robot_params(rc.root)
D = os.path.expanduser('~/mnt/Vision-And-Planning/debug')
for S in ('20261119-132454', '20261119-130747', '20261119-115626'):
    res = {}
    for label, zones in (("before", ""), ("now", prm["tof_self_zones"])):
        head = T.make_head(rc, dict(prm, tof_self_zones=zones))
        n = trips = 0; pan = tilt = 0.0; cmd = (0.15, 0.0)
        for f in sorted(glob.glob(f"{D}/{S}/[0-9]*.json")):
            m = json.load(open(f)); note = m.get("note") or {}
            if note.get("cmd"): cmd = tuple(note["cmd"])
            if m.get("gimbal"): pan, tilt = m["gimbal"]
            rows = (m.get("tof") or {}).get("rows"); pose = (m.get("extras") or {}).get("pose")
            if not rows or not pose or abs(cmd[0]) < 0.02 and abs(cmd[1]) < 0.05: continue   # only while moving
            n += 1
            zs = head.project(rows, pan, tilt, pitch_deg=(m.get("extras") or {}).get("pitch_deg") or 0.0)
            trips += bool(rc.Reflex(head, None, lambda: cmd, lambda: pose).danger(zs))
        res[label] = (trips, n)
    print(f"{S}: emergency brake would fire on {res['before'][0]}/{res['before'][1]} moving grids before, {res['now'][0]}/{res['now'][1]} now")
