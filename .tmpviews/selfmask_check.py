import sys, os, math
sys.argv=['x']; sys.path.insert(0, os.path.expanduser('~/mnt/Monitoring-Service'))
import navigation_debugging as T
rc = T.robot_core(None); prm = T.robot_params(rc.root)
data = T.default_data_dir(); S = '20261119-132454'
frames = T.session_frames(data, S)
for label, zones in (("before (one 15 cm cutoff)", ""), ("with the per-zone self mask", prm["tof_self_zones"])):
    p = dict(prm, tof_self_zones=zones)
    nav = T.NavReplay(rc, p)
    blocked = near = n = 0
    for rel in frames:
        st = nav.upto(data, rel)
        if not st.get("pose"): continue
        n += 1
        blocked += any("BLOCKED" in t for t in st.get("text", []))
        x, y, th = st["pose"]
        cl = float(nav.cm.clearance([x + 0.2 * math.cos(th)], [y + 0.2 * math.sin(th)])[0])
        near += cl < 0.1                                    # an obstacle within 10 cm of a point 20 cm ahead
    print(f"{label:30s}: DWA finds no safe way in {blocked}/{n} uploads; obstacle just in front in {near}/{n}")
