# Monitoring-Service

Tools for checking and calibrating the robot's sensors from the Mac. The robot (Robot-RPi-Core) runs
with `debug_mode: true`, sends what its sensors collect to Vision-And-Planning on this
Mac, and the tools here read what the service stored. They make no network calls and
put no load on the robot.

```
 Robot (Pi)                         Mac
 camera frame + ToF grid  ──HTTP──▶ Vision-And-Planning  ──files──▶  navigation_debugging.py
 robot_core/debug_upload.py          POST /v1/debug/frame             (this repo)
 at most 2 a second                  debug/<session>/000001.jpg|json
```

## navigation_debugging.py --report: a whole run on one page, its problems marked

```bash
python3 navigation_debugging.py --report --open                       # the newest session, in the browser
python3 navigation_debugging.py --report --session 20261123-194111    # -> reports/20261123-194111.html
```

One HTML page per follow or explore run (explore needs the Pi's debug_mode on, as follow), to find
problems by eye instead of reading logs. Play it, drag the slider, or click a problem to jump there.
The terminal prints the same problem list.

- **Camera:** the frame with the ToF grid of that moment laid over it. Each zone is coloured by distance and outlined **red** where the Pi's costmap counts it as an obstacle, **green** where it counts as floor. Zones are shifted by however far the head turned between the grid and the frame. Follow's person boxes are drawn too (thick: the one it picked). A red zone on bare floor, or a wall where the picture shows none, is a sensor or calibration problem.
- **Map:** the Mac's global map of that moment (black: wall, orange: YOLO marks, **magenta: wall cells on ground the robot had already driven over**, i.e. ghosts). Also the plan (red) and the straight line to its goal (dashed), the robot (body circle, heading, ToF field of view), its trail, the person's track, and the ToF hits of that grid and the last 3 s. Wheel to zoom, drag to pan, double-click to follow the robot again.
- **Timeline:** bands for follow state, nav state and ORB-SLAM3, and a tick for every problem.
- **Problems found:**
  - floor read as obstacles: a whole-run ToF height / IMU pitch check, with the share of floor returns the Pi marks as obstacles, by column;
  - no path, and detours (path much longer than the straight line);
  - the path swapping sides within seconds;
  - reflex stops and nav blocked;
  - motion health (stalls, slips, ToF gaps);
  - the person lost;
  - ORB-SLAM3 losing track or crashing;
  - explore steps that failed.

Pages go to `reports/` (not in git). The camera frames are read from the debug folder where they are.

## navigation_debugging.py --nav / --sim: planning and obstacle avoidance

A top-down panel of the robot's nav stack (Robot-RPi-Core `robot_core/nav`: rolling costmap, 50 Hz
reflex, DWA, local Theta*, gimbal aim), beside the camera + ToF view. It imports the robot's own nav
code from `../Robot-RPi-Core` (or `--robot-core`), and its numbers from that folder's robot.yaml.

```bash
python3 navigation_debugging.py --nav                          # live, with the panel
python3 navigation_debugging.py --nav --session 20261119-101426   # rebuild a recorded session
python3 navigation_debugging.py --sim box                      # simulate: box | door | trap | lookaway | sudden | follow
python3 navigation_debugging.py --sim follow --snapshot f.png  # run to the end, save the last view
```

- **The panel:** occupied cells (white; brighter means surer), the robot's radius plus margin around them (blue tint), seen-free floor (grey), this grid's readings (dots, by range), the ToF's view (yellow wedge), the reflex zone (red, filled when it would brake), DWA's arcs at top speed (green open, red blocked) and its choice (bold green), the Mac's path (blue), the Pi's own Theta* (cyan), the goal and keep distance (magenta).
- **`--nav` on recordings:** the costmap is rebuilt from the uploads and DWA and Theta* are run again on it toward the goal follow recorded. Next to that, it shows what the robot itself reported (nav state, command, reflex trips). Uploads from before Nov 19 carry no head angle on ToF-only frames, so their grids are placed with the last known one.
- **Mac panel (third, `g` toggles):** what Vision-And-Planning computed during the session, recorded in `debug/<session>/nav/` (sessions from Nov 19 on, debug mode on): the global map as of that upload, the Mac's planned path, visual SLAM's base trajectory (green) against raw odometry (red) and SLAM-corrected odometry (yellow), the odometry fit's point pairs (purple), tracking state, frame counts, scale, residual and `odom_to_map`. If SLAM was not running it says so and why.
- **`--sim`:** the real `LocalNav` on a simulated tank, head and ToF, and the Mac's real `GlobalMap` planner if `../Vision-And-Planning` is there. The left panel is what the simulated ToF sees. Keys: space pause, n step, r restart, 1-6 scenario, s save.

## calibrate_and_apply.py: checks, calibration, and config application

One script for every hardware check, calibration, and directly applying the results to `robot.yaml` and the Pi (it replaces Robot-RPi-Core's
old `tests/hardware/` scripts and `apply_calibration.py`). It needs nothing installed beyond Python 3. The `tof`
overlay picture and the `service` latency test use OpenCV from `requirements.txt`.

```bash
python3 calibrate_and_apply.py           # interactive menu
python3 calibrate_and_apply.py all       # check, encoders, imu, tof, align, tfluna, head, drive
python3 calibrate_and_apply.py align     # ToF 8x8 vs camera alignment and FOV fitting
python3 calibrate_and_apply.py align --apply  # calibrate and immediately apply to robot.yaml and Pi
python3 calibrate_and_apply.py apply     # apply newest run from calibrations/ directly to robot.yaml
python3 calibrate_and_apply.py drive     # or one mode: check | encoders | imu | tof | align | tfluna | head | drive | door | service | apply
python3 calibrate_and_apply.py door      # what the ToF makes of a doorway of known width (not part of 'all')
```

### Settings: all in `.calibrate.json`

Every setting of every mode is in `.calibrate.json` next to the script. Edit it there. On each run the full set is
written back, so the file always lists them all:

| Section | What it sets |
|---|---|
| `host`, `repo` | the Pi's SSH address and the repo path on it |
| `target` | `align`: the box's size and distance |
| `door` | `door`: see below |
| `head` | `head`: pan and tilt sweep ranges and steps (commanded degrees) |
| `drive` | `drive`: the backing steps (m) and the turn (deg) |
| `encoders` | `encoders`: how long each track is watched |
| `imu` | `imu`: the drift check's length |

### Door (`door` mode)

The robot stands before a doorway of known width and only its head moves (step-and-hold, then one normal sweep each
way). It finds the wall the door is in and, along it, the door's two edges, look by look. It reports:

- **The width two ways.** Look by look, and as a map that keeps every look's edge would hold it, which is what the
  planner gets.
- **How far one look's edge can be from another's.** This is the jamb's zone flickering between the jamb and the room
  beyond.
- **Error that grows with distance**: the zones' 7.5° cones. **Error that is the same at every distance**: the angles
  or ranges are scaled.
- **The wall's angle**: the robot placed off square, or the head pointing off. If you squared it up, run `head`.
- **The range to the wall** against your placement.
- **Sweeping against holding the head**: the servos have no feedback, so a head that lags its command shows here.
- **For the left / right placements**: the near jamb's clearance, and whether the planner's 0.177 m circle plus
  4 cm would let it drive straight through.
- **Last, the gyro check.** The robot turns 60° each way, its head keeping the door in view. The gyro scale that makes
  the door land in one place.

`door` in `.calibrate.json` sets:

| Key | Meaning |
|---|---|
| `width_m` | the door, jamb to jamb (the mode asks; Enter keeps it) |
| `distances_m` | the door's wall to the ToF sensor's front face |
| `positions` | `centre`, `left`, `right`: left and right put the robot's side `edge_margin_m` inside that jamb |
| `pan_deg`, `step_deg`, `dwell_s`, `grids_per_step` | the head's steps |
| `sweep`, `sweep_dps` | the sweep |
| `turn_deg`, `turn_distance_m` | the gyro check (0 skips it) |
| `width_tol_m`, `edge_spread_tol_m` | what passes |
| `place_s` | the countdown to place the robot |

Every look is saved to `calibrations/door-<time>.json`.

### Target Calibration Configuration (`align` mode)

Target dimensions default to **56" H x 38" W x 30" D at 1.0 m** and can be configured through:

1. **Config file** (`.calibrate.json` or custom `--target-config <path.json>`):
```json
{
  "host": "user@pi",
  "repo": "~/Robot-RPi-Core",
  "target": {
    "width_in": 38.0,
    "height_in": 56.0,
    "depth_in": 30.0,
    "distance_m": 1.0
  }
}
```

2. **Command-line flags**:
```bash
python3 calibrate_and_apply.py align --target-width 38 --target-height 56 --target-depth 30 --target-distance 1.0 --apply
```

3. **Environment variables**:
`CALIB_TARGET_WIDTH_IN`, `CALIB_TARGET_HEIGHT_IN`, `CALIB_TARGET_DEPTH_IN`, `CALIB_TARGET_DIST_M`.

**What it does:**
1. It asks whether the Pi is on, then connects with `ssh`. Your Pi's address is asked once and kept in `.calibrate.json` next to the script; change it with `--host user@pi`, and the repo path with `--repo`.
2. It offers to stop `robot-voice` (`sudo` may ask for the Pi's password) and runs `robot_core/calibrate_pi.py` on the Pi through the same SSH session.
3. For each step it says what to do and counts down; press Enter to start early.
4. The robot measures with its own sensors:
   - the gyro measures turns
   - encoders count the tracks
   - the ToF sweeps a wall for pan zero, and the floor for tilt zero and the head's height
   - the TF-Luna reads the floor
5. You use a tape only in `drive`: from the wall to the ToF before and after the robot backs away 1 m.
6. At the end it prints the `robot.yaml` lines to change (with the old value and why). With `--apply` (or on confirmation), it updates `ros2_ws/src/robot_bringup/config/robot.yaml`, copies it to the Pi over SCP, and saves the run in `calibrations/<time>.json`.

It never edits `robot.yaml` or moves the robot without asking first (unless `--apply --yes` is specified). Ctrl-C skips the current step.

Modes and what each sets are listed in Robot-RPi-Core's README, "Calibration and checks".
`service` checks Vision-And-Planning on this Mac (health, detect latency) and needs no Pi.

## Testing: test_software.py and test_hardware.py

All of the robot's testing lives here (Robot-RPi-Core has no tests/ of its own any more).
Two end-to-end scripts, run from this Mac; nothing is installed on the Pi:

| | what it tests | does the robot move? |
|---|---|---|
| `test_software.py` | Vision-And-Planning on this Mac, called directly; then the robot's own code on the Pi over SSH, fed made-up situations with its real `robot.yaml` | no |
| `test_hardware.py` | the running robot: its ROS nodes, sensors, moves, e-stop, a real "follow me" | yes |

Each run prints PASS/FAIL as it goes, saves `test-runs/<time>.json`, and opens
`test-runs/<time>.html` in the browser: a big green or red verdict, a tile per area
("Robot moves correctly", "Vision service on this Mac", ...), a "What needs attention"
list, and every check in plain words, so anyone can read the result. Robot-voice keeps
running for both. The Pi's address comes from `.calibrate.json` (shared with
`calibrate.py`); `--domain` sets `ROS_DOMAIN_ID` if yours isn't 42.

### test_software.py: the software, end to end

```bash
python3 test_software.py           # a menu
python3 test_software.py all       # vision, then robot
python3 test_software.py vision    # this Mac only, no Pi needed
```

| mode | what passes |
|---|---|
| `vision` | the service is up and quick (detect p50 under 300 ms); models loaded; detect echoes the frame number and image size; depth returns one reading per point; a malformed request gets a clean 422; a faceless picture gets "no face" (404); plants, faces and the room map are readable; the queue isn't backed up |
| `robot` | the Pi runs the same commit as `../Robot-RPi-Core` with no local edits, and the same `robot.yaml`; every `robot_core` module loads (no missing packages); `robot_core` imports no ROS; `robot.yaml` has its sections; the voice tools are declared and `stop` takes no arguments; with the robot's own settings the terrain check drives over flat floor and a 2 cm sill and stops at a box and stairs, the planner goes round a box and gives up at a wall, the edge guard brakes at a drop but not on one bad reading, speed control starts gently and stops smoothly, odometry reads 1 m and 90°; the robot's own vision client reaches this Mac and gets correct answers |

### test_hardware.py: the running robot

`calibrate.py` stops the robot and measures its parts. `test_hardware.py` tests the whole
robot *while it runs*.

```bash
python3 test_hardware.py            # a menu
python3 test_hardware.py all        # stack, topics, mac, moves, follow, logs
python3 test_hardware.py follow     # or one mode
```

| mode | what passes |
|---|---|
| `stack` | robot-voice active with no restarts; drivetrain, voice and display nodes up; `/turn`, `/drive`, `/estop`, `/task/stop` there; firmware v9 with the TF-Luna answering; no `fatal` since start |
| `topics` | `/odom` and `/motion_health` arrive (2 Hz or more); idle and level while still |
| `mac` | the Pi reaches Vision-And-Planning on this Mac; debug uploads keep arriving |
| `moves` | turn 90° right and left, drive 30 cm out and back: each result comes back, odometry agrees, you confirm what you saw; the e-stop brakes a 1 m drive |
| `follow` | you say "follow me": 10+ frames in 30 s, no gap over 6 s, it decides to drive, no lost move results, `/odom` flowing, you confirm it came to you; then `/task/stop`, and the voice session is back within 8 s |
| `logs` | no errors in `logs.json` since robot-voice started (warnings are listed) |

`follow` reads follow's per-frame lines, so it needs `debug_mode: true`. Ctrl-C skips a
step and brakes the robot.

## Catching nav and follow bugs before the robot does: test_replay.py, test_scenarios.py, scorecard.py

Three tools, no robot needed. Each writes the same kind of plain-English page as the test scripts
(`test-runs/<time>.html`). What each guards against is listed in Robot-RPi-Core `docs/FAILURE-MODES.md`.

```bash
python3 scorecard.py                                    # score the newest debug session against fixed limits
python3 scorecard.py --session NEW --compare OLD        # ... and say what OLD scored on each line
python3 test_replay.py                                  # every saved scenario through today's code
python3 test_replay.py --session 20261119-182630        # one debug session straight from Vision-And-Planning
python3 test_replay.py --save doorway --session S --about "what happened"   # freeze a session as a scenario
python3 test_scenarios.py                               # the real nav stack in 19 simulated rooms
python3 test_scenarios.py doorway chair                 # some of them; --list names them all
```

- **scorecard.py** measures a session (`session_checks.py`, which also prints raw numbers as JSON):
  how much of the time the person was in view, how often they were lost, goal jumps, door-frame and
  too-near ranges taken as the person, left-right-left turns on the spot, brakes per minute, steps
  without odometry, degraded time, the Mac's step time, visual SLAM tracking, and the rule breaks the
  robot counted. The limits live in one place, `session_checks.LIMITS`.
- **test_replay.py** re-runs a recording through today's decision code: the Mac's range rules on every
  recorded person, the Pi's tracker on the recorded answers, today's planner on the map the Mac had at
  each plan (`nav/map-*.npz`) with the person as target, and the steering target and turn commitment
  on the costmap rebuilt from the recorded ToF grids. Saved scenarios live in `scenarios/replay/<name>/`
  (uploads as JSON without pictures, follow.jsonl, nav.jsonl, the maps) with a `scenario.json`: what
  happened, the limits it may not exceed, and what is known to be left.
- **test_scenarios.py** runs `navigation_debugging.py --sim`'s simulator headless and deterministic
  (planning inline), with bumps reported after 0.3 s of contact like the drivetrain's stall check, and a
  pass rule per room (got there, touched nothing, no left-right-left turns, braked when it had to). A
  picture of each room's end goes to `test-runs/sim-<time>/`. Rooms that fail for a known reason go in
  `KNOWN` and are reported separately, so a new failure stands out.

**The habit:** an incident on the robot → find it in the scorecard → freeze the session with
`test_replay.py --save` (or write the situation as a simulated room) → see the test fail → fix →
see it pass, and every other scenario still pass.

## navigation_debugging.py: camera + ToF overlay

![camera frame with the ToF grid laid over it](docs/sample-view.png)

The camera frame in black and white, with the 8x8 ToF grid laid over it as dots:
**red = close** (0.3 m or less), yellow in between, **green = far** (2 m or more), and a
grey ring where the sensor gave no reading. Each dot carries its range in metres. If the
red dots sit on the box in the picture, the ToF and the camera agree.

The header shows the frame number, head pan and tilt, how much older or newer the ToF
grid is than the image, how long the upload took from the robot to the Mac, and the
robot's pitch and pose.

### Setup (once)

```bash
cd ~/src/Monitoring-Service
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

### Run

```bash
.venv/bin/python navigation_debugging.py                     # live: the newest upload, as it arrives
.venv/bin/python navigation_debugging.py --list              # saved sessions
.venv/bin/python navigation_debugging.py --session 20261118-104200   # step through one
.venv/bin/python navigation_debugging.py --snapshot view.png # newest upload to a PNG, no window
```

Keys: `q` / `Esc` quit · `space` pause or play · `n` / `p` or the arrow keys step ·
`i` image on/off · `d` range numbers on/off · `s` save the view as a PNG.
Stepping while live pauses and moves through the current session.

It reads `../Vision-And-Planning/debug` by default (the two repos side by side in
`~/src`). Point it elsewhere with `--data <folder>` or `VS_DEBUG_DIR`.

### Turning debug mode on

On the robot, `ros2_ws/src/robot_bringup/config/robot.yaml`:

```yaml
/voice:
  ros__parameters:
    debug_mode: true
```

Then restart the robot (`sudo systemctl restart robot-voice`). Uploads start with the
next task that uses the camera or the ToF (follow, enroll, match). The service keeps the
newest 20 sessions (`VS_DEBUG_KEEP`).

### How the overlay is placed

Each ToF zone is drawn where its centre direction lands in the camera image: the grid
covers 60 x 60 degrees in 7.5-degree zones, and the camera's horizontal field of view
(`hfov_deg` in robot.yaml, 66 by default) gives the pixel scale. Zones outside the
camera's view (the ToF sees further up and down than the camera) are drawn in the margin.
The few centimetres between the ToF and the camera are ignored, so up close (under about
0.5 m) the dots sit slightly off what they measure.
