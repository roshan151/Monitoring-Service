# Monitoring-Service

Tools for checking and calibrating the robot's sensors from the Mac. The robot (Robot-RPi-Core) runs
with `debug_mode: true`, sends what its sensors collect to Vision-Microservice on this
Mac, and the tools here read what the service stored. They make no network calls and
put no load on the robot.

```
 Robot (Pi)                         Mac
 camera frame + ToF grid  ──HTTP──▶ Vision-Microservice  ──files──▶  tof_camera_overlaid.py
 robot_core/debug_upload.py          POST /v1/debug/frame             (this repo)
 at most 2 a second                  debug/<session>/000001.jpg|json
```

## calibrate.py: checks and calibration

One script for every hardware check and calibration (it replaces Robot-RPi-Core's
old `tests/hardware/` scripts). It needs nothing installed beyond Python 3. The `tof`
overlay picture and the `service` latency test use OpenCV from `requirements.txt`.

```bash
python3 calibrate.py           # a menu
python3 calibrate.py all       # check, encoders, imu, tof, tfluna, head, drive
python3 calibrate.py drive     # or one mode: check | encoders | imu | tof | tfluna | head | drive | service
```

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
6. At the end it prints the `robot.yaml` lines to change (with the old value and why), and any firmware constant. It saves the run in `calibrations/<time>.json`, then offers to start `robot-voice` again.

It never edits `robot.yaml` or moves the robot without asking first. Ctrl-C skips the current step.

Modes and what each sets are listed in Robot-RPi-Core's README, "Calibration and checks".
`service` checks Vision-Microservice on this Mac (health, detect latency) and needs no Pi.

### apply_calibration.py: write a run into robot.yaml

```bash
python3 apply_calibration.py              # the newest run in calibrations/
python3 apply_calibration.py --dry-run    # show the changes only
python3 apply_calibration.py calibrations/20261118-143012.json
```

It shows each change (old value → new, the line, and why), asks, then edits only
those lines in `../Robot-RPi-Core/ros2_ws/src/robot_bringup/config/robot.yaml`, so your
comments and layout stay as they are. A key the section doesn't have yet is added
under its `ros__parameters:` line. A firmware constant (`REVERSE_TRIM_PCT`) goes into
`firmware/drivetrain/drivetrain.ino`. It then offers to copy `robot.yaml` to the Pi
(the address `calibrate.py` remembered) and tells you to restart `robot-voice`. A
firmware change still needs `./flash.sh` from this Mac. Use `--repo` if Robot-RPi-Core
lives elsewhere, and `--yes` to skip the question.

## Testing: test_software.py and test_hardware.py

All of the robot's testing lives here (Robot-RPi-Core has no tests/ of its own any more).
Two end-to-end scripts, run from this Mac; nothing is installed on the Pi:

| | what it tests | does the robot move? |
|---|---|---|
| `test_software.py` | Vision-Microservice on this Mac, called directly; then the robot's own code on the Pi over SSH, fed made-up situations with its real `robot.yaml` | no |
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
| `mac` | the Pi reaches Vision-Microservice on this Mac; debug uploads keep arriving |
| `moves` | turn 90° right and left, drive 30 cm out and back: each result comes back, odometry agrees, you confirm what you saw; the e-stop brakes a 1 m drive |
| `follow` | you say "follow me": 10+ frames in 30 s, no gap over 6 s, it decides to drive, no lost move results, `/odom` flowing, you confirm it came to you; then `/task/stop`, and the voice session is back within 8 s |
| `logs` | no errors in `logs.json` since robot-voice started (warnings are listed) |

`follow` reads follow's per-frame lines, so it needs `debug_mode: true`. Ctrl-C skips a
step and brakes the robot.

## tof_camera_overlaid.py: camera + ToF overlay

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
.venv/bin/python tof_camera_overlaid.py                     # live: the newest upload, as it arrives
.venv/bin/python tof_camera_overlaid.py --list              # saved sessions
.venv/bin/python tof_camera_overlaid.py --session 20261118-104200   # step through one
.venv/bin/python tof_camera_overlaid.py --snapshot view.png # newest upload to a PNG, no window
```

Keys: `q` / `Esc` quit · `space` pause or play · `n` / `p` or the arrow keys step ·
`i` image on/off · `d` range numbers on/off · `s` save the view as a PNG.
Stepping while live pauses and moves through the current session.

It reads `../Vision-Microservice/debug` by default (the two repos side by side in
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
