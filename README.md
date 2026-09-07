# ROB Semestrálka

**Vision-guided maze threading on the CRS97 robot**

Semester project by Václav Beran, Mathias Palme, and Tomáš Janoušek  
Czech Technical University in Prague — Faculty of Electrical Engineering

The robot holds a hoop on its end effector. A calibrated camera finds the maze on the table via ArUco markers, the planned path is transformed into robot coordinates, and the arm threads the hoop through the maze and back out.

---

## What this project does

Physical maze pieces (puzzles A–C) sit on a marked base. Each maze is a 3D path made of line segments. The software:

1. **Sees** the maze with the robot camera (ArUco IDs `1` and `2` on the base)
2. **Places** a known geometric model of that maze into the camera frame
3. **Transforms** the path into the CRS97 base frame
4. **Chooses** a hoop orientation that has inverse-kinematics solutions along the whole path
5. **Moves** the hoop through the maze, pauses, then retraces the path and returns home

```
camera image  →  ArUco pose  →  maze points in camera frame
                                          ↓
                              T_RC  (camera → robot)
                                          ↓
                              hoop SE(3) poses  →  IK  →  joint motion
```

---

## Repository layout

| File | Role |
| --- | --- |
| [`solution.py`](solution.py) | Main pipeline: detect maze, plan hoop poses, execute forward and reverse motion |
| [`puzzle.py`](puzzle.py) | Geometric models of mazes A–C as sequences of lines, origin at the base center |
| [`components.py`](components.py) | `Line` and `Arc` primitives that sample discrete 3D points |
| [`PoseComposer.py`](PoseComposer.py) | Builds the hoop’s SE(3) pose from a path point, hoop normal, and arm direction |
| [`camera.py`](camera.py) | ArUco detection, camera-to-robot calibration (`solvePnP`), and point transforms |
| [`camera_intrinsics.npz`](camera_intrinsics.npz) | Intrinsic matrix and distortion coefficients |
| [`calibrate_cam_to_rob.py`](calibrate_cam_to_rob.py) | Standalone camera–robot extrinsics from matched 3D/2D points |
| [`imageClicker.py`](imageClicker.py) | Click pixels in a photo to collect 2D correspondences |
| [`Showing.ipynb`](Showing.ipynb) | Live demo: initialize the robot and run `solve_maze` |
| [`Planning.ipynb`](Planning.ipynb) | Offline IK feasibility checks for hoop directions |
| [`Find aruco, move robot hoop there.ipynb`](Find%20aruco,%20move%20robot%20hoop%20there.ipynb) | Early detection + motion experiments |
| [`Camera test.ipynb`](Camera%20test.ipynb) | Camera / marker debugging |
| [`docs/ROB_report.pdf`](docs/ROB_report.pdf) | Project report |

---

## Maze models

Paths are defined in millimetres relative to the centre of the base, then sampled into trajectories.

| Puzzle | Shape |
| --- | --- |
| **A** | Straight vertical line |
| **B** | Vertical → diagonal → vertical |
| **C** | Four-segment 3D polyline |

```python
from puzzle import PuzzleC

maze = PuzzleC()
maze.show_reverse_trajectory()   # 3D matplotlib preview
points = maze.get_reverse_trajectory(number_of_points=10)
```

`get_reverse_trajectory()` samples each component from the start of the maze toward the far end. `solution.py` then flips that path into the camera/robot frames.

---

## How the hoop pose is built

The end effector is not at the hoop centre. `PoseComposer.make_se3_matrix` takes:

- **circle position** — where the hoop centre should be
- **circle normal** — orientation of the hoop plane (typically `[0, 0, 1]`)
- **arm vector** — direction from the hoop centre toward the robot wrist

and returns a 4×4 SE(3) matrix for the CRS97 inverse kinematics. The wrist is offset along the arm vector by the hoop arm length (`0.135 m` by default).

`find_options_around_point` samples candidate arm directions in a circle around a point (radius, angular step, and arc fraction are configurable). `solve_maze` keeps the first direction that has a valid IK solution at every waypoint.

---

## Running a maze

Hardware required: CRS97, powered arm, calibrated camera, maze on the marked base.

```python
from ctu_crs import CRS97
import puzzle
import solution

robot = CRS97()
robot.initialize(home=False)

# Camera-to-robot transform (mm). Recalibrate if the camera moved.
T_RC = [
    [-3.09275095e-02,  9.98889991e-01, -3.55285092e-02,  4.71849312e+02],
    [ 9.99249525e-01,  3.00701923e-02, -2.44165920e-02, -1.25522968e+01],
    [-2.33211402e-02, -3.62569903e-02, -9.99070345e-01,  1.19959167e+03],
    [ 0.0,             0.0,             0.0,             1.0           ],
]

maze = puzzle.PuzzleA()
solution.solve_maze(maze, robot, T_RC, mat_thickness=30)
```

`solve_maze` prints the planned Cartesian waypoints and asks for confirmation (`Y/n`) before moving. After a short pause at the far end, the arm retraces the path and soft-homes.

The same flow is in [`Showing.ipynb`](Showing.ipynb). Swap `PuzzleA` for `PuzzleB` or `PuzzleC` as needed.

`mat_thickness` (mm) accounts for the physical height of the base plate under the ArUco markers so the hoop is aimed at the maze, not the marker plane.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Notable dependencies: `numpy`, `opencv-python`, `opencv-contrib-python` (ArUco), `matplotlib`, `ctu-crs` (CRS97 driver).

---

## Camera calibration

**Intrinsics** are stored in `camera_intrinsics.npz` (`camera_matrix`, `dist_coeffs`) and loaded by `camera.Camera`.

**Extrinsics** (`T_RC`, camera → robot) come from paired points:

1. Collect robot-frame 3D positions and matching image pixels (`imageClicker.py` helps with the 2D side).
2. Run `Camera.transformFromCameraToRobot(robot_xyz_mm, image_uv)` or `calibrate_cam_to_rob.py`.
3. Pass the resulting 4×4 matrix into `solve_maze`.

At runtime, `get_mid_points` detects markers `1` and `2`, averages their translations for the base centre, and uses yaw from the marker pose to rotate the maze model in the table plane.

---

## Authors


Václav Beran, Mathias Palme, Tomáš Janoušek

FEL ČVUT, course **ROB** (Robotics).
