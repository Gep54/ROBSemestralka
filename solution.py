import numpy as np

import PoseComposer
import camera


# import puzzle


def move_robot_to(position, robot):
    q0 = robot.q_home
    ik_sols = robot.ik(position)
    assert len(ik_sols) > 0
    closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))
    robot.move_to_q(closest_solution)
    robot.wait_for_motion_stop()


def solution(puzzle, robot, T_RC):
    robot.soft_home()
    cam = camera.Camera()
    img = robot.grab_image()
    midpoint, R = cam.get_mid_points(img)
    midpoint[2] -= 35
    trajectory_R = puzzle.get_reverse_trajectory()
    trajectory_F = trajectory_R[::-1]
    trajectory_F[:, 2] *= -1
    trajectory_F_camera_coord = trajectory_F @ R + midpoint
    trajectory_F_robot_coord = [cam.cameraToRobot(x, T_RC) for x in trajectory_F_camera_coord]
    # for p in trajectory_F_robot_coord:
    #     p[2] += 0.05
    start = trajectory_F_robot_coord[0].copy()
    start[2] += 0.05
    arm_vector = np.array([-1, 1, 0])
    normal_vector = np.array([0, 0, 1])
    StartPose = PoseComposer.make_se3_matrix(start, arm_vector, normal_vector)
    move_robot_to(StartPose, robot)

    for position in trajectory_F_robot_coord:
        Pose = PoseComposer.make_se3_matrix(position, arm_vector, normal_vector)
        move_robot_to(Pose, robot)

    for position in trajectory_F_robot_coord[::-1]:
        Pose = PoseComposer.make_se3_matrix(position, arm_vector, normal_vector)
        move_robot_to(Pose, robot)

    move_robot_to(StartPose, robot)
    robot.soft_home()
