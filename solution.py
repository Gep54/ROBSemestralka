from time import sleep

import numpy as np

import PoseComposer
import camera


def move_robot_to(position, robot):
    q0 = robot.get_q()
    configuration = robot_configuration_maker(position, robot, q0)
    assert configuration is not None
    robot.move_to_q(configuration)
    robot.wait_for_motion_stop()


def robot_configuration_maker(position, robot, q0):
    ik_sols = robot.ik(position)
    if len(ik_sols) == 0:
        return None
    closest_configuration = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))
    if not robot.in_limits(closest_configuration):
        return None
    return closest_configuration


def handle_camera_coords(robot, maze, T_RC, mat_thickness=35):
    cam = camera.Camera()
    img = robot.grab_image()
    midpoint, R = cam.get_mid_points(img)
    midpoint[2] -= mat_thickness
    trajectory_R = maze.get_reverse_trajectory()
    trajectory_F = trajectory_R[::-1]
    trajectory_F[:, 2] *= -1
    trajectory_F_camera_coord = trajectory_F @ R + midpoint
    trajectory_F_robot_coord = [cam.cameraToRobot(x, T_RC) for x in trajectory_F_camera_coord]
    return trajectory_F_robot_coord


def arm_vector_decider(maze_f_robot_coord, normal, robot):
    first = maze_f_robot_coord[0].copy()
    _, vectors = PoseComposer.find_options_around_point(first, normal)
    vectors = [tuple(v) for v in vectors]
    for vector in vectors:
        q0 = robot.q_home
        for point in maze_f_robot_coord:
            pose = PoseComposer.make_se3_matrix(point, vector, normal)
            robot_config = robot_configuration_maker(pose, robot, q0)
            q0 = robot_config
            if robot_config is None:
                vectors.remove(vector)
                break
    if len(vectors) == 0:
        return None
    arm_vector = vectors[0]
    return arm_vector


def solve_maze(maze, robot, T_RC, normal_vector=np.array([0, 0, 1])):
    robot.soft_home()

    maze_f_robot_coord = handle_camera_coords(robot, maze, T_RC, mat_thickness=25)

    start = maze_f_robot_coord[0].copy()
    start[2] += 0.05

    arm_vector = arm_vector_decider(maze_f_robot_coord, normal_vector, robot)
    if arm_vector is None:
        raise Exception('No valid arm vector found')

    print("Planned Trajectory:")
    i = 0
    poses = []
    for point in maze_f_robot_coord:
        pose = PoseComposer.make_se3_matrix(point, arm_vector, normal_vector)
        poses.append(pose)
        print(f"{i}: {pose[:3, -1]}")
        i += 1
    a = input("Do you agree with the trajectory? [Y/n]")
    if a.lower() != 'y':
        raise Exception("Aborted")

    start_pose = PoseComposer.make_se3_matrix(start, arm_vector, normal_vector)
    move_robot_to(start_pose, robot)

    for pose in poses:
        move_robot_to(pose, robot)

    sleep(2)

    for pose in poses[::-1]:
        move_robot_to(pose, robot)

    move_robot_to(start_pose, robot)
    robot.soft_home()

    return 1


def solution(puzzle, robot, T_RC, arm_vector=np.array([-1, 1, 0]), normal_vector=np.array([0, 0, 1])):
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
    # arm_vector = np.array([-1, 1, 0])
    # normal_vector = np.array([0, 0, 1])
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
