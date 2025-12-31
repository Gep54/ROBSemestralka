import numpy as np
from numpy.typing import NDArray


def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def make_se3_matrix(circle_pos: NDArray, circle_arm_vector: NDArray, circle_normal_vector: NDArray,
                    tolerance: float = 0.001, circle_arm_length: float = 0.135):
    # TODO: write docstring
   

    # check that the normal and arm vectors are perpendicular
    assert (np.isclose(np.dot(circle_arm_vector, circle_normal_vector), 0.0, atol=tolerance))
    # directional vectors cannot be 0
    assert (np.linalg.norm(circle_arm_vector) != 0 and np.linalg.norm(circle_normal_vector) != 0)

    z = normalize(-circle_normal_vector)
    x = normalize(circle_arm_vector)
    y = np.cross(z, x)

    # rotation_matrix = np.array([x, y, z]).T #for potential use
    arm_position = circle_pos + x * circle_arm_length  # TODO: + or minus?

    x_extend = np.append(x, 0.0)
    y_extend = np.append(y, 0.0)
    z_extend = np.append(z, 0.0)
    arm_position_extend = np.append(arm_position, 1.0)
    se3 = np.array([x_extend, y_extend, z_extend, arm_position_extend]).T
    return se3


def find_perpendicular_vector(v):
    # v is the vector along which axis rotation happens
    # Find a vector perpendicular to v by crossing v with a non-collinear reference vector
    ref_vector = np.array([1, 0, 0])  # Reference vector
    if np.allclose(v, ref_vector):  # If v is collinear with reference vector
        ref_vector = np.array([0, 1, 0])  # Choose a different reference vector
    return np.cross(v, ref_vector)


def rotate_vector(v, axis, angle):
    # Rotate vector `v` around `axis` by `angle` degrees using Rodrigues' rotation formula
    axis = axis / np.linalg.norm(axis)  # Normalize axis
    angle_rad = np.radians(angle)  # Convert angle to radians
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    return (
            v * cos_theta +
            np.cross(axis, v) * sin_theta +
            axis * np.dot(axis, v) * (1 - cos_theta)
    )


def find_options_around_point(circle_pos: NDArray, circle_normal_vector: NDArray, angle: float = 45, n_steps: int = 8,
                                first_vector: NDArray = None, tolerance: float = 0.001, arm_length: float = 0.14):
    # TODO: docstring
    assert (first_vector is None or np.isclose(np.dot(first_vector, circle_normal_vector), 0.0, atol=tolerance))

    circle_n = normalize(circle_normal_vector)
    if first_vector is None:
        first_vector = normalize(find_perpendicular_vector(circle_n))

    arm_positions = [circle_pos + first_vector * arm_length]
    circle_arm_vectors = [first_vector]
    new_vector = first_vector
    for i in range(1, n_steps):
        new_vector = rotate_vector(new_vector, circle_n, angle)
        arm_positions.append(circle_pos + new_vector * arm_length)
        circle_arm_vectors.append(new_vector)

    return arm_positions, circle_arm_vectors
