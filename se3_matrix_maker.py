from ctu_crs import CRS97
import numpy as np
from numpy.typing import NDArray


def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector  # Vrátí původní vektor, pokud je norm = 0
    return vector / norm

def make_se3_matrix(circle_pos: NDArray, circle_arm_vector: NDArray, circle_normal_vector: NDArray,
                    toleranc: float = 0.001, circle_arm_length: float = 0.14):
    # TODO: write docstring
    # TODO: figure out cirle_arm length
    assert (np.isclose(np.dot(circle_arm_vector, circle_normal_vector), 0.0,
                       atol=toleranc))  # check that the normal and arm vectors are perpendicular
    z = normalize(-circle_normal_vector)

    x = normalize(circle_arm_vector)
    y = np.cross(x,
                 z)  # TODO: this needs to be checked, x and z might need to be switched, depending ot the axis orientations
    rotation_matrix = np.array([x, y, z]).T
    arm_position = circle_pos + x * circle_arm_length  # TODO: + or minus?

    x_extend = np.append(x, 0.0)
    y_extend = np.append(y, 0.0)
    z_extend = np.append(z, 0.0)
    arm_position_extend = np.append(arm_position, 1.0)
    se3 = np.array([x_extend, y_extend, z_extend, arm_position_extend]).T
    return se3