import numpy as np

#this method takes the direction the cirle should be facing and its normal and computes the rotational matrix for the robots end effector, returns it as numpy array
#direction is the difference between target position and robots arm position, it should be selected on a 14cm circle around the target. dir = t - r
def main(direction, normal):
    z = normal / np.linalg.norm(normal)
    x = -direction / np.linalg.norm(direction) #cirle is extended in the -x direction from the robots arm
    y = np.cross(z, x) #y is orthogonal to both x and z
    return np.array([x, y, z].T)

def 