This is the semestral project of Václav Beran, Mathias Palme and Tomáš Janoušek. 

puzzle.py contains code describing each mazeas series of points in space, originating from the center of the base
component.py contains supporting methods for puzzle.py

PoseComposer.py has method make_se3_matrix that creates Pose matrix, usable in IK. This requires the point where the circle must be, normal of the circle and direction from the point to the robot arm.
It also contains find_options_around_point which returns points in a circle around a point in the plane defined by its normal vector. The radius and point density is configurable, as well as only certain part of the circle can be generated, using the number of points and angle.