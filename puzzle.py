import matplotlib.pyplot as plt
import numpy as np

from components import Line, Arc
from camera import Camera
import cv2
import os

# img = cv2.imread(os.path.join("podlozka", "Image__2025-11-26__11-22-06.bmp"))
# img = cv2.imread(os.path.join("podlozka", "Image__2025-11-26__11-23-54.bmp"))
# img = cv2.imread(os.path.join("podlozka", "Image__2025-11-26__11-21-48.bmp"))

# img = cv2.imread(os.path.join("podlozka", "Image__2025-11-26__11-23-27.bmp"))

# camera = Camera()
# rvecs, tvecs, ids = camera.detect_markers(img, draw=True)
# cv2.imshow("img", img)
# mid_point, R = camera.get_mid_points(img)

 

class Puzzle:
    """
        Parent Puzzle class - defines general template for other children puzzles
        Defines unified methods get_points() and show() som
    """

    def __init__(self, rotation):
        self.puzzle_type: str = None
        self.components: list[Line, Arc] = None
        self.R = rotation

    def _init_components(self) -> list[Line, Arc]:
        # NOTE that the return value is a list, if only one component is returned
        # do not forget to pack it in a list! e.g. PuzzleA
        pass

    def get_forward_trajectory(self) -> np.array:
        points = [c.get_descrete_points() for c in self.components]

        points = np.flip(np.vstack([*points]))
        # M = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        # points = [M @ p for p in points]
        #
        # points = [self.R @ p for p in points] # Rotation

        return points   

    def get_reverse_trajectory(self, number_of_points=10) -> np.array:
        points = [c.get_descrete_points(num_of_points=number_of_points) for c in self.components]

        points = np.vstack([*points])        
        # points = [self.R @ p for p in points] # Rotation
        # print(points.shape)
        return points

    def show_forward_trajectory(self) -> None:
        points = self.get_forward_trajectory()
        
        points = np.array(points)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        # ax.scatter(0, 0, 0)
        ax.quiver(0, 0, 0, 100, 0, 0, color='red')
        ax.quiver(0, 0, 0, 0, 100, 0, color='green')
        ax.quiver(0, 0, 0, 0, 0, 100, color='blue')
        ax.set_aspect('equal')
        plt.show()

    def show_reverse_trajectory(self) -> None:
        points = self.get_reverse_trajectory()
        points = np.array(points)
        # points = [R @ p for p in points] # Rotation
        
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        ax.quiver(0, 0, 0, 100, 0, 0, color='red')
        ax.quiver(0, 0, 0, 0, 100, 0, color='green')
        ax.quiver(0, 0, 0, 0, 0, 100, color='blue')
        ax.set_aspect('equal')
        plt.show()


class PuzzleA(Puzzle):
    def __init__(self):
        self.puzzle_type = "B"
        self.components = self._init_components()

    def _init_components(self) -> list:
        # ----- line1 -----
        p1 = np.array([0, 0, 20])
        p2 = np.array([0, 0, 200])
        line1 = Line(p1, p2)

        return (line1,)


class PuzzleB(Puzzle):
    def __init__(self):
        self.puzzle_type = "B"
        self.components = self._init_components()

    def _init_components(self) -> list:
        # ----- line1 -----
        p1 = np.array([0, 0, 20])
        p2 = np.array([0, 0, 80])
        line1 = Line(p1, p2)

        # ----- line2 -----
        p1 = np.array([0, 0, 80])
        p2 = np.array([70, 0, 150])
        line2 = Line(p1, p2)

        # ----- line3 -----
        p1 = np.array([70, 0, 150])
        p2 = np.array([70, 0, 200])
        line3 = Line(p1, p2)

        return line1, line2, line3


class PuzzleC(Puzzle):
    def __init__(self):
        self.puzzle_type = "C"
        self.components = self._init_components()

    def _init_components(self) -> list:
        # ----- line1 -----
        p1 = np.array([0, 0, 20])
        p2 = np.array([0, 0, 50])
        line1 = Line(p1, p2)

        # ----- line2 -----
        p1 = np.array([0, 0, 50])
        p2 = np.array([-50, 0, 100])
        line2 = Line(p1, p2)

        # ----- line3 -----
        p1 = np.array([-50, 0, 100])
        p2 = np.array([-50, 50, 150])
        line3 = Line(p1, p2)

        # ----- line4 -----
        p1 = np.array([-50, 50, 150])
        p2 = np.array([-50, 50, 200])
        line4 = Line(p1, p2)
        return line1, line2, line3, line4


class PuzzleD(Puzzle):
    def __init__(self):
        self.puzzle_type = "D"
        self.components = self._init_components()

    def _init_components(self) -> list:
        # ----- line1 -----
        p1 = np.array([0, 0, 0])
        p2 = np.array([0, 0, 30])
        line1 = Line(p1, p2)

        # ----- line2 -----
        p1 = np.array([0, 0, 30])
        p2 = np.array([-15, 0, 45])
        line2 = Line(p1, p2)

        # ----- line3 -----
        p1 = np.array([-15, 0, 45])
        p2 = np.array([-15, 0, 105])
        line3 = Line(p1, p2)

        # ------ arc1 ------
        center = np.array([35, 0, 105])
        diameter = 100
        angle_start = np.pi / 2
        angle_end = np.pi
        arc1 = Arc(center, diameter, angle_start, angle_end, plane="XZ")

        # ----- line4 -----
        p1 = np.array([35, 0, 155])
        p2 = np.array([85, 0, 155])
        line4 = Line(p1, p2)

        return line1, line2, line3, arc1, line4


class PuzzleE(Puzzle):
    def __init__(self):
        self.puzzle_type = "E"
        self.components = self._init_components()

    def _init_components(self) -> list:
        # ----- line1 -----
        p1 = np.array([0, 0, 0])
        p2 = np.array([0, 0, 30])
        line1 = Line(p1, p2)

        # ----- line2 -----
        p1 = np.array([0, 0, 30])
        p2 = np.array([-17.321, 0, 60])
        line2 = Line(p1, p2)

        # ----- line3 -----
        p1 = np.array([-17.321, 0, 60])
        p2 = np.array([-17.321, -50, 110])
        line3 = Line(p1, p2)

        # ------ arc1 ------
        center = np.array([-17.321, 0, 110])
        diameter = 100
        angle_start = np.pi / 2
        angle_end = np.pi
        arc1 = Arc(center, diameter, angle_start, angle_end, plane="YZ")

        # ----- line4 -----
        p1 = np.array([-17.321, 0, 160])
        p2 = np.array([32.679, 50, 160])
        line4 = Line(p1, p2)

        # ----- line5 -----
        p1 = np.array([32.679, 50, 160])
        p2 = np.array([32.679, 100, 160])
        line5 = Line(p1, p2)

        return line1, line2, line3, arc1, line4, line5


if __name__ == "__main__":
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    puzzle = PuzzleD()
    puzzle.R = R
    # points = puzzle.get_forward_trajectory()
    # puzzle.show_forward_trajectory()
    points = puzzle.get_reverse_trajectory()
    puzzle.show_reverse_trajectory()
