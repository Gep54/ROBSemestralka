import numpy as np

class Component:
    def __init__(self):
        pass
    def get_descrete_points(self, num_of_points:int = 10)->np.array:
        pass

class Line(Component):
    def __init__(self, p1:np.array, p2:np.array):
        self.p1 = p1 # line start
        self.p2 = p2 # line end
    
    def get_descrete_points(self, num_of_points:int = 10)->np.array:
        t = np.linspace(0, 1, num_of_points)

        points = (1 - t)[:, None] * self.p1 + t[:, None] * self.p2
        return points
    
    def get_normal_plane(self):
        # Direction vector of line
        d = self.p2 - self.p1
        d = d / np.linalg.norm(d)

        # Pick any vector not parallel to d
        if abs(d[0]) < 0.9:
            tmp = np.array([1, 0, 0])
        else:
            tmp = np.array([0, 1, 0])

        # Compute two orthonormal basis vectors perpendicular to d
        u = np.cross(d, tmp)
        u = u / np.linalg.norm(u)

        v = np.cross(d, u)

        # Plane origin is p1 (any point on line)
        origin = self.p1

        return origin, d, u, v
    
class Arc(Component):
    def __init__(self, center:np.array, diameter:float, angle_start:float, angle_end:float, plane:str):
        self.center = center
        self.radius = diameter / 2.0
        self.angle_start = angle_start
        self.angle_end = angle_end
        self.plane = plane

    def get_descrete_points(self, num_of_points: int = 10)->np.array:
        angles = np.linspace(self.angle_start, self.angle_end, num_of_points)

        if self.plane == "XY":
            x = self.center[0] + self.radius * np.cos(angles)
            y = self.center[1] + self.radius * np.sin(angles)
            z = np.full_like(angles, self.center[2])

        elif self.plane == "XZ":
            x = self.center[0] + self.radius * np.cos(angles)
            y = np.full_like(angles, self.center[1])
            z = self.center[2] + self.radius * np.sin(angles)

        elif self.plane == "YZ":
            # výpočet bodů v rovině YZ
            x = np.full_like(angles, self.center[0])
            y = self.center[1] + self.radius * np.cos(angles)
            z = self.center[2] + self.radius * np.sin(angles)
        else:
            raise NotImplementedError
        return np.vstack([x, y, z]).T