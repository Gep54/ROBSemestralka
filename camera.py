import cv2
import numpy as np
import cv2.aruco as aruco
import os

def z_axis_rotation(z_angle):
    R = np.array([[np.cos(z_angle), -np.sin(z_angle), 0], 
                 [np.sin(z_angle), np.cos(z_angle), 0], 
                 [0, 0, 1]])
    return R

def rotationMatrixToEulerXYZ(R):
    x = np.arctan2(R[1, 0], R[0, 0])
    y = np.arcsin(-R[2, 0])
    z = np.arctan2(R[2, 1], R[2, 2])
    return x, y, z

class Camera:
    def __init__(self):
        self.aruco_size = 20.0
        self.calibration_path = "camera_intrinsics.npz"
        self.calibration_data = np.load("camera_intrinsics.npz")
        self.intrinsic_matrix = self.calibration_data["camera_matrix"]
        self.dist_coeffs = self.calibration_data["dist_coeffs"]
    
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()

    def detect_markers(self, img, draw=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.parameters
        )
        
        if draw and ids is not None:
            aruco.drawDetectedMarkers(img, corners, ids)
        
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners,
            self.aruco_size,
            self.intrinsic_matrix,
            self.dist_coeffs
        )
        
        return rvecs, tvecs, ids

    def get_mid_points(self, img):
        rvecs, tvecs, ids = self.detect_markers(img)
        print(ids)
        # return None, None
        # if len(markers_coords) != 2:
        if [1] in ids and [2] in ids:
            mid_point = np.mean(tvecs, axis = 0)
            R, _ = cv2.Rodrigues(rvecs[0])
            x_rot, y_rot, z_rot = rotationMatrixToEulerXYZ(R)
            # print("rot: ", x_rot*57.2957795, y_rot*57.2957795, z_rot*57.2957795)
            R = z_axis_rotation(-1 * x_rot - np.pi)
            return mid_point[0], R
        else:
            return None, None
        
            



if __name__ == "__main__":
    # img = cv2.imread(os.path.join("podlozka", "Image__2025-11-26__11-22-00.bmp"))
    img = cv2.imread(os.path.join("podlozka", "Image__2025-11-26__11-23-54.bmp"))
    camera = Camera()
    mid_point, R = camera.get_mid_points(img)
    # rvecs, tvecs, ids = camera.detect_markers(img, draw=True)
    
    print(mid_point)
    print(R)

    # print(rvecs)
    # print(img.shape)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # intrinsic = np.load("camera_intrinsics.npz")
    # print(intrinsic["camera_matrix"])
    
    # pass