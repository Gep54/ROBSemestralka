import cv2
import cv2.aruco as aruco


def z_axis_rotation(z_angle):
    R = np.array([[np.cos(z_angle), -np.sin(z_angle), 0],
                  [np.sin(z_angle), np.cos(z_angle), 0],
                  [0, 0, 1]])
    return R


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return roll, pitch, yaw


class Camera:
    def __init__(self):
        self.aruco_size = 40.0
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

        if draw and ids is not None:
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(
                    img,
                    self.intrinsic_matrix,
                    self.dist_coeffs,
                    rvec,
                    tvec,
                    self.aruco_size * 0.5  # axis length in meters
                )

        return rvecs, tvecs, ids

    def get_mid_points(self, img):
        rvecs, tvecs, ids = self.detect_markers(img)
        print(ids)
        if [1] in ids and [2] in ids:
            mid_point = np.mean(tvecs, axis=0)
            R, _ = cv2.Rodrigues(rvecs[0])
            roll, pitch, yaw = rotationMatrixToEulerAngles(R)
            # print(roll, pitch, yaw)
            R = z_axis_rotation(-yaw)
            return mid_point[0], R
        else:
            return None, None

    def transformFromCameraToRobot(self, robot_xyz, image_uv):
        # objectPoints: Nx3 v robotické bázi (např. mm nebo m, ale konzistentně)
        obj = np.array(robot_xyz, dtype=np.float64).reshape(-1, 3)

        # imagePoints: Nx2 pixely
        img = np.array(image_uv, dtype=np.float64).reshape(-1, 2)

        # K, dist: z intrinsics kalibrace
        K = self.intrinsic_matrix
        dist = self.dist_coeffs
        # dist = np.zeros((5, 1))

        # robustně:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, K, dist,
            flags=cv2.SOLVEPNP_SQPNP,
            reprojectionError=8.0,
            iterationsCount=400
        )
        # ok, rvec, tvec = cv2.solvePnP(
        #     obj, img, K, dist,
        #     flags=cv2.SOLVEPNP_EPNP
        # )
        assert ok

        # dooptimalizace jen na inliers:
        obj_in = obj[inliers[:, 0]]
        img_in = img[inliers[:, 0]]
        ok, rvec, tvec = cv2.solvePnP(obj_in, img_in, K, dist, rvec, tvec, True)

        R, _ = cv2.Rodrigues(rvec)

        # kamera -> robot
        R_RC = R.T
        t_RC = -R.T @ tvec

        T_RC = np.eye(4)
        T_RC[:3, :3] = R_RC
        T_RC[:3, 3] = t_RC.ravel()
        # print(T_RC)
        return T_RC

    def cameraToRobot(self, cameraCoord, T_RC=None):
        """
        returns position of aruco code middle in robot coordinates in m
        :param cameraCoord:
        :return:
        """

        if T_RC is None:
            T_RC = [[-4.91410691e-03, 9.98341087e-01, -5.73665839e-02, 4.90644721e+02],
                    [9.97906036e-01, 1.19594619e-03, -6.46692634e-02, 2.38816422e+01],
                    [-6.44933754e-02, -5.75642520e-02, -9.96256474e-01, 1.19549166e+03],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        p_cam_h = np.append(cameraCoord, 1.0)
        p_robot_h = T_RC @ p_cam_h
        p_robot = p_robot_h[:3] * 0.001
        return p_robot


if __name__ == "__main__":
    # img = cv2.imread(os.path.join("podlozka", "Image__2025-11-26__11-22-00.bmp"))
    # img = cv2.imread(os.path.join("podlozka", "Image__2025-11-26__11-23-54.bmp"))
    camera = Camera()
    # mid_point, R = camera.get_mid_points(img)
    # rvecs, tvecs, ids = camera.detect_markers(img, draw=True)
    #
    # print(mid_point)
    # print(R)
    #
    # # print(rvecs)
    # # print(img.shape)
    #
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # intrinsic = np.load("camera_intrinsics.npz")
    # print(intrinsic["camera_matrix"])

    spacePoints = [[0.53, 0.0, 0.2], [0.53, 0.12, 0.2], [0.5, -0.1, 0.3], [0.4, -0.1, 0.3], [0.42, 0.05, 0.3],
                   [0.4, -0.13, 0.4], [0.35, -0.05, 0.2]]
    pixelPoints = [[1025., 1012.], [1595., 1002.], [509., 861.], [507., 340.], [1311., 445.], [260., 285.],
                   [817., 155.]]

    import numpy as np

    obj = np.asarray(spacePoints, dtype=np.float64).reshape(-1, 3)
    obj_mm = obj * 1000.0
    img = np.asarray(pixelPoints, dtype=np.float64).reshape(-1, 2)

    result = camera.transformFromCameraToRobot(obj_mm, img)
    print(result)
