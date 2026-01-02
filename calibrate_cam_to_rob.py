import numpy as np
import cv2

robot_xyz = None
image_uv = None

calibration_path = "camera_intrinsics.npz"
calibration_data = np.load("camera_intrinsics.npz")
camera_matrix = calibration_data["camera_matrix"]
dist_coeffs = calibration_data["dist_coeffs"]

obj = np.array(robot_xyz, dtype=np.float64).reshape(-1, 3)

img = np.array(image_uv, dtype=np.float64).reshape(-1, 2)

K = camera_matrix
dist = dist_coeffs

ok, rvec, tvec, inliers = cv2.solvePnPRansac(
    obj, img, K, dist,
    flags=cv2.SOLVEPNP_ITERATIVE,
    reprojectionError=3.0,
    iterationsCount=200
)
assert ok

obj_in = obj[inliers[:,0]]
img_in = img[inliers[:,0]]
ok, rvec, tvec = cv2.solvePnP(obj_in, img_in, K, dist, rvec, tvec, True)

R, _ = cv2.Rodrigues(rvec)

# kamera -> robot
R_RC = R.T
t_RC = -R.T @ tvec

T_RC = np.eye(4)
T_RC[:3,:3] = R_RC
T_RC[:3, 3] = t_RC.ravel()
print(T_RC)
