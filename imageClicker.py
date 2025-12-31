import cv2
import numpy as np

points = []

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Clicked: {x}, {y}")

img = cv2.imread("photos/foto7.jpg")
cv2.imshow("image", img)
cv2.setMouseCallback("image", mouse_cb)

cv2.waitKey(0)
cv2.destroyAllWindows()

points = np.array(points, dtype=np.float64)
print(points)