spacePoints = [[0.53, 0.0, 0.2], [0.53, 0.12, 0.2], [0.5, -0.1, 0.3], [0.4, -0.1, 0.3],[0.42, 0.05, 0.3],[0.4, -0.13, 0.4],[0.35, -0.05, 0.2]]
pixelPoints = [[1025., 1012.],[1595. ,1002.],[ 509. , 861.],[507. ,340.], [1311.,  445.],[260. ,285.],[817. ,155.]]

import numpy as np

obj = np.asarray(spacePoints, dtype=np.float64).reshape(-1, 3)
img = np.asarray(pixelPoints, dtype=np.float64).reshape(-1, 2)

print(obj.shape,  # (7, 3)
img.shape  # (7, 2)
)