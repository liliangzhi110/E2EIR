import cv2
import numpy as np

for i in range(100):
    points1 = np.float32([[0, 0], [128, 0], [128, 128], [0, 128]])
    points2 = np.float32([[0,np.random.randint(0,50,size=(1,))[0]],
                                  [np.random.randint(80,128,size=(1,))[0], 0],
                                  [128, np.random.randint(85,128,size=(1,))[0]],
                                  [np.random.randint(0,40,size=(1,))[0], 128]])
    matrix = cv2.getPerspectiveTransform(points1, points2)

    print(matrix)
    print("")

