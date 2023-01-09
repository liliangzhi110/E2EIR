import cv2
from skimage.external.tifffile import TiffFile
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei']

image_cv2=cv2.imread("C:\\Users\\kylenate\\Desktop\\paper_registration1114\\santa_cruz_az-band7.tif",cv2.IMREAD_UNCHANGED)

plt.subplot(2,2,1)
plt.title('原图片')
plt.imshow(image_cv2)


points1 = np.float32([[0,0],[1000,0],[1000,1000],[0,1000]])
points2 = np.float32([[0,200],[870,0],[1000,750],[160,1000]])
matrix = cv2.getPerspectiveTransform(points1,points2)
output = cv2.warpPerspective(image_cv2, matrix, (1000, 1000))

print(matrix)

#投影变换
plt.subplot(2,2,2)
plt.title('投影变换')
plt.imshow(output)


#点逆变换
points1=np.float32([[0,200],[870,0],[1000,750],[160,1000]])
points2=np.float32([[0,0],[1000,0],[1000,1000],[0,1000]])

matrix1 = cv2.getPerspectiveTransform(points1,points2)
output1 = cv2.warpPerspective(output, matrix1, (1000, 1000))

print('')
print(matrix1)
plt.subplot(2,2,3)
plt.title('点逆变换')
plt.imshow(output1)


#对矩阵求逆，再变换
output3=cv2.warpPerspective(output,np.linalg.inv(matrix),(1000,1000))
plt.subplot(2,2,4)
plt.title('矩阵求逆再变换')
plt.imshow(output3)
plt.show()













