import cv2
from skimage.external.tifffile import TiffFile
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
plt.rcParams['font.sans-serif']=['SimHei']

image_cv2=cv2.imread("C:\\Users\\lilia\\Desktop\\santa_cruz_az-band1.tif",cv2.IMREAD_UNCHANGED)

plt.subplot(3,2,1)
plt.title('原图片')
plt.imshow(image_cv2)


points1 = np.float32([[0,0],[1000,0],[1000,1000],[0,1000]])
points2 = np.float32([[0,200],[870,0],[1000,750],[160,1000]])
matrix = cv2.getPerspectiveTransform(points1,points2)
output = cv2.warpPerspective(image_cv2, matrix, (2000, 2000))


#投影变换
plt.subplot(3,2,2)
plt.title('投影变换')
plt.imshow(output)


#其他四点变换
point1_256=np.float32([[400,400],[600,400],[600,600],[400,600]])
point2_256=np.float32([[ 405, 359.], [610, 410],[606 , 590 ],[ 400 ,605 ]])
matrix1 = cv2.getPerspectiveTransform(point1_256,point2_256)
output1 = cv2.warpPerspective(image_cv2, matrix1, (2000, 2000))




plt.subplot(3,2,3)
plt.title('H(AB)4点变换')
plt.imshow(output1)



#对矩阵求逆，再变换
matrix2 = cv2.getPerspectiveTransform(point2_256,point1_256)
output2 = cv2.warpPerspective(image_cv2, matrix2, (2000, 2000))

plt.subplot(3,2,4)
plt.title('H(BA)4点变换')
plt.imshow(output2)



matrix3 = cv2.getPerspectiveTransform(point1_256,point2_256)
output3 = cv2.warpPerspective(output2, matrix3, (2000, 2000))

plt.subplot(3,2,5)
plt.title('用H(AB)变换H(BA)的结果')
plt.imshow(output3)




plt.show()