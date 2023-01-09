import cv2
from skimage.external.tifffile import TiffFile
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line

np.set_printoptions(suppress=True)
plt.rcParams['font.sans-serif']=['SimHei']


image_cv2=cv2.imread("C:\\Users\\lilia\\Desktop\\dataresult\\origin\\data.png",cv2.IMREAD_UNCHANGED)

image_cv2=np.pad(np.array(image_cv2),((200,300),(200,300),(0,0)),constant_values=0)
print(image_cv2.shape)


rr,cc=line(500,500,500,900)
image_cv2[rr,cc]=(0,0,255)

rr,cc=line(500,900,900,900)
image_cv2[rr,cc]=(0,0,255)

rr,cc=line(900,900,900,500)
image_cv2[rr,cc]=(0,0,255)

rr,cc=line(900,500,500,500)
image_cv2[rr,cc]=(0,0,255)

plt.subplot(2,2,1)
plt.title('原图片')
plt.imshow(image_cv2)
cv2.imwrite('C:\\Users\\lilia\\Desktop\\data\\origin.tif',image_cv2)


#其他四点变换
# point1_256=np.float32([[400,400],[600,400],[600,600],[400,600]])
# point2_256=np.float32([[402,398],[603,402],[598,596 ],[400,605]])

point1_256=np.float32([[500,500],[900,500],[900,900],[500,900]])
point2_256=np.float32([[485,480],[920,490],[880,880 ],[488,879]])

matrix1 = cv2.getPerspectiveTransform(point1_256,point2_256)
output1 = cv2.warpPerspective(image_cv2, matrix1, (1500, 1500))


p_400_400_displace=np.matmul(matrix1,np.array([1200,1200,1]).reshape((3,1)))
print(matrix1)
print('')
print(p_400_400_displace/p_400_400_displace[2,0])




rr,cc=line(500,500,500,900)
output1[rr,cc]=(0,255,0)

rr,cc=line(500,900,900,900)
output1[rr,cc]=(0,255,0)

rr,cc=line(900,900,900,500)
output1[rr,cc]=(0,255,0)

rr,cc=line(900,500,500,500)
output1[rr,cc]=(0,255,0)


rr,cc=line(200,200,200,1200)
output1[rr,cc]=(0,255,0)

rr,cc=line(200,1200,1200,1200)
output1[rr,cc]=(0,255,0)

rr,cc=line(1200,1200,1200,200)
output1[rr,cc]=(0,255,0)

rr,cc=line(1200,200,200,200)
output1[rr,cc]=(0,255,0)

plt.subplot(2,2,2)
plt.title('H(AB)4点变换')
plt.imshow(output1)
cv2.imwrite('C:\\Users\\lilia\\Desktop\\data\\origin_H(AB).tif',output1)




#对矩阵求逆，再变换
matrix2 = cv2.getPerspectiveTransform(point2_256,point1_256)
output2 = cv2.warpPerspective(image_cv2, matrix2, (1500, 1500))


plt.subplot(2,2,3)
plt.title('H(BA)4点变换')
plt.imshow(output2)
cv2.imwrite('C:\\Users\\lilia\\Desktop\\data\\origin_H(BA).tif',output2)



matrix3 = cv2.getPerspectiveTransform(point1_256,point2_256)
output3 = cv2.warpPerspective(output2, matrix3, (1500, 1500))

plt.subplot(2,2,4)
plt.title('用H(AB)变换H(BA)的结果')
plt.imshow(output3)
cv2.imwrite('C:\\Users\\lilia\\Desktop\\data\\origin_H(AB)_H(BA).tif',output3)

#
# plt.show()