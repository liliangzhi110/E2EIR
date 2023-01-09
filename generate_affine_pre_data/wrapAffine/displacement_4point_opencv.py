import cv2
from skimage.external.tifffile import TiffFile
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line

np.set_printoptions(suppress=True)
plt.rcParams['font.sans-serif']=['SimHei']


image_cv2=cv2.imread("C:\\Users\\lilia\\Desktop\\data.tif",cv2.IMREAD_UNCHANGED)

image_cv2=np.pad(np.array(image_cv2),((200,200),(200,200),(0,0)),constant_values=0)

rr,cc=line(400,400,400,600)

image_cv2[rr,cc]=(255,0,0)

print(image_cv2.shape)

cv2.imshow('result.jpg',image_cv2)


# #其他四点变换
point1_256=np.float32([[400,400],[600,400],[600,600],[400,600]])
point2_256=np.float32([[ 402, 398.], [603, 402],[598 , 596 ],[ 400 ,605 ]])
matrix1 = cv2.getPerspectiveTransform(point1_256,point2_256)
output1 = cv2.warpPerspective(image_cv2, matrix1, (1000, 1000))

rr,cc=line(400,400,400,600)

output1[rr,cc]=(255,0,0)




#对矩阵求逆，再变换
matrix2 = cv2.getPerspectiveTransform(point2_256,point1_256)
output2 = cv2.warpPerspective(image_cv2, matrix2, (1000, 1000))



matrix3 = cv2.getPerspectiveTransform(point1_256,point2_256)
output3 = cv2.warpPerspective(output2, matrix3, (1000, 1000))


images=np.hstack([output1,output2])

cv2.imshow('result.jpg',images)

cv2.waitKey(0)