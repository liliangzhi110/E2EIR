import cv2
import numpy as np

image_cv2_band7=cv2.imread("D:\\ProgramData_second\\Similarityfunction\\data\\origin_image\\xi_an_sub10000band1.tif",cv2.IMREAD_UNCHANGED)
image_cv2_band1=cv2.imread("D:\\ProgramData_second\\Similarityfunction\\data\\origin_image\\xi_an_sub10000band3.tif",cv2.IMREAD_UNCHANGED)

image_cv2_band7=image_cv2_band7[0:100,0:100]/255
image_cv2_band1=image_cv2_band1[0:100,0:100]/255
print(np.max(image_cv2_band7),np.min(image_cv2_band1))

image1=[]

for i in range(3,image_cv2_band7.shape[0]-3):
    for j in range(2,image_cv2_band7.shape[1]-3):

        temp=[np.fabs(image_cv2_band7[i][j+3]-image_cv2_band7[i][j]),
              np.fabs(image_cv2_band7[i][j-3]-image_cv2_band7[i][j]),
              np.fabs(image_cv2_band7[i-3][j]-image_cv2_band7[i][j]),
              np.fabs(image_cv2_band7[i+3][j]-image_cv2_band7[i][j])]
        temp=np.array(temp)
        image1.append(np.mean(temp))


image2=[]

for i in range(3,image_cv2_band1.shape[0]-3):
    for j in range(2,image_cv2_band7.shape[1]-3):

        temp=[np.fabs(image_cv2_band1[i][j+3]-image_cv2_band1[i][j]),
              np.fabs(image_cv2_band1[i][j-3]-image_cv2_band1[i][j]),
              np.fabs(image_cv2_band1[i-3][j]-image_cv2_band1[i][j]),
              np.fabs(image_cv2_band1[i+3][j]-image_cv2_band1[i][j])]
        temp=np.array(temp)
        image2.append(np.mean(temp))


image1=np.array(image1)
image2=np.array(image2)

print('直接减法',np.sum(image_cv2_band7-image_cv2_band1))
print('相似性计算',np.sum(image1-image2))