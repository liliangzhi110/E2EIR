import cv2
import numpy as np

image_cv2_band7=cv2.imread("D:\\Second_paper_VAE_CNN\\xian_image\\xi_an_sub10000band1.tif",cv2.IMREAD_UNCHANGED)
image_cv2_band1=cv2.imread("D:\\Second_paper_VAE_CNN\\xian_image\\xi_an_sub10000band3.tif",cv2.IMREAD_UNCHANGED)




image1=[]

for i in range(3,image_cv2_band7.shape[0]-3):
    for j in range(2,image_cv2_band7.shape[1]-3):
        print(i)
        temp=[np.square(image_cv2_band7[i][j+3]-image_cv2_band7[i][j]),
              np.square(image_cv2_band7[i][j-3]-image_cv2_band7[i][j]),
              np.square(image_cv2_band7[i-3][j]-image_cv2_band7[i][j]),
              np.square(image_cv2_band7[i+3][j]-image_cv2_band7[i][j])]
        temp=np.array(temp)
        image1.append(np.mean(temp))


image2=[]

for i in range(3,image_cv2_band1.shape[0]-3):
    for j in range(2,image_cv2_band7.shape[1]-3):
        print(i)
        temp=[np.square(image_cv2_band1[i][j+3]-image_cv2_band1[i][j]),
              np.square(image_cv2_band1[i][j-3]-image_cv2_band1[i][j]),
              np.square(image_cv2_band1[i-3][j]-image_cv2_band1[i][j]),
              np.square(image_cv2_band1[i+3][j]-image_cv2_band1[i][j])]
        temp=np.array(temp)
        image2.append(np.mean(temp))


image1=np.array(image1)
image2=np.array(image2)


print(image1-image2)


