import cv2
import numpy as np
import matplotlib.pyplot as plt
data=np.load('C:\\Users\\kylenate\\Desktop\\paper_registration1114\\land_sat.npz')['data_image']

image_cv2_band7=data[0:1,0:100,0:100].reshape((100,100))
# image_cv2_band1=data[5:6,0:100,0:100].reshape((100,100))
image_cv2_band1=data[1:2,100:200,0:100].reshape((100,100))



plt.subplot(1,2,1)
plt.imshow(image_cv2_band7)

plt.subplot(1,2,2)
plt.imshow(image_cv2_band1)
plt.show()


def cicle(row,col):
    point1 = (row -3, col)
    point2 = (row -3, col +1)
    point3 = (row -2, col +2)
    point4 = (row -1, col +3)
    point5 = (row, col +3)
    point6 = (row +1, col +3)
    point7 = (row +2, col +2)
    point8 = (row +3, col +1)
    point9 = (row +3, col)
    point10 = (row +3, col -1)
    point11 = (row +2, col -2)
    point12 = (row +1, col -3)
    point13 = (row, col -3)
    point14 = (row -1, col -3)
    point15 = (row -2, col -2)
    point16 = (row -3, col -1)

    con_point=[point1, point2 ,point3 ,point4 ,point5 ,point6 ,point7 ,point8 ,point9 ,point10 ,point11 ,point12, point13
        ,point14 ,point15 ,point16]

    return con_point


image1=[]

for i in range(3,image_cv2_band7.shape[0]-3):
    for j in range(2,image_cv2_band7.shape[1]-3):

        temp=[

            np.fabs(image_cv2_band7[i-3][j]-image_cv2_band7[i][j]),
              np.fabs(image_cv2_band7[i-3][j+1]-image_cv2_band7[i][j]),
              np.fabs(image_cv2_band7[i-2][j+2]-image_cv2_band7[i][j]),
              np.fabs(image_cv2_band7[i-1][j+3]-image_cv2_band7[i][j]),
              np.fabs(image_cv2_band7[i][j+3] - image_cv2_band7[i][j]),
              np.fabs(image_cv2_band7[i +1][j+3] - image_cv2_band7[i][j]),
              np.fabs(image_cv2_band7[i + 2][j+2] - image_cv2_band7[i][j]),
              np.fabs(image_cv2_band7[i + 3][j+1] - image_cv2_band7[i][j]),
              np.fabs(image_cv2_band7[i + 3][j] - image_cv2_band7[i][j]),
              np.fabs(image_cv2_band7[i + 3][j-1] - image_cv2_band7[i][j]),
              np.fabs(image_cv2_band7[i + 2][j-2] - image_cv2_band7[i][j]),
              np.fabs(image_cv2_band7[i + 1][j-3] - image_cv2_band7[i][j]),
              np.fabs(image_cv2_band7[i ][j-3] - image_cv2_band7[i][j]),
              np.fabs(image_cv2_band7[i -1][j-3] - image_cv2_band7[i][j]),
              np.fabs(image_cv2_band7[i -2][j-2] - image_cv2_band7[i][j]),
              np.fabs(image_cv2_band7[i -3][j-1] - image_cv2_band7[i][j])
              ]
        temp=np.array(temp)

        image1.append(np.mean(temp))



image2=[]

for i in range(3,image_cv2_band1.shape[0]-3):
    for j in range(2,image_cv2_band7.shape[1]-3):

        temp=[np.fabs(image_cv2_band1[i-3][j]-image_cv2_band1[i][j]),
              np.fabs(image_cv2_band1[i-3][j+1]-image_cv2_band1[i][j]),
              np.fabs(image_cv2_band1[i-2][j+2]-image_cv2_band1[i][j]),
              np.fabs(image_cv2_band1[i-1][j+3]-image_cv2_band1[i][j]),
              np.fabs(image_cv2_band1[i][j+3] - image_cv2_band1[i][j]),
              np.fabs(image_cv2_band1[i +1][j+3] - image_cv2_band1[i][j]),
              np.fabs(image_cv2_band1[i + 2][j+2] - image_cv2_band1[i][j]),
              np.fabs(image_cv2_band1[i + 3][j+1] - image_cv2_band1[i][j]),
              np.fabs(image_cv2_band1[i + 3][j] - image_cv2_band1[i][j]),
              np.fabs(image_cv2_band1[i + 3][j-1] - image_cv2_band1[i][j]),
              np.fabs(image_cv2_band1[i + 2][j-2] - image_cv2_band1[i][j]),
              np.fabs(image_cv2_band1[i + 1][j-3] - image_cv2_band1[i][j]),
              np.fabs(image_cv2_band1[i ][j-3] - image_cv2_band1[i][j]),
              np.fabs(image_cv2_band1[i -1][j-3] - image_cv2_band1[i][j]),
              np.fabs(image_cv2_band1[i -2][j-2] - image_cv2_band1[i][j]),
              np.fabs(image_cv2_band1[i -3][j-1] - image_cv2_band1[i][j])]

        temp=np.array(temp)

        image2.append(np.mean(temp))





image1=np.array(image1)
image2=np.array(image2)


reduce=image1-image2

print(reduce)
squr=np.mean(reduce**2)
print(squr)

print('直接减法',np.sum(image_cv2_band7-image_cv2_band1))
print('FAST点均方误差',np.sum(squr)/10000)




































