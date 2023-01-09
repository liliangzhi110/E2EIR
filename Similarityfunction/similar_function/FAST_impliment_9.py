import cv2
import numpy as np
import matplotlib.pyplot as plt

data=np.load('C:\\Users\\kylenate\\Desktop\\paper_registration1114\\land_sat.npz')['data_image']

image_cv2_band7=data[0:1,0:100,0:100].reshape((100,100))
#image_cv2_band1=data[5:6,0:100,0:100].reshape((100,100))

image_cv2_band1=data[3:4,0:100,0:100].reshape((100,100))

plt.subplot(1,3,1)
plt.imshow(image_cv2_band7)

plt.subplot(1,3,2)
plt.imshow(image_cv2_band1)

plt.subplot(1,3,3)
plt.plot(np.arange(100).reshape(100,),image_cv2_band7[50:51,:].reshape(100,))
plt.plot(np.arange(100).reshape(100,),image_cv2_band1[50:51,:].reshape(100,))


plt.show()



origin=130
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

        # image1.append(np.mean(temp))
        t=[]
        for k in temp:
            if k >origin:
                t.append(1)
            else:t.append(0)
        image1.append(np.sum(t))


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

        t=[]
        # image2.append(np.mean(temp))
        for k in temp:
            if k >origin:
                t.append(1)
            else:t.append(0)
        image2.append(np.sum(t))




image1=np.array(image1)
image2=np.array(image2)

print(image1)
print(image2)


reduce=image2-image1
squr=np.mean(reduce**2)

print('直接减法',np.sum(image_cv2_band7-image_cv2_band1))
print('FAST 打分的均方误差',np.sum(squr))






































