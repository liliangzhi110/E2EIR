import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
data=np.load('C:\\Users\\kylenate\\Desktop\\paper_registration1114\\land_sat.npz')['data_image']

# image_cv2_band7=data[0:1,0:100,0:100].reshape((100,100))
# # image_cv2_band1=data[5:6,0:100,0:100].reshape((100,100))
# image_cv2_band1=data[3:4,100:200,0:100].reshape((100,100))
#
#
#
# plt.subplot(1,2,1)
# plt.imshow(image_cv2_band7)
#
# plt.subplot(1,2,2)
# plt.imshow(image_cv2_band1)
# plt.show()


img = data[0:1,0:200,0:200].reshape((200,200))
img1 = data[2:3,0:200,0:200].reshape((200,200))

plt.subplot(1,2,1)
plt.imshow(img)

plt.subplot(1,2,2)
plt.imshow(img1)
plt.show()


#在这里设置参数
winSize = (128,128)
blockSize = (64,64)
blockStride = (8,8)
cellSize = (16,16)
nbins = 9

#定义对象hog，同时输入定义的参数，剩下的默认即可
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

winStride = (8,8)
padding = (8,8)
descripor = hog.compute(img, winStride, padding).reshape((-1,))

descripor1 = hog.compute(img1, winStride, padding).reshape((-1,))


descripor=np.array(descripor)
descripor1=np.array(descripor1)

print(descripor)
print(descripor1)

reduce=descripor-descripor1
squr=np.mean((reduce**2))


print('FAST 打分的均方误差',squr)