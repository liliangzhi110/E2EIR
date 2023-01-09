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
img1 = data[2:3,100:300,100:300].reshape((200,200))

plt.subplot(1,2,1)
plt.imshow(img)

plt.subplot(1,2,2)
plt.imshow(img1)
plt.show()


# Initiate FAST detector
star = cv2.xfeatures2d.StarDetector_create()

# Initiate BRIEF extractor
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# find the keypoints with STAR
kp = star.detect(img,None)
print(kp)
# compute the descriptors with BRIEF

kp=cv2.KeyPoint(28.0,28.0,1)

kp=[kp]

kp, des = brief.compute(img, kp)
kp1, des1 = brief.compute(img1, kp)
# print( brief.descriptorSize() )

print(kp)
# print(kp[0].pt,kp[0].size,kp[0].angle,kp[0].response,kp[0].octave,kp[0].class_id)
print(des)

print(kp1)

print(des1)

out=des1-des
print('')
print(out)

# img1 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
# cv2.imshow('BRIEF',img1)
# cv2.waitKey()

descripor=[]
for i in range(28,img.shape[0]-28):
    for j in range(28,img.shape[1]-28):
        kp = cv2.KeyPoint(i, j, 1)

        kp = [kp]

        kp, des = brief.compute(img, kp)
        descripor.append(des)


descripor1=[]
for i in range(28,img1.shape[0]-28):
    for j in range(28,img1.shape[1]-28):
        kp = cv2.KeyPoint(i, j, 1)

        kp = [kp]

        kp, des = brief.compute(img1, kp)
        descripor1.append(des)


descripor=np.array(descripor)
descripor1=np.array(descripor1)

print(descripor)
print(descripor1)

reduce=descripor-descripor1
squr=np.mean((reduce**2))


print('FAST 打分的均方误差',squr)








