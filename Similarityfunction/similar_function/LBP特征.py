import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
data=np.load('C:\\Users\\kylenate\\Desktop\\paper_registration1114\\land_sat.npz')['data_image']

image_cv2_band7=data[0:1,0:100,0:100].reshape((100,100))
# image_cv2_band1=data[5:6,0:100,0:100].reshape((100,100))
image_cv2_band1=data[3:4,100:200,0:100].reshape((100,100))
# image_cv2_band1=image_cv2_band7


plt.subplot(1,4,1)
plt.imshow(image_cv2_band7,cmap=cm.gray)

plt.subplot(1,4,2)
plt.imshow(image_cv2_band1,cmap=cm.gray)




def LBP(src):
    '''
    :param src:灰度图像
    :return:
    '''
    height = src.shape[0]
    width = src.shape[1]
    dst = src.copy()

    lbp_value = np.zeros((1,8), dtype=np.uint8)
    neighbours = np.zeros((1,8), dtype=np.uint8)
    for x in range(1, width-1):
        for y in range(1, height-1):
            neighbours[0, 0] = src[y - 1, x - 1]
            neighbours[0, 1] = src[y - 1, x]
            neighbours[0, 2] = src[y - 1, x + 1]
            neighbours[0, 3] = src[y, x - 1]
            neighbours[0, 4] = src[y, x + 1]
            neighbours[0, 5] = src[y + 1, x - 1]
            neighbours[0, 6] = src[y + 1, x]
            neighbours[0, 7] = src[y + 1, x + 1]

            center = src[y, x]

            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 0] * 128

            dst[y, x] = lbp

    return dst




img=LBP(image_cv2_band7)
img1=LBP(image_cv2_band1)

plt.subplot(1,4,3)
plt.imshow(img,cmap=cm.gray)

plt.subplot(1,4,4)
plt.imshow(img1,cmap=cm.gray)

plt.show()






image1=np.array(img)
image2=np.array(img1)



reduce=image1-image2

print(reduce)

squr=np.mean(reduce**2)
print(squr)

print('直接减法',np.sum(image_cv2_band7-image_cv2_band1))
print('LBP',np.sum(squr))