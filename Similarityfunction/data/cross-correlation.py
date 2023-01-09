from sklearn import metrics
import cv2
import numpy as np
import matplotlib.pyplot as plt
data=np.load('C:\\Users\\kylenate\\Desktop\\paper_registration1114\\land_sat.npz')['data_image']

image_cv2_band7=data[0:1,0:100,0:100].reshape((100,100))
image_cv2_band1=data[3:4,100:200,0:100].reshape((100,100))
# image_cv2_band1=data[5:6,100:200,100:200].reshape((100,100))


plt.subplot(1,2,1)
plt.imshow(image_cv2_band7)

plt.subplot(1,2,2)
plt.imshow(image_cv2_band1)
plt.show()
value=np.correlate(image_cv2_band1.reshape((10000,)),image_cv2_band7.reshape((10000,)))

print(value)