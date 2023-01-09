from sklearn import metrics
import math
import numpy as np
import matplotlib.pyplot as plt
data=np.load('C:\\Users\\kylenate\\Desktop\\paper_registration1114\\land_sat.npz')['data_image']

image_cv2_band7=data[0:1,0:100,0:100].reshape((100,100))
image_cv2_band1=data[1:2,0:100,0:100].reshape((100,100))


plt.subplot(1,2,1)
plt.imshow(image_cv2_band7)

plt.subplot(1,2,2)
plt.imshow(image_cv2_band1)
plt.show()
value=metrics.adjusted_rand_score(image_cv2_band7.reshape((10000,)),image_cv2_band1.reshape((10000,)))

print(value)