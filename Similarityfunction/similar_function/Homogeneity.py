from sklearn import metrics
from sklearn.cluster import KMeans
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

value=metrics.homogeneity_score(image_cv2_band1.reshape((10000,)),image_cv2_band7.reshape((10000,)))
value1=metrics.completeness_score(image_cv2_band1.reshape((10000,)),image_cv2_band7.reshape((10000,)))
value2=metrics.v_measure_score(image_cv2_band1.reshape((10000,)),image_cv2_band7.reshape((10000,)))
value3=metrics.fowlkes_mallows_score(image_cv2_band1.reshape((10000,)),image_cv2_band7.reshape((10000,)))
# value4=metrics.calinski_harabaz_score(image_cv2_band1.reshape((10000,)),image_cv2_band7.reshape((10000,)))

print(value,value1,value2,value3)
print(np.fabs((value*100)),np.fabs((value1*100)),np.fabs((value2*100)))