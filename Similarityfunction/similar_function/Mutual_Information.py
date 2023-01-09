import tensorflow as tf
from sklearn import metrics
import cv2
import numpy as np
import matplotlib.pyplot as plt
data=np.load('C:\\Users\\kylenate\\Desktop\\paper_registration1114\\land_sat.npz')

data=np.load('C:\\Users\\kylenate\\Desktop\\paper_registration1114\\land_sat.npz')['data_image']

image_cv2_band7=data[0:1,0:100,0:100].reshape((100,100))
image_cv2_band1=data[3:4,100:200,0:100].reshape((100,100))
# image_cv2_band1=data[5:6,100:200,100:200].reshape((100,100))

# image_cv2_band1=image_cv2_band1[100:200,100:200]
#image_cv2_band1=image_cv2_band1[100:200,100:200]/255



plt.subplot(1,2,1)
plt.imshow(image_cv2_band7)

plt.subplot(1,2,2)
plt.imshow(image_cv2_band1)
plt.show()
value=metrics.adjusted_mutual_info_score(image_cv2_band1.reshape((10000,)),image_cv2_band7.reshape((10000,)))

print(value)


# p_y = tf.reduce_sum(p_y_on_x, axis=0, keepdim=True) / num_x  # 1-by-num_y
# h_y = -tf.reduce_sum(p_y * tf.math.log(p_y))
# p_c = tf.reduce_sum(p_c_on_x, axis=0) / num_x  # 1-by-num_c
# h_c = -tf.reduce_sum(p_c * tf.math.log(p_c))
# p_x_on_y = p_y_on_x / num_x / p_y  # num_x-by-num_y
# p_c_on_y = tf.matmul(p_c_on_x, p_x_on_y, transpose_a=True)  # num_c-by-num_y
# h_c_on_y = -tf.reduce_sum(tf.reduce_sum(p_c_on_y * tf.math.log(p_c_on_y), axis=0) * p_y)
# i_y_c = h_c - h_c_on_y
# nmi = 2 * i_y_c / (h_y + h_c)