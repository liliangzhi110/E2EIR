import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from STN.TPS_STN import TPS_STN

img = cv2.imread("D:\\ProgramData_second\\generate_affine_pre_data\\data\\generated_npz_image\\fixed.jpg",cv2.IMREAD_UNCHANGED)
out_size = list(img.shape)
shape = [1]+out_size+[1]



nx = 4
ny = 4
# x,y=np.ones(shape=(4,)),np.ones(shape=(4,))

x,y=np.random.uniform(-1.0,1.0,4),np.random.uniform(-1.0,1.0,4)
# x, y = np.linspace(0, 1, 3), np.linspace(0, 1, 3)
x, y = np.meshgrid(x, y)
xs = x.flatten()
ys = y.flatten()
cps = np.vstack([xs, ys]).T
print(cps)



v = np.array([
               [-1., - 1.],
               [0. ,- 1.],
               [1. ,- 1.],
               [-1.,  0.],
             [0.,0.],
[1. , 0.],
[-1.,1.],
[0. , 1.],
[1.,1.],
])

p = tf.constant(cps.reshape([1, nx*ny, 2]), dtype=tf.float32)
t_img = tf.constant(img.reshape(shape), dtype=tf.float32)
t_img = TPS_STN(t_img, nx, ny, p, out_size)



plt.subplot(1,2,1)
plt.imshow(img)

plt.subplot(1,2,2)
plt.imshow(t_img.numpy().reshape((128,128)))
plt.show()















