import cv2
from skimage.external.tifffile import TiffFile
import matplotlib.pyplot as plt
import numpy as np

image_cv2=cv2.imread("C:\\Users\\kylenate\\Desktop\\paper_registration1114\\santa_cruz_az-band7.tif",cv2.IMREAD_UNCHANGED)
plt.subplot(1,3,1)
plt.imshow(image_cv2)

image_cv2= np.pad(image_cv2, ((600, 300), (800, 600)), 'constant', constant_values=(0, 0))

plt.subplot(1,3,2)
plt.imshow(image_cv2)



pts1 = np.float32([[0,0],[1000,0],[1000,1000]])
pts2 = np.float32([[0,200],[870,0],[1000,750]])

M = cv2.getAffineTransform(pts1,pts2)

wrap = cv2.warpAffine(image_cv2,M,(1000,1000))

plt.subplot(1,3,3)
plt.imshow(wrap)

plt.show()




