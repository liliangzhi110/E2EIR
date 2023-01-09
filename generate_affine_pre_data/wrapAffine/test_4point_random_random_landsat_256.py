import cv2
from skimage.external.tifffile import TiffFile
import matplotlib.pyplot as plt
import numpy as np

image_cv2=cv2.imread("C:\\Users\\kylenate\\Desktop\\paper_registration1114\\santa_cruz_az-band7.tif",cv2.IMREAD_UNCHANGED)



pts1 = np.float32([[0,0],[1000,0],[1000,1000]])
pts2 = np.float32([[0,np.random.randint(0,200,(1,))[0]],[np.random.randint(800,1000,(1,))[0],0],[1000,np.random.randint(800,1000,(1,))[0]]])
M1 = cv2.getAffineTransform(pts1,pts2)
wrap1 = cv2.warpAffine(image_cv2,M1,(1000,1000))


plt.subplot(2,2,1)
plt.imshow(wrap1)


M2=cv2.getAffineTransform(pts2,pts1)
wrap2 = cv2.warpAffine(wrap1,M2,(1000,1000))

plt.subplot(2,2,2)
plt.imshow(wrap2)


pts3 = np.float32([[0,0],[1000,0],[1000,1000]])
pts4 = np.float32([[0,np.random.randint(0,200,(1,))[0]],[np.random.randint(800,1000,(1,))[0],0],[1000,np.random.randint(800,1000,(1,))[0]]])
M3 = cv2.getAffineTransform(pts3,pts4)
wrap3 = cv2.warpAffine(wrap2,M3,(1000,1000))

plt.subplot(2,2,3)
plt.imshow(wrap3)


pts5 = pts2
pts6 = pts4
M4 = cv2.getAffineTransform(pts6,pts5)
wrap4 = cv2.warpAffine(wrap3,M4,(1000,1000))


plt.subplot(2,2,4)
plt.imshow(wrap4)
plt.show()




































































































































































