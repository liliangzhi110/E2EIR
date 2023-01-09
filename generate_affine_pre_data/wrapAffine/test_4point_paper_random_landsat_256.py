import cv2
from skimage.external.tifffile import TiffFile
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
image_cv2=cv2.imread("C:\\Users\\kylenate\\Desktop\\paper_registration1114\\santa_cruz_az-band7.tif",cv2.IMREAD_UNCHANGED)

image_cv2=image_cv2[0:512,0:512]


sub_fixed_256 = image_cv2[128:384, 128:384]

points1 = np.float32([[128, 128], [384, 128], [384, 384], [128, 384]])
points2 = np.float32([[np.random.randint(112,144,size=(1,))[0],np.random.randint(112,144,size=(1,))[0]],
                    [np.random.randint(365,396,size=(1,))[0], np.random.randint(112,146,size=(1,))[0]],
                 [np.random.randint(362,396,size=(1,))[0], np.random.randint(362,396,size=(1,))[0]],
                [np.random.randint(112,144,size=(1,))[0], np.random.randint(362,396,size=(1,))[0]]])

print(points2.reshape((2,4)))
print(points1.reshape((2,4)))

matrix=cv2.getPerspectiveTransform(points2,points1)
bb=cv2.findHomography(points2,points1)

print(points1-points2)


print(matrix)
print(bb)

moving_image=cv2.warpPerspective(image_cv2,matrix,(512,512))

moving_image_256=moving_image[128:384, 128:384]


wrap=cv2.getPerspectiveTransform(points1,points2)

moving_image_256_return=cv2.warpPerspective(moving_image_256,wrap,(256,256))




plt.subplot(2,2,1)
plt.title('256_fixed')
plt.imshow(sub_fixed_256)



plt.subplot(2,2,2)
plt.title('256_moving')
plt.imshow(moving_image_256)

plt.subplot(2,2,3)
plt.imshow(moving_image_256_return)


wrap=cv2.getPerspectiveTransform(points1-128,points2-128)

moving_image_256_return=cv2.warpPerspective(moving_image_256,wrap,(256,256))

plt.subplot(2,2,4)
plt.imshow(moving_image_256_return)


plt.show()



