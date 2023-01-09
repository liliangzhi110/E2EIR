import cv2
from skimage.external.tifffile import TiffFile
import matplotlib.pyplot as plt
import numpy as np

image_cv2=cv2.imread("C:\\Users\\kylenate\\Desktop\\paper_registration1114\\santa_cruz_az-band7.tif",cv2.IMREAD_UNCHANGED)



# pts1 = np.float32([[50,50],[200,50],[50,200]])
# pts2 = np.float32([[10,100],[200,50],[100,250]])
pts1 = np.float32([[0,0],[1000,0],[1000,1000]])
pts2 = np.float32([[0,200],[870,0],[1000,750]])

M = cv2.getAffineTransform(pts1,pts2)
print(M)

wrap = cv2.warpAffine(image_cv2,M,(1000,1000))
plt.subplot(1,2,1)
plt.imshow(wrap)


arr=np.array([0,0,1]).reshape((1,3))
con=np.concatenate((M,arr),axis=0)
x_point=np.array([1000,0,1]).reshape((3,1))


tra=np.matmul(con,x_point)


print(con)

print(tra)


# points1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
# points2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

points1 = np.float32([[0,0],[1000,0],[1000,1000],[0,1000]])
points2 = np.float32([[1000,1000],[0,1000],[0,0],[1000,0]])
matrix = cv2.getPerspectiveTransform(points1,points2)
output = cv2.warpPerspective(image_cv2, matrix, (1000, 1000))

print(matrix)


plt.subplot(1,2,2)
plt.imshow(output)
plt.show()




































































































































































