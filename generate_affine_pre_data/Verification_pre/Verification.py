import cv2
import matplotlib.pyplot as plt
import numpy as np
# np.set_printoptions(suppress=True)

matrixt_file= 'D:\\ProgramData_second\\generate_affine_pre_data\\data\\generated_npz_image\\pre_displacement_4_point.npz'

fixed_image=np.load(matrixt_file)['fixed'].reshape((128,128))
moving_image=np.load(matrixt_file)['moving'].reshape((128,128))
displacement_4_point=np.load(matrixt_file)['displacement_4_point'].reshape((4,2))


points1 = np.float32([[0, 0], [128, 0], [128, 128], [0, 128]])
displacement_4_point=points1+displacement_4_point




matrix_pre =np.array([[ 10.363248,  13.420181],
                      [-13.995538,    6.0418906],
                      [-7.536574,  -1.4410772],
                      [13.6674795,  -6.705563]])
matrix_pre=(matrix_pre+points1).astype('float32')


print(displacement_4_point)
print('')
print(matrix_pre)


matrix = cv2.getPerspectiveTransform(matrix_pre,points1)
output = cv2.warpPerspective(moving_image, matrix, (128, 128))


plt.subplot(1,3,1)
plt.title('fixed')
plt.imshow(fixed_image)

plt.subplot(1,3,2)
plt.title('moving')
plt.imshow(moving_image)

plt.subplot(1,3,3)
plt.title('moved')
plt.imshow(output)


plt.show()
