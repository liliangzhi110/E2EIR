import cv2
import numpy as np

one=np.load("D:\\Second_paper_VAE_CNN\\xi_an_landsat8\\one\\landsat1_1.npz")['image'][0]
second=np.load("D:\\Second_paper_VAE_CNN\\xi_an_landsat8\\second\\landsat2_2.npz")['image'][0]

one=(one-np.min(one))/(np.max(one)-np.min(one))
second=(second-np.min(second))/(np.max(second)-np.min(second))

fixed=[]
moving=[]
displacement_4_point=[]

for x in range(128,one.shape[0]-128,32):
    for y in range(128,one.shape[1]-128,64):


        sub_fixed =    one[x - 128:x + 128, y - 128:y + 128]
        sub_moved = second[x - 128:x + 128, y - 128:y + 128]

        points1 = np.float32([[0, 0], [256, 0], [256, 256], [0, 256]])
        points2 = np.float32([[np.random.randint(0, 30, size=(1,))[0], np.random.randint(0, 30, size=(1,))[0]],
                              [np.random.randint(226, 256, size=(1,))[0], np.random.randint(0, 30, size=(1,))[0]],
                              [np.random.randint(226, 256, size=(1,))[0], np.random.randint(226, 256, size=(1,))[0]],
                              [np.random.randint(0, 30, size=(1,))[0], np.random.randint(226, 256, size=(1,))[0]]])

        matrix1 = cv2.getPerspectiveTransform(points1, points2)

        sub_fixed = cv2.warpPerspective(sub_fixed, matrix1, (256, 256))

        fixed.append(sub_fixed.reshape((256, 256, 1)))


        points3 = np.float32([[0, 0], [256, 0], [256, 256], [0, 256]])
        points4 = np.float32([[np.random.randint(0,30,size=(1,))[0],np.random.randint(0,30,size=(1,))[0]],
                              [np.random.randint(226,256,size=(1,))[0], np.random.randint(0,30,size=(1,))[0]],
                              [np.random.randint(226,256,size=(1,))[0], np.random.randint(226,256,size=(1,))[0]],
                              [np.random.randint(0,30,size=(1,))[0], np.random.randint(226,256,size=(1,))[0]]])

        matrix2 = cv2.getPerspectiveTransform(points3, points4)
        sub_moving = cv2.warpPerspective(sub_moved, matrix2, (256, 256))
        moving.append(sub_moving.reshape((256, 256, 1)))



        temp=points4-points2

        displacement_4_point.append(temp.reshape(1,8))



fixed=np.array(fixed)
moving=np.array(moving)
displacement_4_point=np.array(displacement_4_point)

np.savez("D:\\ProgramData_second\\generate_affine_pre_data\\data\\generated_npz_image\\displacement_4_point_random_random_landsat_256.npz",
         fixed=fixed,
         moving=moving,
         displacement_4_point=displacement_4_point)