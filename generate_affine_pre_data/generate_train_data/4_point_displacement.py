import cv2
import numpy as np
image_cv2_band7=cv2.imread("D:\\Second_paper_VAE_CNN\\xian_image\\xi_an_sub10000band1.tif",cv2.IMREAD_UNCHANGED)
image_cv2_band1=cv2.imread("D:\\Second_paper_VAE_CNN\\xian_image\\xi_an_sub10000band3.tif",cv2.IMREAD_UNCHANGED)


fixed=[]
moving=[]
displacement_4_point=[]
for x in range(64,image_cv2_band7.shape[0]-64,128):
    for y in range(64,image_cv2_band7.shape[1]-64,128):
        x_y=np.array([x,y,1]).reshape(3,1)
        sub_fixed=image_cv2_band7[x-64:x+64,y-64:y+64].reshape((128,128,1))
        sub_moved = image_cv2_band1[x - 64:x + 64, y - 64:y + 64]

        fixed.append(sub_fixed)

        points1 = np.float32([[0, 0], [128, 0], [128, 128], [0, 128]])
        points2 = np.float32([[np.random.randint(0,15,size=(1,))[0],np.random.randint(0,15,size=(1,))[0]],
                              [np.random.randint(113,128,size=(1,))[0], np.random.randint(0,15,size=(1,))[0]],
                              [np.random.randint(113,128,size=(1,))[0], np.random.randint(113,128,size=(1,))[0]],
                              [np.random.randint(0,15,size=(1,))[0], np.random.randint(113,128,size=(1,))[0]]])

        matrix = cv2.getPerspectiveTransform(points1, points2)

        temp=points2-points1
        displacement_4_point.append(temp.reshape(1,8))

        sub_moving = cv2.warpPerspective(sub_moved, matrix, (128, 128))
        moving.append(sub_moving.reshape((128,128,1)))


fixed=np.array(fixed)
moving=np.array(moving)
displacement_4_point=np.array(displacement_4_point)

np.savez("D:\\ProgramData_second\\generate_affine_pre_data\\data\\generated_npz_image\\displacement_4_point.npz",
         fixed=fixed,
         moving=moving,
         displacement_4_point=displacement_4_point
         )
