import cv2
import pandas as pd
import numpy as np
import os


image_cv2_band7=cv2.imread("D:\\Second_paper_VAE_CNN\\xian_image\\xi_an_sub10000band1.tif",cv2.IMREAD_UNCHANGED)
image_cv2_band1=cv2.imread("D:\\Second_paper_VAE_CNN\\xian_image\\xi_an_sub10000band3.tif",cv2.IMREAD_UNCHANGED)

if not os.path.exists("D:\\ProgramData\\DenseNet_registration\\Densenet\\densenet_train_fixed"): os.makedirs(
    "D:\\ProgramData\\DenseNet_registration\\Densenet\\densenet_train_fixed")

if not os.path.exists("D:\\ProgramData\DenseNet_registration\\Densenet\\densenet_train_moving"): os.makedirs(
    "D:\\ProgramData\\DenseNet_registration\\Densenet\\densenet_train_moving")

if not os.path.exists("D:\\ProgramData\\DenseNet_registration\\Densenet\\densenet_Affine_matrix"): os.makedirs(
    "D:\\ProgramData\\DenseNet_registration\\Densenet\\densenet_Affine_matrix")


path_fixed=[]
path_moving=[]
Perpecttive=[]
for x in range(64,image_cv2_band7.shape[0]-64,128):
    for y in range(64,image_cv2_band7.shape[1]-64,128):
        x_y=np.array([x,y,1]).reshape(3,1)
        sub_fixed=image_cv2_band7[x-64:x+64,y-64:y+64]
        sub_moved = image_cv2_band1[x - 64:x + 64, y - 64:y + 64]

        path_fixed.append(sub_fixed)


        points1 = np.float32([[0, 0], [128, 0], [128, 128], [0, 128]])
        points2 = np.float32([[0,np.random.randint(0,50,size=(1,))[0]],
                              [np.random.randint(80,128,size=(1,))[0], 0],
                              [128, np.random.randint(85,128,size=(1,))[0]],
                              [np.random.randint(0,40,size=(1,))[0], 128]])
        matrix = cv2.getPerspectiveTransform(points1, points2)


        matrix2=cv2.getPerspectiveTransform(points2,points1)

        Perpecttive.append(matrix2)
        sub_moving = cv2.warpPerspective(sub_moved, matrix, (128, 128))
        path_moving.append(sub_moving)


path_fixed=np.array(path_fixed)
path_moving=np.array(path_moving)
Perpecttive=np.array(Perpecttive)

np.savez("D:\\ProgramData\\DenseNet_registration\\Densenet\\densenet_Affine_matrix\\Perspective_matrix.npz",
         path_fixed=path_fixed,
         path_moving=path_moving,
         Perpecttive=Perpecttive
         )

# cvssave=pd.DataFrame({"path_fixed":path_fixed,"path_moving":path_moving,"Perpecttive":Perpecttive})
#
# cvssave.to_csv("D:\\ProgramData\\DenseNet_registration\\Densenet\\densenet_Affine_matrix\\matrix.csv")
#
#
# read=pd.read_csv("D:\\ProgramData\\DenseNet_registration\\Densenet\\densenet_Affine_matrix\\matrix.csv")
#
# fixed=read["path_fixed"]
#
#
# print(fixed)













