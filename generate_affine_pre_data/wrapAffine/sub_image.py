import cv2
import numpy as np
import matplotlib.pyplot as plt




image_cv2_band7=cv2.imread("D:\\Second_paper_VAE_CNN\\xian_image\\xi_an_sub10000band1.tif",cv2.IMREAD_UNCHANGED)
image_cv2_band1=cv2.imread("D:\\Second_paper_VAE_CNN\\xian_image\\xi_an_sub10000band3.tif",cv2.IMREAD_UNCHANGED)

image_cv2_band7=np.array(image_cv2_band7)
#变换矩阵
print(image_cv2_band7.shape)
points=[]
for x in range(64,image_cv2_band7.shape[0]-64,128):
    for y in range(64,image_cv2_band7.shape[1]-64,128):
        x_y=np.array([x,y,1]).reshape(3,1)
        sub_fixed=image_cv2_band7[x-64:x+64,y-64:y+64]
        sub_moved = image_cv2_band1[x - 64:x + 64, y - 64:y + 64]

        cv2.imwrite("D:\\Second_paper_VAE_CNN\\fixed_image\\"+"%03d"%x+"_"+"%03d"%y+".jpg",sub_fixed)
        cv2.imwrite("D:\\Second_paper_VAE_CNN\\moved_image\\" + "%03d"%x + "_" + "%03d"%y + ".jpg", sub_moved)

        points1 = np.float32([[0, 0], [128, 0], [128, 128], [0, 128]])
        points2 = np.float32([[0,np.random.randint(0,50,size=(1,))[0]],
                              [np.random.randint(80,128,size=(1,))[0], 0],
                              [128, np.random.randint(85,128,size=(1,))[0]],
                              [np.random.randint(0,40,size=(1,))[0], 128]])
        matrix = cv2.getPerspectiveTransform(points1, points2)
        sub_moving = cv2.warpPerspective(sub_moved, matrix, (128, 128))

        cv2.imwrite("D:\\Second_paper_VAE_CNN\\moving_image\\" + "%03d"%x + "_" + "%03d"%y + ".jpg", sub_moving)

        print("next")































