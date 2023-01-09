import cv2
import numpy as np
import matplotlib.pyplot as plt
image_cv2_band7=cv2.imread("D:\\ProgramData_second\\Similarityfunction\\data\\origin_image\\xi_an_sub10000band1.tif",cv2.IMREAD_UNCHANGED)
image_cv2_band1=cv2.imread("D:\\ProgramData_second\\Similarityfunction\\data\\origin_image\\xi_an_sub10000band3.tif",cv2.IMREAD_UNCHANGED)

image_cv2_band7=image_cv2_band7[0:100,0:100]
# image_cv2_band1=image_cv2_band1[100:200,100:200]/255
image_cv2_band1=image_cv2_band1[0:100,0:100]

















