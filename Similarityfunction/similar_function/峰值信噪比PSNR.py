import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
data=np.load('C:\\Users\\kylenate\\Desktop\\paper_registration1114\\land_sat.npz')['data_image']

image_cv2_band7=data[0:1,0:100,0:100].reshape((100,100))
# image_cv2_band1=data[5:6,0:100,0:100].reshape((100,100))
image_cv2_band1=data[3:4,100:200,0:100].reshape((100,100))



plt.subplot(1,2,1)
plt.imshow(image_cv2_band7)

plt.subplot(1,2,2)
plt.imshow(image_cv2_band1)
plt.show()



def psnr(target, ref, scale):
    # target:目标图像  ref:参考图像  scale:尺寸大小
    # assume RGB image
    target_data = np.array(target)
    target_data = target_data[scale:-scale ,scale:-scale]

    ref_data = np.array(ref)
    ref_data = ref_data[scale:-scale ,scale:-scale]

    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt( np.mean(diff ** 2.) )
    return 20 *math.log10(1.0 /rmse)



out=psnr(image_cv2_band1,image_cv2_band7,1)



print(out)













