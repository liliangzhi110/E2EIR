import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from STN.Proj_tr_matrix import getPerspectiveTransformMatrix,Matrix
from STN.STN_proj import spatial_transformer_network
np.set_printoptions(suppress=True)

if __name__=='__main__':

    batch=3

    i=tf.convert_to_tensor([[ 10. , -15.],[ -100.,  -50.],[-10.  ,10.],[ -8. ,-10.]])
    i=tf.reshape(i,(1,8))
    i=tf.tile(i,[batch,1])


    H=Matrix(i)

    img = cv2.imread('C:\\Users\\kylenate\\Desktop\\panda1.jpg')
    img = cv2.cvtColor(img,cv2.IMREAD_COLOR)/255

    height = img.shape[0]
    width = img.shape[1]

    imgs = tf.convert_to_tensor(img, dtype='float32')
    imgs=tf.reshape(imgs,(1,500,500,3))

    imgs=tf.tile(imgs,[batch,1,1,1])


    out_image=spatial_transformer_network(imgs,H)

    out_image=out_image.numpy()[0]


    indix=np.where(out_image[:,:,0]<0.001)


    x=indix[0]
    y=indix[1]

    for i in range(len(x)):

        img[x[i],y[i]]=0.5


    plt.subplot(121)
    plt.imshow(img)


    plt.subplot(122)
    plt.imshow(out_image)
    plt.show()















