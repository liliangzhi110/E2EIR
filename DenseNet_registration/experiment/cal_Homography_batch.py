import tensorflow as tf
import numpy as np
import cv2
np.set_printoptions(suppress=True)




def getPerspectiveTransformMatrix(input):

    batch=tf.shape(input)[0]

    point1 = tf.convert_to_tensor([[0, 0], [256, 0], [256, 256], [0, 256]],dtype=tf.float32)
    point1 = tf.reshape(point1, (1, 8))
    point1 = tf.tile(point1, [batch, 1])

    point2 = tf.subtract(point1,input)
    print(batch.numpy())

    batch_A=[]
    for i in range(0, batch):
        print(i)

        x1, x2, x3, x4 = point1[i:(i+1), 0].numpy()[0], point1[i:(i+1), 2].numpy()[0], point1[i:(i+1), 4].numpy()[0], point1[i:(i+1), 6].numpy()[0]
        y1, y2, y3, y4 = point1[i:(i+1), 1].numpy()[0], point1[i:(i+1), 3].numpy()[0], point1[i:(i+1), 5].numpy()[0], point1[i:(i+1), 7].numpy()[0]

        u1, u2, u3, u4 = point2[i:(i+1), 0].numpy()[0], point2[i:(i+1), 2].numpy()[0], point2[i:(i+1), 4].numpy()[0], point2[i:(i+1), 6].numpy()[0]
        v1, v2, v3, v4 = point2[i:(i+1), 1].numpy()[0], point2[i:(i+1), 3].numpy()[0], point2[i:(i+1), 5].numpy()[0], point2[i:(i+1), 7].numpy()[0]

        A = [[x1, y1, 1, 0, 0, 0, -u1 * x1, -u1 * y1, -u1],
             [0, 0, 0, x1, y1, 1, -v1 * x1, -v1 * y1, -v1],
             [x2, y2, 1, 0, 0, 0, -u2 * x2, -u2 * y2, -u2],
             [0, 0, 0, x2, y2, 1, -v2 * x2, -v2 * y2, -v2],
             [x3, y3, 1, 0, 0, 0, -u3 * x3, -u3 * y3, -u3],
             [0, 0, 0, x3, y3, 1, -v3 * x3, -v3 * y3, -v3],
             [x4, y4, 1, 0, 0, 0, -u4 * x4, -u4 * y4, -u4],
             [0, 0, 0, x4, y4, 1, -v4 * x4, -v4 * y4, -v4]]

        batch_A.append(A)

    batch_A=tf.stack(batch_A)

    # arrayA = tf.convert_to_tensor(batch_A)
    arrayA=tf.reshape(batch_A,(batch,8,9))

    U,S,Vh=tf.linalg.svd(arrayA,full_matrices=True)

    Vh=tf.transpose(Vh)

    L = Vh[-1, :] / Vh[-1, -1]

    L=tf.transpose(L)

    return L


i=tf.convert_to_tensor([[ 10. , -9.],
 [ -1.,  -5.],
 [-10.  ,10.],
 [ -8. ,-10.]])

i=tf.reshape(i,(1,8))

i=tf.tile(i,[3,1])

H=getPerspectiveTransformMatrix(i)



p1=np.float32([[0, 0], [256, 0], [256, 256], [0, 256]])

displace_4_point=np.array([[ 10. , -9.],[ -1.,  -5.],[-10.  ,10.],[ -8. ,-10.]])


p2=p1-displace_4_point
p2=np.float32(p2)

m=cv2.getPerspectiveTransform(p1,p2)



print(H)

print(m)





