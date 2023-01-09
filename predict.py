import tensorflow as tf
import numpy as np
from Registration_model.densenet_model import Registration_model
from Registration_model.data_generator import vxm_data_generator
np.set_printoptions(suppress=True)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



class regis_model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net1=Registration_model()
        self.net2=Registration_model()

        self.fully_con1 = tf.keras.layers.Dense(units=1024,activation=tf.nn.relu,dtype='float32')
        self.fully_con2 = tf.keras.layers.Dense(units=512,activation=tf.nn.relu,dtype='float32')
        self.fully_con3 = tf.keras.layers.Dense(units=128,activation=tf.nn.relu,dtype='float32')
        self.fully_con4 = tf.keras.layers.Dense(units=64,activation=tf.nn.relu,dtype='float32')
        self.out_Affine = tf.keras.layers.Dense(units=8,dtype='float32')


    def call(self,Inputtensor):
        input1=Inputtensor[0]
        input2=Inputtensor[1]
        ou1=self.net1(input1)
        ou2=self.net2(input2)
        output = tf.keras.layers.concatenate([ou1, ou2])

        output=self.fully_con1(output)
        output=self.fully_con2(output)
        output = self.fully_con3(output)
        output = self.fully_con4(output)
        output=self.out_Affine(output)


        return output

matrixt_file= 'D:\\ProgramData_second\\generate_affine_pre_data\\data\\generated_npz_image\\pre_displacement_4_point.npz'

fixed_image=np.load(matrixt_file)['fixed'].reshape((1, 128, 128,1))
moving_image=np.load(matrixt_file)['moving'].reshape((1, 128, 128,1))
per_matrix=np.load(matrixt_file)['displacement_4_point'].reshape((fixed_image.shape[0],8))


fixed_image=fixed_image.astype('float32')/255
moving_image=moving_image.astype('float32')/255
per_matrix=per_matrix.astype('float32')

data=vxm_data_generator(fixed_image,moving_image,per_matrix,batch_size=1)
con,m=next(data)
print(m)
model=regis_model()
model.compile(loss='MSE',
              optimizer=tf.keras.optimizers.Adam(0.0001)
              )

# Load the state of the old model
model.load_weights('D:/ProgramData_second/DenseNet_registration/save_model_3/model')

# Check that the model state has been preserved
new_predictions = model.predict(con)
temp=m-new_predictions

print(new_predictions)
print('')
print(np.reshape(temp,(8,)))






































# with tf.device("/cpu:0"):
#     model = regis_model()
#
#     model.load_weights('D:/ProgramData/DenseNet_registration/Densenet/model.h5',by_name=True)






























