import tensorflow as tf
import numpy as np
from Registration_model.densenet_model import Registration_model
from Registration_model.densenet_model import getPerspectiveTransformMatrix
from Registration_model.densenet_model import spatial_transformer_network
from Registration_model.loss import matix_loss_mes,image_loss_mes


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
        self.matix = getPerspectiveTransformMatrix(name='spatial_transformer')
        self.image = spatial_transformer_network()

    def call(self,Inputtensor):

        input1=Inputtensor[:,:,:,0:1]
        input2=Inputtensor[:,:,:,1:2]

        ou1=self.net1(input1)
        ou2=self.net2(input2)
        output = tf.keras.layers.concatenate([ou1, ou2])
        output=self.fully_con1(output)
        output=self.fully_con2(output)
        output = self.fully_con3(output)
        output = self.fully_con4(output)
        output1=self.out_Affine(output)

        output2 = self.matix(output1)
        output3 = self.image([input2,output2])

        return [output1,output3]

import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

matrixt_file= 'E:\\ProgramData_second\\generate_affine_pre_data\\data\\generated_npz_image\\displacement_4_point_paper_random_landsat_256.npz'
fixed_image=np.load(matrixt_file)['fixed'][0:300,:,:,:]
moving_image=np.load(matrixt_file)['moving'][0:300,:,:,:]
per_matrix=np.load(matrixt_file)['displacement_4_point'][0:300,:].reshape((300,8))


fixed_image=fixed_image.astype('float32')
moving_image=moving_image.astype('float32')
matrix=per_matrix.astype('float32')


con_fixed_moving_image=np.concatenate((fixed_image,moving_image),axis=3)
train_ds = tf.data.Dataset.from_tensor_slices((con_fixed_moving_image,(matrix,fixed_image))).shuffle(100).batch(4)
train_ds=train_ds.prefetch(tf.data.experimental.AUTOTUNE)



losses=[matix_loss_mes(),image_loss_mes()]

model=regis_model()

model.compile(loss=losses,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss_weights=[1., 0.2]
              )

model.fit(train_ds,epochs=50,verbose = 1)

# model.fit_generator(vxm_data_generator(fixed_image,moving_image,per_matrix), epochs=1, steps_per_epoch=4, verbose = 1)
model.save_weights('E:/ProgramData_second/DenseNet_registration/save_model/model',save_format='tf')















