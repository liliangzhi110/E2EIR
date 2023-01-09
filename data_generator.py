import numpy as np



def vxm_data_generator(fixed_image,moving_image, per_matrix,batch_size=32):


    while True:
        # prepare inputs
        # inputs need to be of the size [batch_size, H, W, number_features]
        #   number_features at input is 1 for us
        idx1 = np.random.randint(0, fixed_image.shape[0], size=batch_size)
        fixed_images = fixed_image[idx1, ..., ]

        # idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = moving_image[idx1, ..., ]
        tr_matrix=per_matrix[idx1, ...,]


        inputs1=fixed_images
        inputs2=moving_images
        outputs = tr_matrix

        yield [inputs1,inputs2], outputs





# matrixt_file= 'D:/ProgramData/DenseNet_registration/Densenet/matrix.npz'
# fixed_image=np.load(matrixt_file)['path_fixed']
# moving_image=np.load(matrixt_file)['path_moving']
# per_matrix=np.load(matrixt_file)['Perpecttive'].reshape(fixed_image.shape[0],9)
#
# fixed_image=fixed_image.astype('float')/255
# moving_image=moving_image.astype('float')/255
#
# data=vxm_data_generator(fixed_image,moving_image,per_matrix)
#
# x,y,z=next(data)



























































































