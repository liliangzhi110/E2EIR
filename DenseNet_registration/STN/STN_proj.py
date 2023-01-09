import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from STN.Proj_tr_matrix import Matrix

def spatial_transformer_network(input_fmap, theta, out_dims=None, **kwargs):

    # grab input dimensions
    B = tf.shape(input_fmap)[0]
    H = tf.shape(input_fmap)[1]
    W = tf.shape(input_fmap)[2]


    # reshape theta to (B, 2, 3)
    theta = tf.reshape(theta, [B, 3, 3])

    # generate grids of same size or upsample/downsample if specified
    if out_dims:
        out_H = out_dims[0]
        out_W = out_dims[1]
        batch_grids = affine_grid_generator(out_H, out_W, theta)
    else:
        batch_grids = affine_grid_generator(H, W, theta)

    x_s = batch_grids[:, 0, :, :]/batch_grids[:, 2, :, :]
    y_s = batch_grids[:, 1, :, :]/batch_grids[:, 2, :, :]


    # sample input with grid to get output
    out_fmap = bilinear_sampler(input_fmap, x_s, y_s)


    return out_fmap


def get_pixel_value(img, x, y):

    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def affine_grid_generator(height, width, theta):

    num_batch = tf.shape(theta)[0]

    # create normalized 2D grid
    x = tf.linspace(0.0, 500.0, width)
    y = tf.linspace(0.0, 500.0, height)
    x_t, y_t = tf.meshgrid(x, y)


    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])


    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)#(1, 3, 250000)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))#(2, 3, 250000)


    # cast to float32 (required for matmul)
    theta = tf.cast(theta, 'float32')#(2,6)

    sampling_grid = tf.cast(sampling_grid, 'float32')#(2, 3, 250000)


    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(theta, sampling_grid)#(2, 2, 250000)

    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, H, W, 2)
    batch_grids = tf.reshape(batch_grids, [num_batch, 3, height, width])


    return batch_grids


def bilinear_sampler(img, x, y):

    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    # x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    # y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))


    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out

if __name__=='__main__':
    img = cv2.imread('C:\\Users\\kylenate\\Desktop\\panda1.jpg')

    img = cv2.cvtColor(img, cv2.IMREAD_COLOR)/255

    height = img.shape[0]
    width = img.shape[1]

    imgs = np.concatenate((np.expand_dims(img, 0), np.expand_dims(img, 0)))
    imgs = tf.convert_to_tensor(imgs, dtype='float32')

    points1 = np.float32([[0, 0], [500, 0], [500, 500],[0,500]])
    points2 = np.float32([[0, 100], [400, 0], [500, 400],[50,500]])

    affine_matrix1 = cv2.getPerspectiveTransform(points2, points1)
    print(affine_matrix1)

    affine_matrix2 = cv2.getPerspectiveTransform(points1, points2)
    print(affine_matrix2)

    # thetas = [
    #     [[1.28034883e+00, - 1.60043603e-01 , 1.60043603e+01],
    #      [3.48330195e-01,  1.39332078e+00 ,- 1.39332078e+02],
    #     [1.40719453e-04 , 2.84907343e-04 , 1.00000000e+00]],
    #
    #     [[7.78947368e-01,  8.94736842e-02,  0.00000000e+00],
    #      [-2.00000000e-01 , 6.94736842e-01 , 1.00000000e+02],
    #     [-5.26315789e-05, - 2.10526316e-04,  1.00000000e+00]]
    # ]
    # thetas=np.reshape(thetas,(2,9))
    # thetas = tf.convert_to_tensor(thetas, dtype='float32')

    t=points1-points2
    t=t.reshape((1,8))
    t=tf.convert_to_tensor(t)
    t=tf.tile(t,[2,1])

    matrix=Matrix(t)
    print(matrix)

    output = spatial_transformer_network(imgs, matrix, (height, width))
    # output = spatial_transformer_network(imgs, thetas,(height, width))

    cv_out = cv2.warpPerspective(img, affine_matrix2, (width, height))

    plt.figure()
    plt.subplot(221)
    plt.title('origin')
    plt.imshow(img)

    plt.subplot(222)
    plt.title('network__point2-point1')
    plt.imshow(output[0])

    plt.subplot(223)
    plt.title('network__point1-point2')
    plt.imshow(output[1])

    plt.subplot(224)
    plt.title('point1-point2')
    plt.imshow(cv_out)
    plt.show()












