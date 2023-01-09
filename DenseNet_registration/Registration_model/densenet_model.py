import tensorflow as tf


class conv_block(tf.keras.layers.Layer):

    def __init__(self,number_filter,
                 bottleneck=True,):
        super(conv_block,self).__init__()


        self.bottleneck=bottleneck
        concat_axis = 1 if tf.keras.backend.image_data_format() == 'channel_first' else -1

        self.bn1 = tf.keras.layers.BatchNormalization(axis=concat_axis)

        self.relu1 = tf.keras.layers.Activation('relu')
        inter_channel = number_filter * 4

        self.conv1 = tf.keras.layers.Conv2D(
                                            filters=inter_channel,
                                            kernel_size=[1, 1],
                                            padding='same',
                                            use_bias=False,
                                            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
                                            )
        self.bn2 = tf.keras.layers.BatchNormalization(axis=concat_axis)

        self.relu2 = tf.keras.layers.Activation('relu')

        self.conv2 = tf.keras.layers.Conv2D(
                                            filters=number_filter,
                                            kernel_size=(3, 3),
                                            padding='same',
                                            use_bias=False,
                                            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
                                            )


    def call(self,Inputtensor,training=True):

        x=self.bn1(Inputtensor,training=training)
        x=self.relu1(x)
        if self.bottleneck:
            x=self.conv1(x)
            x=self.bn2(x,training=training)
            x=self.relu2(x)
        x=self.conv2(x)
        return x


class transition_block(tf.keras.layers.Layer):

    def __init__(self,nb_filter, compression=1):
        super(transition_block,self).__init__()

        concat_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

        self.bn = tf.keras.layers.BatchNormalization(axis=concat_axis)

        self.relu = tf.keras.layers.Activation('relu')

        self.conv = tf.keras.layers.Conv2D(filters=int(nb_filter * compression),
                                           kernel_size=(1, 1),
                                           padding='same',
                                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)
                                           )

        self.pool = tf.keras.layers.MaxPool2D((2, 2), strides=2)


    def call(self,Inputtensor,training=True):
        x=self.bn(Inputtensor,training=training)
        x=self.relu(x)
        x=self.conv(x)
        x=self.pool(x)
        return x



class dense_block(tf.keras.layers.Layer):
    def __init__(self,
                 nb_layers,
                 growth_rate,
                 bottleneck=True
                ):
        super(dense_block,self).__init__()

        self.nb_layers=nb_layers
        self.growth_rate = growth_rate
        self.bottleneck=bottleneck

        self.concat_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

        self.blocks = []
        for _ in range(int(self.nb_layers)):
            self.blocks.append(conv_block(self.growth_rate,  # 每一层输出的特征图数目(不包括前面层的concatenate)
                                          self.bottleneck
                                         ))


    def call(self,Input_tensor,training=True):

        for i in range(self.nb_layers):
            output = self.blocks[i](Input_tensor, training=training)
            Input_tensor = tf.concat([Input_tensor, output], self.concat_axis)
        return Input_tensor







class Registration_model(tf.keras.layers.Layer):

    def __init__(self):
        super(Registration_model,self).__init__()

        self.conv1=tf.keras.layers.Conv2D(filters=64,kernel_size=[3,3],padding='same')
        self.pool1=tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.densenet1=dense_block(nb_layers=1,growth_rate=8)
        self.transition1=transition_block(nb_filter=96)
        #
        self.densenet2=dense_block(nb_layers=2,growth_rate=8)
        self.transition2=transition_block(nb_filter=128)
        #
        self.densenet3=dense_block(nb_layers=4,growth_rate=8)
        self.transition3=transition_block(nb_filter=256)
        #
        self.densenet4=dense_block(nb_layers=8,growth_rate=8)
        self.transition4=transition_block(nb_filter=512)

        self.densenet5=dense_block(nb_layers=16,growth_rate=8)
        self.transition5=transition_block(nb_filter=1024)

        # self.densenet6=dense_block(nb_layers=32,nb_filter=64,growth_rate=8)
        # self.transition6=transition_block(nb_filter=64)
        self.conv2=tf.keras.layers.Conv2D(filters=1024,kernel_size=[1,1])
        self.pool2=tf.keras.layers.MaxPool2D(pool_size=[1,1])

        self.flatten=tf.keras.layers.Flatten()


    def call(self,Inputensor_1):

        x=self.conv1(Inputensor_1)
        x=self.pool1(x)

        x=self.densenet1(x)
        x=self.transition1(x)
        x=self.densenet2(x)
        x=self.transition2(x)
        x = self.densenet3(x)
        x = self.transition3(x)
        x = self.densenet4(x)
        x = self.transition4(x)
        x = self.densenet5(x)
        x = self.transition5(x)

        # x = self.densenet6(x)
        # x = self.transition6(x)
        x=self.conv2(x)
        x=self.pool2(x)

        x=self.flatten(x)
        return x



class spatial_transformer_network(tf.keras.layers.Layer):

    def __init__(self):

        super(spatial_transformer_network,self).__init__()

    def call(self,inputs_theta):

        return self.spatial_transformer_network(inputs_theta)

    @tf.function
    def spatial_transformer_network(self,input_fmap_theta):

        input_fmap, theta=input_fmap_theta
        # grab input dimensions
        B = tf.shape(input_fmap)[0]
        H = tf.shape(input_fmap)[1]
        W = tf.shape(input_fmap)[2]

        # reshape theta to (B, 2, 3)
        theta = tf.reshape(theta, [B, 3, 3])

        batch_grids = self.affine_grid_generator(H, W, theta,B)

        x_s = batch_grids[:, 0, :, :] / batch_grids[:, 2, :, :]
        y_s = batch_grids[:, 1, :, :] / batch_grids[:, 2, :, :]

        out_fmap = self.bilinear_sampler(input_fmap, x_s, y_s)

        return out_fmap

    @tf.function
    def get_pixel_value(self,img, x, y):

        shape = tf.shape(img)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, height, width))

        indices = tf.stack([b, y, x], 3)

        return tf.gather_nd(img, indices)

    @tf.function
    def affine_grid_generator(self,height, width, theta,batchsize):


        num_batch = batchsize
        # create normalized 2D grid
        x = tf.linspace(0.0, 256.0, width)
        y = tf.linspace(0.0, 256.0, height)
        x_t, y_t = tf.meshgrid(x, y)

        # flatten
        x_t_flat = tf.reshape(x_t, [-1])
        y_t_flat = tf.reshape(y_t, [-1])

        # reshape to [x_t, y_t , 1] - (homogeneous form)
        ones = tf.ones_like(x_t_flat)
        sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

        # repeat grid num_batch times
        sampling_grid = tf.expand_dims(sampling_grid, axis=0)  # (1, 3, 250000)
        sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))  # (2, 3, 250000)

        # cast to float32 (required for matmul)
        theta = tf.cast(theta, 'float32')  # (2,6)

        sampling_grid = tf.cast(sampling_grid, 'float32')  # (2, 3, 250000)

        # transform the sampling grid - batch multiply
        batch_grids = tf.matmul(theta, sampling_grid)  # (2, 2, 250000)


        # reshape to (num_batch, H, W, 2)
        batch_grids = tf.reshape(batch_grids, [num_batch, 3, height, width])

        return batch_grids

    @tf.function
    def bilinear_sampler(self,img, x, y):

        H = tf.shape(img)[1]
        W = tf.shape(img)[2]
        max_y = tf.cast(H - 1, 'int32')
        max_x = tf.cast(W - 1, 'int32')
        zero = tf.zeros([], dtype='int32')

        # rescale x and y to [0, W-1/H-1]
        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')


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
        Ia = self.get_pixel_value(img, x0, y0)
        Ib = self.get_pixel_value(img, x0, y1)
        Ic = self.get_pixel_value(img, x1, y0)
        Id = self.get_pixel_value(img, x1, y1)

        # recast as float for delta calculation
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        # calculate deltas
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        # add dimension for addition
        wa = tf.expand_dims(wa, axis=3)
        wb = tf.expand_dims(wb, axis=3)
        wc = tf.expand_dims(wc, axis=3)
        wd = tf.expand_dims(wd, axis=3)

        # compute output
        out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

        return out


class getPerspectiveTransformMatrix(tf.keras.layers.Layer):

    def __init__(self,**kwargs):
        super(getPerspectiveTransformMatrix,self).__init__(**kwargs)

    def call(self,inputs):
        return self.Matrix(inputs)

    @tf.function
    def Matrix(self,input):

        batch = tf.shape(input)[0]
        pointtemp1 = tf.convert_to_tensor([[0, 0], [500, 0], [500, 500], [0, 500]], dtype=tf.float32)
        pointtemp1 = tf.reshape(pointtemp1, (1, 8))
        pointtemp1 = tf.tile(pointtemp1, [batch, 1])
        pointtemp2 = tf.subtract(pointtemp1, input)

        point1 = pointtemp2
        point2 = pointtemp1

        x1, x2, x3, x4 = tf.reshape(point1[:, 0:1],(-1,1,1)), tf.reshape(point1[:, 2:3],(-1,1,1)), tf.reshape(point1[:, 4:5],(-1,1,1)), tf.reshape(point1[:, 6:7],(-1,1,1))
        y1, y2, y3, y4 = tf.reshape(point1[:, 1:2],(-1,1,1)), tf.reshape(point1[:, 3:4],(-1,1,1)), tf.reshape(point1[:, 5:6],(-1,1,1)), tf.reshape(point1[:, 7:8],(-1,1,1))

        u1, u2, u3, u4 = tf.reshape(point2[:, 0:1],(-1,1,1)), tf.reshape(point2[:, 2:3],(-1,1,1)), tf.reshape(point2[:, 4:5],(-1,1,1)), tf.reshape(point2[:, 6:7],(-1,1,1))
        v1, v2, v3, v4 = tf.reshape(point2[:, 1:2],(-1,1,1)), tf.reshape(point2[:, 3:4],(-1,1,1)), tf.reshape(point2[:, 5:6],(-1,1,1)), tf.reshape(point2[:, 7:8],(-1,1,1))

        #============================================第一点============================================
        oneline_x1pad0=tf.pad(x1,[[0,0],[0,7],[0,8]],constant_values=0)
        oneline_y1pad1 = tf.pad(y1, [[0, 0],[0,7], [1, 7]], constant_values=0)
        oneline_1pad2=tf.pad(tf.ones(shape=(batch,1,1),dtype=tf.float32),[[0, 0],[0,7], [2, 6]],constant_values=0)

        oneline_u1pad6=tf.pad(u1,[[0, 0],[0,7], [6, 2]],constant_values=0)
        oneline_x1pad6=tf.pad(x1,[[0, 0],[0,7], [6, 2]],constant_values=0)

        oneline_u1pad7=tf.pad(u1,[[0, 0],[0,7], [7, 1]],constant_values=0)
        oneline_y1pad7=tf.pad(y1,[[0, 0],[0,7], [7, 1]],constant_values=0)

        oneline_u1pad8=tf.pad(u1,[[0, 0],[0,7], [8, 0]],constant_values=0)



        twoline_x1pad3=tf.pad(x1,[[0,0],[1,6],[3,5]],constant_values=0)
        twoline_y1pad4=tf.pad(y1, [[0, 0],[1,6], [4, 4]], constant_values=0)
        twoline_1pad5 = tf.pad(tf.ones(shape=(batch, 1,1), dtype=tf.float32), [[0, 0],[1,6], [5, 3]], constant_values=0)

        twoline_v1pad6=tf.pad(v1,[[0, 0],[1,6], [6, 2]],constant_values=0)
        twoline_x1pad6 = tf.pad(x1, [[0, 0], [1, 6], [6, 2]], constant_values=0)

        twoline_v1pad7 = tf.pad(v1, [[0, 0], [1, 6], [7, 1]], constant_values=0)
        twoline_y1pad7 = tf.pad(y1, [[0, 0], [1, 6], [7, 1]], constant_values=0)

        twoline_v1pad8 = tf.pad(v1, [[0, 0], [1, 6], [8, 0]], constant_values=0)

        con1=oneline_x1pad0+oneline_y1pad1+oneline_1pad2+(-oneline_u1pad6*oneline_x1pad6)+(-oneline_u1pad7*oneline_y1pad7)+(-oneline_u1pad8)
        con2=twoline_x1pad3+twoline_y1pad4+twoline_1pad5+(-twoline_v1pad6*twoline_x1pad6)+(-twoline_v1pad7*twoline_y1pad7)+(-twoline_v1pad8)



        # ============================================第2点============================================
        threeline_x2pad0 = tf.pad(x2, [[0, 0], [2, 5], [0, 8]], constant_values=0)
        threeline_y2pad1 = tf.pad(y2, [[0, 0], [2, 5], [1, 7]], constant_values=0)
        threeline_1pad2 = tf.pad(tf.ones(shape=(batch, 1,1), dtype=tf.float32), [[0, 0], [2, 5], [2, 6]], constant_values=0)

        threeline_u2pad6 = tf.pad(u2, [[0, 0], [2, 5], [6, 2]], constant_values=0)
        threeline_x2pad6 = tf.pad(x2, [[0, 0], [2, 5], [6, 2]], constant_values=0)

        threeline_u2pad7 = tf.pad(u2, [[0, 0], [2, 5], [7, 1]], constant_values=0)
        threeline_y2pad7 = tf.pad(y2, [[0, 0], [2, 5], [7, 1]], constant_values=0)

        threeline_u2pad8 = tf.pad(u2, [[0, 0], [2, 5], [8, 0]], constant_values=0)



        fourline_x2pad3 = tf.pad(x2, [[0, 0], [3, 4], [3, 5]], constant_values=0)
        fourline_y2pad4 = tf.pad(y2, [[0, 0], [3, 4], [4, 4]], constant_values=0)
        fourline_1pad5 = tf.pad(tf.ones(shape=(batch, 1,1), dtype=tf.float32), [[0, 0], [3, 4], [5, 3]], constant_values=0)

        fourline_v2pad6 = tf.pad(v2, [[0, 0], [3, 4], [6, 2]], constant_values=0)
        fourline_x2pad6 = tf.pad(x2, [[0, 0], [3, 4], [6, 2]], constant_values=0)

        fourline_v2pad7 = tf.pad(v2, [[0, 0], [3, 4], [7, 1]], constant_values=0)
        fourline_y2pad7 = tf.pad(y2, [[0, 0], [3, 4], [7, 1]], constant_values=0)

        fourline_v2pad8 = tf.pad(v2, [[0, 0], [3, 4], [8, 0]], constant_values=0)

        con3 = threeline_x2pad0+threeline_y2pad1+threeline_1pad2+(-threeline_u2pad6*threeline_x2pad6)+(-threeline_u2pad7*threeline_y2pad7)+(-threeline_u2pad8)
        con4=fourline_x2pad3+fourline_y2pad4+fourline_1pad5+(-fourline_v2pad6*fourline_x2pad6)+(-fourline_v2pad7*fourline_y2pad7)+(-fourline_v2pad8 )
        # ============================================第3点============================================
        fiveline_x3pad0 = tf.pad(x3, [[0, 0], [4, 3], [0, 8]], constant_values=0)
        fiveline_y3pad1 = tf.pad(y3, [[0, 0], [4, 3], [1, 7]], constant_values=0)
        fiveline_1pad2 = tf.pad(tf.ones(shape=(batch, 1,1), dtype=tf.float32), [[0, 0], [4, 3], [2, 6]], constant_values=0)

        fiveline_u3pad6 = tf.pad(u3, [[0, 0], [4, 3], [6, 2]], constant_values=0)
        fiveline_x3pad6 = tf.pad(x3, [[0, 0], [4, 3], [6, 2]], constant_values=0)

        fiveline_u3pad7 = tf.pad(u3, [[0, 0], [4, 3], [7, 1]], constant_values=0)
        fiveline_y3pad7 = tf.pad(y3, [[0, 0], [4, 3], [7, 1]], constant_values=0)

        fiveline_u3pad8 = tf.pad(u3, [[0, 0], [4, 3], [8, 0]], constant_values=0)



        sixline_x3pad3 = tf.pad(x3, [[0, 0], [5, 2], [3, 5]], constant_values=0)
        sixline_y3pad4 = tf.pad(y3, [[0, 0], [5, 2], [4, 4]], constant_values=0)
        sixline_1pad5 = tf.pad(tf.ones(shape=(batch, 1,1), dtype=tf.float32), [[0, 0], [5, 2], [5, 3]], constant_values=0)

        sixline_v3pad6 = tf.pad(v3, [[0, 0], [5, 2], [6, 2]], constant_values=0)
        sixline_x3pad6 = tf.pad(x3, [[0, 0], [5, 2], [6, 2]], constant_values=0)

        sixline_v3pad7 = tf.pad(v3, [[0, 0], [5, 2], [7, 1]], constant_values=0)
        sixline_y3pad7 = tf.pad(y3, [[0, 0], [5, 2], [7, 1]], constant_values=0)

        sixline_v3pad8 = tf.pad(v3, [[0, 0], [5, 2], [8, 0]], constant_values=0)

        con5 = fiveline_x3pad0 + fiveline_y3pad1 + fiveline_1pad2 + (-fiveline_u3pad6 * fiveline_x3pad6) + (
                    -fiveline_u3pad7 * fiveline_y3pad7) + (-fiveline_u3pad8)
        con6= sixline_x3pad3 + sixline_y3pad4 + sixline_1pad5 + (-sixline_v3pad6 * sixline_x3pad6) + (
                    -sixline_v3pad7 * sixline_y3pad7) + (-sixline_v3pad8)

        # ============================================第4点============================================
        sevenline_x4pad0 = tf.pad(x4, [[0, 0], [6, 1], [0, 8]], constant_values=0)
        sevenline_y4pad1 = tf.pad(y4, [[0, 0], [6, 1], [1, 7]], constant_values=0)
        sevenline_1pad2 = tf.pad(tf.ones(shape=(batch, 1,1), dtype=tf.float32), [[0, 0], [6, 1], [2, 6]], constant_values=0)

        sevenline_u4pad6 = tf.pad(u4, [[0, 0], [6, 1], [6, 2]], constant_values=0)
        sevenline_x4pad6 = tf.pad(x4, [[0, 0], [6, 1], [6, 2]], constant_values=0)

        sevenline_u4pad7 = tf.pad(u4, [[0, 0], [6, 1], [7, 1]], constant_values=0)
        sevenline_y4pad7 = tf.pad(y4, [[0, 0], [6, 1], [7, 1]], constant_values=0)

        sevenline_u4pad8 = tf.pad(u4, [[0, 0], [6, 1], [8, 0]], constant_values=0)



        eightline_x4pad3 = tf.pad(x4, [[0, 0], [7, 0], [3, 5]], constant_values=0)
        eightline_y4pad4 = tf.pad(y4, [[0, 0], [7, 0], [4, 4]], constant_values=0)
        eightline_1pad5 = tf.pad(tf.ones(shape=(batch, 1,1), dtype=tf.float32), [[0, 0], [7, 0], [5, 3]], constant_values=0)

        eightline_v4pad6 = tf.pad(v4, [[0, 0], [7, 0], [6, 2]], constant_values=0)
        eightline_x4pad6 = tf.pad(x4, [[0, 0], [7, 0], [6, 2]], constant_values=0)

        eightline_v4pad7 = tf.pad(v4, [[0, 0], [7, 0], [7, 1]], constant_values=0)
        eightline_y4pad7 = tf.pad(y4, [[0, 0], [7, 0], [7, 1]], constant_values=0)

        eightline_v4pad8 = tf.pad(v4, [[0, 0], [7, 0], [8, 0]], constant_values=0)

        con7 = sevenline_x4pad0 + sevenline_y4pad1 + sevenline_1pad2 + (-sevenline_u4pad6 * sevenline_x4pad6) + (
                -sevenline_u4pad7 * sevenline_y4pad7) + (-sevenline_u4pad8)
        con8 = eightline_x4pad3 + eightline_y4pad4 + eightline_1pad5 + (-eightline_v4pad6 * eightline_x4pad6) + (
                -eightline_v4pad7 * eightline_y4pad7) + (-eightline_v4pad8)

        #=============================================计算=============================================
        con_con=con1+con2+con3+con4+con5+con6+con7+con8

        U,S,Vh=tf.linalg.svd(con_con,full_matrices=True)
        Vh=tf.transpose(Vh)
        L = Vh[-1, :] / Vh[-1, -1]
        L=tf.transpose(L)

        return L






































































































