
import tensorflow as tf
l2 = tf.keras.regularizers.l2


class ConvBlock(tf.keras.Model):

  def __init__(self, num_filters, data_format, bottleneck, weight_decay=1e-4,
               dropout_rate=0):
    super(ConvBlock, self).__init__()
    self.bottleneck = bottleneck

    axis = -1 if data_format == "channels_last" else 1
    inter_filter = num_filters * 4  # 每一层的输出特征图数目是growth_rate的4倍
    # don't forget to set use_bias=False when using batchnorm
    self.conv2 = tf.keras.layers.Conv2D(num_filters,
                                        (3, 3),
                                        padding="same",
                                        use_bias=False,
                                        data_format=data_format,
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=l2(weight_decay))

    self.batchnorm1 = tf.keras.layers.BatchNormalization(axis=axis)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

    if self.bottleneck:
      self.conv1 = tf.keras.layers.Conv2D(inter_filter,  # 这里可以看出如果加入bottleneck操作的话，在conv2之前特征图会降到4 * grow_rate
                                          (1, 1),
                                          padding="same",
                                          use_bias=False,
                                          data_format=data_format,
                                          kernel_initializer="he_normal",
                                          kernel_regularizer=l2(weight_decay))
      self.batchnorm2 = tf.keras.layers.BatchNormalization(axis=axis)

  def call(self, x, training=True):
    output = self.batchnorm1(x, training=training)

    if self.bottleneck:
      output = self.conv1(tf.nn.relu(output))
      output = self.batchnorm2(output, training=training)

    output = self.conv2(tf.nn.relu(output))
    output = self.dropout(output, training=training)

    return output



class transition_block(tf.keras.Model):


  def __init__(self, num_filters, data_format,
               weight_decay=1e-4, dropout_rate=0):
    super(transition_block, self).__init__()
    axis = -1 if data_format == "channels_last" else 1

    self.batchnorm = tf.keras.layers.BatchNormalization(axis=axis)
    self.conv = tf.keras.layers.Conv2D(num_filters,
                                       (1, 1),
                                       padding="same",
                                       use_bias=False,
                                       data_format=data_format,
                                       kernel_initializer="he_normal",
                                       kernel_regularizer=l2(weight_decay))
    self.avg_pool = tf.keras.layers.AveragePooling2D(data_format=data_format)

  def call(self, x, training=True):
    output = self.batchnorm(x, training=training)
    output = self.conv(tf.nn.relu(output))
    output = self.avg_pool(output)
    return output


class dense_block(tf.keras.Model):


  def __init__(self, num_layers, growth_rate, data_format, bottleneck,
               weight_decay=1e-4, dropout_rate=0):
    super(dense_block, self).__init__()
    self.num_layers = num_layers
    self.axis = -1 if data_format == "channels_last" else 1

    self.blocks = []
    for _ in range(int(self.num_layers)):
      self.blocks.append(ConvBlock(growth_rate,  # 每一层输出的特征图数目(不包括前面层的concatenate)
                                   data_format,
                                   bottleneck,
                                   weight_decay,  # 当前层的权重衰减系数
                                   dropout_rate))

  def call(self, x, training=True):

    for i in range(int(self.num_layers)):
      output = self.blocks[i](x, training=training)  # 每一层自身的输出
      x = tf.concat([x, output], axis=self.axis)  # 每一层自身的输出堆叠上前面层的输出

    return x


class Registration_model(tf.keras.Model):

    def __init__(self):
        super(Registration_model,self).__init__()

        self.conv1=tf.keras.layers.Conv2D(filters=64,kernel_size=[3,3],padding='same')
        self.pool1=tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.densenet1=dense_block(num_layers=1,growth_rate=8,data_format="channels_last",bottleneck=True)
        self.transition1=transition_block(num_filters=64,data_format="channels_last")
        #
        self.densenet2=dense_block(num_layers=2,growth_rate=8,data_format="channels_last",bottleneck=True)
        self.transition2=transition_block(num_filters=64,data_format="channels_last")
        #
        self.densenet3=dense_block(num_layers=4,growth_rate=8,data_format="channels_last",bottleneck=True)
        self.transition3=transition_block(num_filters=64,data_format="channels_last")
        #
        self.densenet4=dense_block(num_layers=8,growth_rate=8,data_format="channels_last",bottleneck=True)
        self.transition4=transition_block(num_filters=64,data_format="channels_last")

        self.densenet5=dense_block(num_layers=16,growth_rate=8,data_format="channels_last",bottleneck=True)
        self.transition5=transition_block(num_filters=64,data_format="channels_last")

        # self.densenet6=dense_block(nb_layers=32,nb_filter=64,growth_rate=8)
        # self.transition6=transition_block(nb_filter=64)
        # self.conv2=tf.keras.layers.Conv2D(filters=128,kernel_size=[1,1])
        # self.pool2=tf.keras.layers.MaxPool2D(pool_size=[1,1])

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
        # x=self.conv2(x)
        # x=self.pool2(x)

        x=self.flatten(x)
        return x
