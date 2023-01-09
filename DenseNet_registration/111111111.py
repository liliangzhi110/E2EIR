import tensorflow as tf
import numpy as np
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,             # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            activation=tf.nn.relu   # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=6)

    def call(self, inputs):
        x = self.conv1(inputs)                  # [batch_size, 28, 28, 32]
        x = self.pool1(x)                       # [batch_size, 14, 14, 32]
        x = self.conv2(x)                       # [batch_size, 14, 14, 64]
        x = self.pool2(x)                       # [batch_size, 7, 7, 64]
        x = self.flatten(x)                     # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)                      # [batch_size, 1024]
        x = self.dense2(x)                      # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output



matrixt_file= 'D:/ProgramData/DenseNet_registration/Densenet/densenet_Affine_matrix/matrix.npz'
fixed_image=np.load(matrixt_file)['path_fixed'].reshape((6084, 128, 128,1))
moving_image=np.load(matrixt_file)['path_moving'].reshape((6084, 128, 128,1))


per_matrix=np.load(matrixt_file)['Perpecttive'].reshape((fixed_image.shape[0],1,6))

fixed_image=fixed_image.astype('float32')/255
moving_image=moving_image.astype('float32')/255
con=np.concatenate((fixed_image,moving_image),axis=3)

per_matrix=per_matrix.astype('float32')

train_ds = tf.data.Dataset.from_tensor_slices((con,per_matrix)).shuffle(10000).batch(4)

print(train_ds)
print('')

model=CNN()



model=CNN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


def train_step(images,labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = tf.reduce_mean(tf.square(labels-predictions))
        print(loss.numpy())

    grads = tape.gradient(loss, model.variables)  # 使用 model.variables 这一属性直接获得模型中的所有变量
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))


EPOCHS =1
for epoch in range(EPOCHS):
  for images1, label in train_ds:
      train_step(images1, label)


