import tensorflow as tf

x = tf.random_uniform([100, 784], minval=0, maxval=1.0)
y = tf.random_uniform([100, 10], minval=0, maxval=1.0)

GRIDS = 20
def core_function1d(x, y, grids = GRIDS):
    return tf.maximum((1/(grids - 1)) - tf.abs(tf.subtract(x, y)), 0)

def core_function2d(x1, x2, y1, y2, grids1 = GRIDS, grids2 = GRIDS):
    return core_function1d(x1, y1, grids1) + core_function1d(x2, y2, grids1)

def entropy1d(x, grids = GRIDS):
    shape1 = [x.get_shape().as_list()[0], 1, x.get_shape().as_list()[1]]
    shape2 = [1, grids, 1]

    gx = tf.linspace(0.0, 1.0, grids)

    X = tf.reshape(x, shape1)
    GX = tf.reshape(gx, shape2)

    mapping = core_function1d(GX, X, grids)
    mapping = tf.reduce_sum(mapping, 0)
    mapping = tf.add(mapping, 1e-10)
    mapping_normalized = tf.divide(mapping, tf.reduce_sum(mapping, 0, keepdims = True))

    entropy = tf.negative(tf.reduce_sum(tf.reduce_sum(tf.multiply(mapping_normalized, tf.log(mapping_normalized * grids)), 0)))

    return entropy

def entropy2d(x, y, gridsx = GRIDS, gridsy = GRIDS):
    batch_size = x.get_shape().as_list()[0]
    x_szie = x.get_shape().as_list()[1]
    y_size = y.get_shape().as_list()[1]

    gx = tf.linspace(0.0, 1.0, gridsx)
    gy = tf.linspace(0.0, 1.0, gridsy)

    X = tf.reshape(x, [batch_size, 1, 1, x_szie, 1])
    Y = tf.reshape(y, [batch_size, 1, 1, 1, y_size])

    GX = tf.reshape(gx, [1, gridsx, 1, 1, 1])
    GY = tf.reshape(gy, [1, 1, gridsy, 1, 1])

    mapping = core_function2d(GX, GY, X, Y, gridsx, gridsy)
    mapping = tf.reduce_sum(mapping, 0)
    mapping = tf.add(mapping, 1e-10)
    mapping_normalized = tf.divide(mapping, tf.reduce_sum(mapping, [0, 1], keepdims = True))

    entropy = tf.negative(tf.reduce_sum(tf.reduce_sum(tf.multiply(mapping_normalized, tf.log(mapping_normalized * (gridsx *gridsy))), [0, 1])))

    return entropy

def matul_info(x, y):
  ex = entropy1d(x)
  ey = entropy1d(y)
  exy = entropy2d(x, y)
  return ex + ey - exy

multi_info = entropy1d(x) + entropy1d(y) - entropy2d(x, y)
with tf.Session():
  print(multi_info.eval())