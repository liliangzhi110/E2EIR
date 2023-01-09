import tensorflow as tf


class image_loss_mes(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):

        index=tf.where(y_pred>0.001)
        data_true_image=tf.gather_nd(y_true,index)
        data_pred_image=tf.gather_nd(y_pred,index)

        return tf.reduce_mean(tf.square(data_pred_image-data_true_image))


class matix_loss_mes(tf.keras.losses.Loss):

    def call(self,y_true,y_predict):

        loss = tf.reduce_mean(tf.square(y_predict - y_true))

        return loss



