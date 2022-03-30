# -*- coding:utf-8 -*-
import tensorflow as tf

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


def resBlock(inputs, filters, kernel_size=3):

    res = inputs
    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding="same")(inputs)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding="same")(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h + res)

    return h

def lr_BLock_l3(inputs, filters, kernel_size=3):

    x_down2 = tf.image.resize(inputs, [int(inputs.shape[1] / 2), int(inputs.shape[2] / 2)])
    x_down4 = tf.image.resize(x_down2, [int(x_down2.shape[1] / 2), int(x_down2.shape[2] / 2)])

    x_reup2 = tf.image.resize(x_down4, [x_down4.shape[1] * 2, x_down4.shape[2] * 2])
    x_reup = tf.image.resize(x_down2, [x_down2.shape[1] * 2, x_down2.shape[2] * 2])

    Laplace_2 = x_down2 - x_reup2
    Laplace_1 = inputs - x_reup

    Scale1 = resBlock(x_down4, filters, kernel_size)
    Scale2 = resBlock(Laplace_2, filters, kernel_size)
    Scale3 = resBlock(Laplace_1, filters, kernel_size)

    output1 = Scale1
    output2 = tf.image.resize(Scale1, [Scale1.shape[1] * 2, Scale1.shape[2] * 2]) + Scale2
    output3 = tf.image.resize(output2, [output2.shape[1] * 2, output2.shape[2] * 2]) + Scale3

    return output3

def lr_BLock_l2(inputs, filters, kernel_size=3):

    x_donw2 = tf.image.resize(inputs, [int(inputs.shape[1] / 2), int(inputs.shape[2] / 2)])

    x_reup = tf.image.resize(x_donw2, [x_donw2.shape[1] * 2, x_donw2.shape[2] * 2])

    Laplace_1 = inputs - x_reup

    Scale1 = resBlock(x_donw2, filters, kernel_size)
    Scale2 = resBlock(Laplace_1, filters, kernel_size)

    output1 = Scale1
    output2 = tf.image.resize(Scale1, [Scale1.shape[1] * 2, Scale1.shape[2] * 2])

    return output2

def LMSB(input_shape=(64, 64, 3), NDF=64):

    h = inputs = tf.keras.Input(input_shape)

    input = h

    h = tf.keras.layers.Conv2D(filters=NDF*1, kernel_size=5, strides=2, padding="same")(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h1 = lr_BLock_l3(h, NDF*1, 3)

    h = tf.keras.layers.Conv2D(filters=NDF*2, kernel_size=5, strides=2, padding="same")(h1)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h2 = lr_BLock_l3(h, NDF*2, 3)

    h = tf.keras.layers.Conv2D(filters=NDF*4, kernel_size=5, strides=2, padding="same")(h2)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = lr_BLock_l3(h, NDF*4, 3)

    h = lr_BLock_l3(h, NDF*4, 3)
    h = tf.keras.layers.Conv2DTranspose(filters=NDF*2, kernel_size=4, strides=2, padding="same")(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = lr_BLock_l3(h+h2, NDF*2, 3)
    h = tf.keras.layers.Conv2DTranspose(filters=NDF*1, kernel_size=4, strides=2, padding="same")(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = lr_BLock_l3(h+h1, NDF*1, 3)
    h = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding="same")(h) + input
    #h = tf.nn.tanh(h)
    
    return tf.keras.Model(inputs=inputs, outputs=h) # later

def LMSB_2(input_shape=(256, 256, 3), NDF2=32):

    h = inputs = tf.keras.Input(input_shape)

    input = h

    h = tf.keras.layers.Conv2D(filters=NDF2*1, kernel_size=5, strides=2, padding="same")(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h1 = lr_BLock_l2(h, NDF2*1, 3)
    
    h = tf.keras.layers.Conv2D(filters=NDF2*2, kernel_size=5, strides=2, padding="same")(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h2 = lr_BLock_l2(h, NDF2*2, 3)

    h = tf.keras.layers.Conv2D(filters=NDF2*4, kernel_size=5, strides=2, padding="same")(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = lr_BLock_l2(h, NDF2*4, 3)

    h = lr_BLock_l2(h, NDF2*4, 3)
    h = tf.keras.layers.Conv2DTranspose(filters=NDF2*2, kernel_size=4, strides=2, padding="same")(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = lr_BLock_l2(h+h2, NDF2*2, 3)
    h = tf.keras.layers.Conv2DTranspose(filters=NDF2*1, kernel_size=4, strides=2, padding="same")(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = lr_BLock_l2(h+h1, NDF2*1, 3)
    h = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding="same")(h) + input
    #h = tf.nn.tanh(h)

    return tf.keras.Model(inputs=inputs, outputs=h) # later
