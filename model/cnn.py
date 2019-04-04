import tensorflow as tf
import model.layers as L


def two_conv_one_fc(input, nr_out_nodes, is_training):
    # kernel initializer and regularizer
    init = tf.initializers.he_normal()
    reg = tf.contrib.layers.l2_regularizer(scale=1.0)

    h = L.conv_bn_relu(input, 32, kernel_init=init, kernel_reg=reg, bn_is_training=is_training, name='CBR1')
    h = tf.layers.max_pooling2d(h, pool_size=2, strides=2, padding='same', data_format='channels_first', name='mp1')

    h = L.conv_bn_relu(h, 32, kernel_init=init, kernel_reg=reg, bn_is_training=is_training, name='CBR2')
    h = tf.layers.max_pooling2d(h, pool_size=2, strides=2, padding='same', data_format='channels_first', name='mp2')

    h = tf.layers.flatten(h)
    h = tf.layers.dropout(h, rate=0.5, training=is_training, name='drp1')
    h = L.fc_bn_relu(h, 512, kernel_init=init, kernel_reg=reg, bn_is_training=is_training, name='FBR1')

    h = tf.layers.dropout(h, rate=0.5, training=is_training, name='drp2')
    h = tf.layers.dense(h, nr_out_nodes, kernel_initializer=init, kernel_regularizer=reg, name='fc_last')
    return h


def vgg16_like(input, nr_out_nodes, is_training):
    # kernel initializer and regularizer
    init = tf.initializers.he_normal()
    reg = tf.contrib.layers.l2_regularizer(scale=1.0)

    h = L.conv_bn_relu(input, 64, kernel_init=init, kernel_reg=reg, bn_is_training=is_training, name='CBR1')
    h = tf.layers.dropout(h, rate=0.3, training=is_training, name='drp1')
    h = L.conv_bn_relu(h, 64, kernel_init=init, kernel_reg=reg, bn_is_training=is_training, name='CBR2')
    h = tf.layers.max_pooling2d(h, pool_size=2, strides=2, padding='same', data_format='channels_first', name='mp1')

    h = L.conv_bn_relu(h, 128, kernel_init=init, kernel_reg=reg, bn_is_training=is_training, name='CBR3')
    h = tf.layers.dropout(h, rate=0.4, training=is_training, name='drp2')
    h = L.conv_bn_relu(h, 128, kernel_init=init, kernel_reg=reg, bn_is_training=is_training, name='CBR4')
    h = tf.layers.max_pooling2d(h, pool_size=2, strides=2, padding='same', data_format='channels_first', name='mp2')

    h = L.conv_bn_relu(h, 256, kernel_init=init, kernel_reg=reg, bn_is_training=is_training, name='CBR5')
    h = tf.layers.dropout(h, rate=0.4, training=is_training, name='drp3')
    h = L.conv_bn_relu(h, 256, kernel_init=init, kernel_reg=reg, bn_is_training=is_training, name='CBR6')
    h = tf.layers.dropout(h, rate=0.4, training=is_training, name='drp4')
    h = L.conv_bn_relu(h, 256, kernel_init=init, kernel_reg=reg, bn_is_training=is_training, name='CBR7')
    h = tf.layers.max_pooling2d(h, pool_size=2, strides=2, padding='same', data_format='channels_first', name='mp3')

    h = L.conv_bn_relu(h, 512, kernel_init=init, kernel_reg=reg, bn_is_training=is_training, name='CBR8')
    h = tf.layers.dropout(h, rate=0.4, training=is_training, name='drp5')
    h = L.conv_bn_relu(h, 512, kernel_init=init, kernel_reg=reg, bn_is_training=is_training, name='CBR9')
    h = tf.layers.dropout(h, rate=0.4, training=is_training, name='drp6')
    h = L.conv_bn_relu(h, 512, kernel_init=init, kernel_reg=reg, bn_is_training=is_training, name='CBR10')
    h = tf.layers.max_pooling2d(h, pool_size=2, strides=2, padding='same', data_format='channels_first', name='mp4')

    h = L.conv_bn_relu(h, 512, kernel_init=init, kernel_reg=reg, bn_is_training=is_training, name='CBR11')
    h = tf.layers.dropout(h, rate=0.4, training=is_training, name='drp7')
    h = L.conv_bn_relu(h, 512, kernel_init=init, kernel_reg=reg, bn_is_training=is_training, name='CBR12')
    h = tf.layers.dropout(h, rate=0.4, training=is_training, name='drp8')
    h = L.conv_bn_relu(h, 512, kernel_init=init, kernel_reg=reg, bn_is_training=is_training, name='CBR13')
    h = tf.layers.max_pooling2d(h, pool_size=2, strides=2, padding='same', data_format='channels_first', name='mp5')

    h = tf.layers.flatten(h)
    h = tf.layers.dropout(h, rate=0.5, training=is_training, name='drp9')
    h = L.fc_bn_relu(h, 512, kernel_init=init, kernel_reg=reg, bn_is_training=is_training, name='FBR1')

    h = tf.layers.dropout(h, rate=0.5, training=is_training, name='drp10')
    h = tf.layers.dense(h, nr_out_nodes, kernel_initializer=init, kernel_regularizer=reg, name='fc_last')
    return h
