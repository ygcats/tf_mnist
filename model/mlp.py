import tensorflow as tf
import model.layers as L


def one_hidden(input, nr_out_nodes, is_training):
    #kernel initializer and regularizer
    init = tf.initializers.he_normal()
    reg = tf.contrib.layers.l2_regularizer(scale=1.0)

    h = tf.layers.flatten(input)
    h = L.fc_bn_relu(h, 512, kernel_init=init, kernel_reg=reg, bn_is_training=is_training, name='FBR1')

    h = tf.layers.dropout(h, rate=0.5, training=is_training, name='drp1')
    h = tf.layers.dense(h, nr_out_nodes, kernel_initializer=init, kernel_regularizer=reg, name='fc_last')
    return h
