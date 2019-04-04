import tensorflow as tf


def fc_bn_relu(
        input,
        nr_nodes,
        kernel_init=tf.initializers.he_normal(),
        kernel_reg=tf.contrib.layers.l2_regularizer(scale=1.0),
        bn_momentum=0.9,
        bn_epsilon=0.0001,
        bn_axis=1,
        bn_is_training=True,
        name='fc_bn_relu'
):
    with tf.variable_scope(name):
        h = tf.layers.dense(
            input,
            units=nr_nodes,
            kernel_initializer=kernel_init,
            kernel_regularizer=kernel_reg,
            name='fc'
        )
        h = tf.layers.batch_normalization(
            h,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            axis=bn_axis,
            training=bn_is_training,
            name='bn'
        )
        h = tf.nn.relu(
            h,
            name='relu'
        )
    return h


def conv_bn_relu(
        input,
        nr_filters,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        data_format='channels_first',
        dilation_rate=(1, 1),
        kernel_init=tf.initializers.he_normal(),
        kernel_reg=tf.contrib.layers.l2_regularizer(scale=1.0),
        bn_momentum=0.9,
        bn_epsilon=0.0001,
        bn_axis=1,
        bn_is_training=True,
        name='conv_bn_relu'
):
    with tf.variable_scope(name):
        h = tf.layers.conv2d(
            input,
            filters=nr_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_init,
            kernel_regularizer=kernel_reg,
            name='conv'
        )
        h = tf.layers.batch_normalization(
            h,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            axis=bn_axis,
            training=bn_is_training,
            name='bn'
        )
        h = tf.nn.relu(
            h,
            name='relu'
        )
    return h

