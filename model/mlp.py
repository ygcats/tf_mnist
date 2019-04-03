import tensorflow as tf


def mlp_3_layers(input, hidden_nodes, output_nodes, l2_lambda, is_training):
    init = tf.initializers.he_normal()
    reg = tf.contrib.layers.l2_regularizer(scale=l2_lambda)

    input_ = tf.layers.flatten(input)
    h = tf.layers.dense(input_, hidden_nodes, kernel_initializer=init, kernel_regularizer=reg, name='fc1')
    h_n = tf.layers.batch_normalization(h, momentum=0.9, epsilon=0.0001, axis=1, training=is_training, name='bn1')
    h_n_a = tf.nn.relu(h_n, name='relu1')
    output = tf.layers.dense(h_n_a, output_nodes, kernel_initializer=init, kernel_regularizer=reg, name='fc2')
    return output