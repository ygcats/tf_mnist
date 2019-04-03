import tensorflow as tf
import numpy as np
import random as rn
import time

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('logdir', './logs/', 'output directory for log files')
tf.flags.DEFINE_integer('device', 0, 'gpu_id')
tf.flags.DEFINE_integer('epochs', 100, 'the number of epochs')
tf.flags.DEFINE_integer('batch_size', 32, 'size of mini-batch')
tf.flags.DEFINE_float('init_lr', 0.05, 'initial learning rate')
tf.flags.DEFINE_integer('step_lr_decay_epochs', 25, 'step learning rate decay is applied every this number of epochs')
tf.flags.DEFINE_float('step_lr_decay_rate', 0.5, 'step learning rate decay is applied with this rate')
tf.flags.DEFINE_float('momentum', 0.9, 'μ in momentum SGD')
tf.flags.DEFINE_float('l2_lambda', 0.0005, 'l2-regularization')
tf.flags.DEFINE_integer('hidden_units', 512, 'the number of units in hidden layers')
tf.flags.DEFINE_integer('random_seed', 0, 'random seed')


def mlp_3_layers(input, hidden_nodes, output_nodes, l2_lambda, is_training):
    init = tf.initializers.he_normal()
    reg = tf.contrib.layers.l2_regularizer(scale=l2_lambda)

    input_ = tf.layers.flatten(input)
    h = tf.layers.dense(input_, hidden_nodes, kernel_initializer=init, kernel_regularizer=reg, name='fc1')
    h_n = tf.layers.batch_normalization(h, momentum=0.9, epsilon=0.0001, axis=1, training=is_training, name='bn1')
    h_n_a = tf.nn.relu(h_n, name='relu1')
    output = tf.layers.dense(h_n_a, output_nodes, kernel_initializer=init, kernel_regularizer=reg, name='fc2')
    return output


def compute_loss_acc(tf_sess, loss_name, confmat_name, input_name, target_name, is_training_name, x_input, y_target, batch):
    if x_input.shape[0] != y_target.shape[0]:
        raise Exception('invalid args')
    nr_samples = x_input.shape[0]
    nr_labels = y_target.shape[1]
    batch = min(nr_samples, max(1, batch))
    N = np.ceil(nr_samples / batch).astype('int32')
    l = 0.
    C = np.zeros(shape=[nr_labels, nr_labels], dtype='int32')
    for i in range(N):
        index = np.arange(i * batch, min((i + 1) * batch, nr_samples))
        x = x_input[index]
        y = y_target[index]
        res = tf_sess.run([loss_name, confmat_name], feed_dict={input_name: x, target_name: y, is_training_name: False})
        l += len(index) * res[0]
        C += res[1]
    l /= nr_samples
    acc_all = C.trace() / nr_samples
    acc_matrix = C / y_target.sum(axis=0).reshape(nr_labels, 1)
    acc_class_ave = acc_matrix.trace() / nr_labels
    return [l, acc_all, acc_class_ave]


def main(argv):
    # parameters
    np.random.seed(FLAGS.random_seed)
    rn.seed(FLAGS.random_seed)
    log_dir = FLAGS.logdir
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    init_lr = FLAGS.init_lr
    lr_decay_epochs = FLAGS.step_lr_decay_epochs
    lr_decay_rate = FLAGS.step_lr_decay_rate
    momentum = FLAGS.momentum
    l2_lambda = FLAGS.l2_lambda
    nr_hidden_nodes = FLAGS.hidden_units
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=str(FLAGS.device),  # specify GPU number
            allow_growth=True
        )
    )

    # TensorBoard
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    #load data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
    x_train = np.squeeze(np.array([[x.flatten()] for x in x_train]))
    x_test = np.squeeze(np.array([[x.flatten()] for x in x_test]))
    nr_input_nodes = x_train.shape[1]
    nr_labels = len(np.unique(y_train))
    nr_training_samples, nr_test_samples = x_train.shape[0], x_test.shape[0]
    nr_iterations_per_epoch = np.ceil(nr_training_samples / batch_size).astype('int32')
    remainder = nr_training_samples % batch_size
    y_train_one_hot = np.eye(nr_labels, dtype='float32')[y_train]
    y_test_one_hot = np.eye(nr_labels, dtype='float32')[y_test]

    #make graph
    MLP = tf.Graph()
    with MLP.as_default():
        with tf.device('/cpu:0'):
            global_step = tf.Variable(0, trainable=False, name='global_step')
        with tf.device('/gpu:0'):
            tf.set_random_seed(0)
            with tf.variable_scope('data'):
                input = tf.placeholder(shape=[None, nr_input_nodes], dtype=tf.float32, name='input')
                target = tf.placeholder(shape=[None, nr_labels], dtype=tf.float32, name='label')
                is_training = tf.placeholder(dtype=tf.bool, name='is_training')
            with tf.variable_scope('model'):
                logit_output = mlp_3_layers(input, nr_hidden_nodes, nr_labels, l2_lambda, is_training)
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit_output, labels=target), name='cross_entropy')
                regularization = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='regularization')
                objective = loss + regularization
            with tf.name_scope('update'):
                learning_rate = tf.train.exponential_decay(init_lr, global_step, lr_decay_epochs * nr_iterations_per_epoch, lr_decay_rate, staircase=True)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(objective, global_step=global_step)
                    # train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(objective, global_step=global_step)
            with tf.variable_scope('model/fc1', reuse=True):
                l2_W1 = tf.norm(tf.get_variable('kernel'), ord='euclidean', axis=0, name='l2_W1')
        with tf.device('/cpu:0'):
            with tf.name_scope('acc'):
                confusion_matrix = tf.confusion_matrix(tf.argmax(target, 1), tf.argmax(logit_output, 1), nr_labels, name='confusion_matrix')
    #session
    #print(MLP.get_operations())
    #loss_iter_summary = tf.summary.scalar('loss_iter', loss)
    l2_W1_summary = tf.summary.histogram('l2_W1', l2_W1)
    start = time.time()
    with tf.Session(graph=MLP, config=config) as sess:
        print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        print(tf.trainable_variables())
        print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        FileWriter_train = tf.summary.FileWriter(log_dir + 'train', sess.graph)
        FileWriter_valid = tf.summary.FileWriter(log_dir + 'valid', sess.graph)
        sess.run(tf.global_variables_initializer())

        w_init = sess.run('model/fc1/kernel:0')
        print(w_init.shape)
        #for i in range(w_init.shape[0]):
        #print(1.0e-4 * 0.5 * np.sum(w_init**2, axis=1))
        print(sess.run('model/fc1/kernel/Regularizer/l2_regularizer:0'))
        #print(1.0e-4 * 0.5 * sess.run(tf.norm(w_init, ord='euclidean', axis=1))**2)
        print(1.0e-4 * 0.5 * np.sum(sess.run(l2_W1)**2))

        for i in range(epochs):
            rand_index = np.random.permutation(nr_training_samples)
            if remainder > 0:
                rand_index = np.append(rand_index, rand_index[np.random.choice(nr_training_samples - remainder, size=batch_size - remainder, replace=False)])
            for j in range(nr_iterations_per_epoch):
                p = rand_index[j * batch_size:(j + 1) * batch_size]
                x = x_train[p]
                y = y_train_one_hot[p]
                sess.run(train_step, feed_dict={input: x, target: y, is_training: True})
                #loss_iter, _ = sess.run([loss_iter_summary, train_step], feed_dict={input: x, target: y, is_training: True})
                #FileWriter_train.add_summary(loss_iter, i * nr_iterations_per_epoch + j)

            FileWriter_train.add_summary(sess.run(l2_W1_summary), i + 1)

            # evaluation on training set
            loss_acc_train = compute_loss_acc(sess, 'loss/cross_entropy:0', 'acc/confusion_matrix/SparseTensorDenseAdd:0', 'data/input:0', 'data/label:0', 'data/is_training:0', x_train, y_train_one_hot, 1000)
            FileWriter_train.add_summary(tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=loss_acc_train[0])]), i + 1)
            FileWriter_train.add_summary(tf.Summary(value=[tf.Summary.Value(tag='acc_all', simple_value=loss_acc_train[1])]), i + 1)
            FileWriter_train.add_summary(tf.Summary(value=[tf.Summary.Value(tag='acc_class', simple_value=loss_acc_train[2])]), i + 1)
            # evaluation on validation set
            loss_acc_valid = compute_loss_acc(sess, 'loss/cross_entropy:0', 'acc/confusion_matrix/SparseTensorDenseAdd:0', 'data/input:0', 'data/label:0', 'data/is_training:0', x_test, y_test_one_hot, 1000)
            FileWriter_valid.add_summary(tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=loss_acc_valid[0])]), i + 1)
            FileWriter_valid.add_summary(tf.Summary(value=[tf.Summary.Value(tag='acc_all', simple_value=loss_acc_valid[1])]), i + 1)
            FileWriter_valid.add_summary(tf.Summary(value=[tf.Summary.Value(tag='acc_class', simple_value=loss_acc_valid[2])]), i + 1)
            # print log
            if i == 0:
                print('{:<10}{:^75}{:^75}'.format('', 'training', 'validation'))
                print('{:<10}{:<25}{:<25}{:<25}{:<25}{:<25}{:<25}'.format('epoch', 'loss_train', 'acc_all_train', 'acc_class_train', 'loss_valid', 'acc_all_valid', 'acc_class_valid'))
            print('{:<10}{:<25.9}{:<25.9}{:<25.9}{:<25.9}{:<25.9}{:<25.9}'.format(i + 1, loss_acc_train[0], loss_acc_train[1], loss_acc_train[2], loss_acc_valid[0], loss_acc_valid[1], loss_acc_valid[2]))
        FileWriter_train.close()
        FileWriter_valid.close()
    print('training time:', time.time() - start, 'sec')


if __name__ == '__main__':
    tf.app.run()
