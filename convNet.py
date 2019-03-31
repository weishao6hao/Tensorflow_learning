#coding=utf8

import tensorflow as tf

filter_weight = tf.Variable('weights',[5,5,3,16],tf.truncated_normal_initializer(stddev=0.1))
biases = tf.get_variable('biases',[16],initializer=tf.constant_initializer(0.1))

conv = tf.nn.conv2d(input,filter_weight,strides=[1,1,1,1],padding='SAME')
bias = tf.nn.bias_add(conv,biases)
actived_conv = tf.nn.relu(bias)

pool = tf.nn.max_pool(actived_conv,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

x = tf.placeholder(tf.float32,[
        BATCH_SIZE,
        mnist_inference.IMAGE_SIZE,
        mnist_inference.IMAGE_SIZE,
        mnist_inference.NUM_CHANNELS],
        name='x-input')
reshape_xs = np.reshape(xs,(BATCH_SIZE,
                            mnist_inference.IMAGE_SIZE,
                            mnist_inference.IMAGE_SIZE,
                            mnist_inference.NUM_CHANNELS))

