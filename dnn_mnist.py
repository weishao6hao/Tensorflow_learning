#coding=utf8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import os
import numpy as np
# mnist = input_data.read_data_sets('data/',one_hot=True)

# mnist 数据集相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10

#配置神经网络参数
LAYER1_NODE = 500
BATCH_SIZE = 100

LEARNING_RATE_BASE  =0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = 'model/'
MODEL_NAME = 'model.ckpt'

#辅助函数，计算网络前向传播结果
# def inference(input_tensor,avg_class,weight1,bias1,
#               weights2,bias2):
#     if avg_class==None:
#         layer1 = tf.nn.relu(tf.matmul(input_tensor,weight1)+bias1)
#         return tf.matmul(layer1,weights2)+bias2
#     else:
#         layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weight1))+bias1)
#         return tf.matmul(layer1,avg_class.average(weights2))+bias2

#模型训练过程
def train(mnist):
    x = tf.placeholder(tf.float32,shape=[None,INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,shape=[None,OUTPUT_NODE],name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x,train=True,regularizer=regularizer)
    global_step = tf.Variable(0,trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    variable_averages_op  = variable_averages.apply(
        tf.trainable_variables())
    # average_y = inference(
    #     x,variable_averages,weights1,biases1,weights2,biases2
    # )

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #计算L2正则项
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    train_op = tf.group(train_step,variable_averages_op)
    # with tf.control_dependencies([train_step]):
    #     train_op = tf.no_op(name='train')
    # correct_predict = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    # accurate = tf.reduce_mean(tf.cast(correct_predict,tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # validate_feed = {x:mnist.validation.images,
        #                  y_:mnist.validation.labels}
        # test_feed = {x:mnist.test.images,
        #              y_:mnist.test.labels}

        for i in range(TRAINING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            # xs = np.reshape(xs,[BATCH_SIZE,28,28,1])
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i%1000==0:
                # validate_acc = sess.run(accurate,feed_dict=validate_feed)
                print('After {} training steps, loss on train'\
                    ' is {}'.format(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets('./data',one_hot=True)
    train(mnist)

if __name__=='__main__':
    tf.app.run()




