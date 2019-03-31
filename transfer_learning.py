#coding=utf8

import tensorflow as tf
import numpy as np
import os.path
import glob
from tensorflow.python.platform import gfile
slim = tf.contrib.slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
#处理好的数据
INPUT_DATA = 'data/flower_photos/flower_processed_data.npy'
TRAIN_FILE = '/model/save_model'
CKPT_FILE = '/model/inception_v3.ckpt'

LEARNING_RATE = 0.0001
STEPS = 300
BATCH = 32
N_CLASS = 5

CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
TRAINABLE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'

def get_tune_variables():
    exclusions = [scope.strip() for scope in \
                  CHECKPOINT_EXCLUDE_SCOPES.split(',')]
    variable_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startwith(exclusion):
                excluded = True
                break
        if not excluded:
            variable_to_restore.append(var)
    return variable_to_restore

def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,scope
        )
        variables_to_train.extend(variables)
    return variables_to_train

def main():
    processed_data = np.load(INPUT_DATA)
    training_images = processed_data[0]
    n_training_example = len(training_images)
    training_labels = processed_data[1]
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    testing_images = processed_data[4]
    testing_labels = processed_data[5]

    print('{} training examples, {} validation '
          'examples and {} testing examples.'.format(n_training_example,len(validation_labels),len(testing_labels)))
    images = tf.placeholder(
        tf.float32,[None,299,299,3],name='input_images')
    labels = tf.placeholder(tf.int64,[None],name='labels')

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(
            images,num_classes=N_CLASS
        )
    trainable_variables = get_trainable_variables()
    tf.losses.softmax_cross_entropy(
        tf.one_hot(labels,N_CLASS),logits,weights=1.0
    )
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())
    #计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits,1),labels)
        evaluation_step = tf.reduce_mean(
            tf.cast(correct_prediction,tf.float32)
        )
    #定义加载模型的函数
    load_fn = slim.assign_from_checkpoint_fn(
        CKPT_FILE,
        get_tune_variables(),
        ignore_missing_vars=True
    )
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print('Loading tuned variables from {}'.format(CKPT_FILE))
        load_fn(sess)

        start = 0
        end = BATCH
        for i in range(STEPS):
            sess.run(train_step,feed_dict={
                images:training_images[start:end],
                labels:training_labels[start,end]
            })
            if i%30 == 0 or i+1==STEPS:
                saver.save(sess,TRAIN_FILE,global_step=i)
                validation_accuracy = sess.run(evaluation_step,feed_dict={
                    images:validation_images, labels:validation_labels
                })
                print('Step {} validation accuracy = {}'.format(i,validation_accuracy*100))
                start = end
                if start == n_training_example:
                    start = 0
                end = end+BATCH
                if end>n_training_example:
                    end = n_training_example
        test_accuracy = sess.run(evaluation_step,feed_dict={
            images:testing_images,labels:testing_labels
        })
        print('Final test accuracy {}'.format(test_accuracy*100))

if __name__ == '__main__':
    tf.app.run()