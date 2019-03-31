#coding=utf8
import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable(
        'v2',[1],initializer=tf.zeros_initializer
    )

g2 = tf.Graph()
with g2.as_default():
    with tf.variable_scope('foo'):
        v = tf.get_variable(
            'v1',[3],initializer=tf.ones_initializer
    )

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('',reuse=True):
        print(sess.run(tf.get_variable('v2')))
        print(tf.get_default_graph())
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('foo',reuse=True):
        print(sess.run(tf.get_variable('v1')))
        # v4 = tf.get_variable('v4',shape=[1])

with tf.variable_scope("test"):
    v2 = tf.get_variable("v", [1])
    print(tf.get_variable_scope().reuse,'reuse result')
    print(v2.name)
    # with tf.variable_scope('',reuse=True):
    #     v3 = tf.get_variable('foo/v4')

with tf.variable_scope('',reuse=True):
    v3 = tf.get_variable('test/v')
    print(v3 is v2)