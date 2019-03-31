#coding=utf8

import tensorflow as tf
v = tf.Variable(0,dtype=tf.float32,name='v')

# for variables in tf.global_variables():
#     print(variables)

ema = tf.train.ExponentialMovingAverage(0.99)
# maintain_average_op = ema.apply(tf.global_variables())
print(ema.variables_to_restore())
# for variables in tf.global_variables():
#     print(variables)
#
saver = tf.train.Saver(ema.variables_to_restore())
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     print(sess.run(tf.assign(v,10)))
#     print(sess.run(maintain_average_op))
#     saver.save(sess,'model/model.ckpt')
#     # print(v)
#     print(sess.run([v,ema.average(v)]))
# a = tf.Variable(tf.constant([1],shape=[1]),name='v1')
# b = tf.Variable(tf.constant([2],shape=[1]),name='v2')
#
# c = a + b
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     print(sess.run(c))
#     saver.save(sess,'./model/model.ckpt')
# saver = tf.train.Saver({'v1':a})
# # saver = tf.train.import_meta_graph('model/model.ckpt.meta')
with tf.Session() as sess:
    saver.restore(sess,'model/model.ckpt')
    print(sess.run(v))
#     # print([n.name for n in tf.get_default_graph().as_graph_def().node])
#     print(sess.run(tf.get_default_graph().get_tensor_by_name('v2:0')))