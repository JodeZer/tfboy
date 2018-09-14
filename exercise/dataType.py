import tensorflow as tf
import numpy as np

a= tf.Variable(tf.zeros([3,4], tf.int32))
b= tf.Variable(tf.zeros([3,4], tf.int32))

addNode = tf.add(a, b)
addNode2 = tf.add(a, b)

with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)
    print(sess.run(addNode))
    print(sess.run(addNode2))