import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a,b)
multi = tf.multiply(a, b)
with tf.Session() as sess:
    print(sess.run(add, feed_dict={a:2, b:2}))
    print(sess.run(multi, feed_dict={a:2, b:3}))