import tensorflow as tf 
# 使用交互式会话方便展示
#sess = tf.InteractiveSession()
#tf.InteractiveSession()
sess = tf.Session()

x = tf.constant([[2, 5, 3, -5], 
                 [0, 3,-2,  5], 
                 [4, 3, 5,  3], 
                 [6, 1, 4,  0]]) 
y = tf.constant([[4, -7, 4, -3, 4], 
                 [6, 4,-7,  4, 7], 
                 [2, 3, 2,  1, 4], 
                 [1, 5, 5,  5, 2]])

floatx = tf.constant([[2., 5., 3., -5.], 
                      [0., 3.,-2.,  5.], 
                      [4., 3., 5.,  3.], 
                      [6., 1., 4.,  0.]]) 
print(type(x))
print(type(tf.transpose(x)))
print (type(tf.transpose(x).eval(session=sess)))
print (tf.matmul(x, y).eval())
# print (tf.matrix_determinant(tf.to_float(x)).eval())
# print (tf.matrix_inverse(tf.to_float(x)).eval())
# print (tf.matrix_solve(tf.to_float(x), [[1],[1],[1],[1]]).eval())

sess.close()