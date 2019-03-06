# prepare data

import numpy as np


# vgg11 (A)

import tensorflow as tf

class VGG11():
    def __init__(self):
#        self.loss
#        self.evaluate
#        self.train
        self._initNet()
        
    def _convLayer(self, inputs, filterSize):
        w, b = tf.Variable(tf.random_normal(filterSize)), tf.Variable(tf.random_normal([filterSize[-1]]))
        l = tf.nn.conv2d(inputs, w,[1,1,1,1], padding = "SAME")
        z = tf.nn.bias_add(l, b)
        return tf.nn.relu(z)

    def _initNet(self):
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224,1])
        self.outputs = tf.placeholder(dtype=tf.float32, shape = [None, 10])
        
        l1 = self._convLayer(self.inputs, [3,3,1, 64])
        
        maxp1 = tf.nn.max_pool(l1, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        
        l2 = self._convLayer(maxp1, [3, 3, 64, 128])
        
        maxp2 = tf.nn.max_pool(l2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        
        l3 = self._convLayer(maxp2, [3, 3, 128, 256])
        
        l4 = self._convLayer(l3, [3, 3, 256, 256])
        
        maxp3 = tf.nn.max_pool(l4, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        
        l5 = self._convLayer(maxp3, [3, 3, 256, 512])
        
        l6 = self._convLayer(l5, [3, 3, 512, 512])
        
        maxp4 = tf.nn.max_pool(l6, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        
        l7 = self._convLayer(maxp4, [3, 3, 512, 512])
        
        l8 = self._convLayer(l7, [3, 3, 512, 512])
        
        maxp5 = tf.nn.max_pool(l8, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        
        # flatten layer
        fcInput = tf.reshape(maxp5, [-1, 7*7*512])
        
        # fully connected layer
        fcl1 = tf.contrib.layers.fully_connected(fcInput, 4096)
        
        fcl2 = tf.contrib.layers.fully_connected(fcl1, 4096)
        
        # output layer
        fcl3 = tf.contrib.layers.fully_connected(fcl2, 10)
        
        ## ?? weight decay?
        # softmaxlayer and loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.outputs, logits=fcl3))

        self.train = tf.train.MomentumOptimizer(0.01,0.9).minimize(self.loss)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.outputs, 1), tf.argmax(fcl3, 1)), tf.float32))

        self.evaluate = tf.argmax(fcl3, 1)
                
    def fit(self,sess, X, y):
        sess.run(self.train, feed_dict={self.inputs:X, self.outputs:y})

    def call_accuracy(self,sess, X, y):
        return sess.run(self.accuracy, feed_dict={self.inputs:X, self.outputs:y})
    
    def call_evaluate(self,sess, X):
        return sess.run(self.evaluate, feed_dict={self.inputs:X})

    def call_loss(self,sess, X, y):
        return sess.run(self.loss, feed_dict={self.inputs:X, self.outputs:y})

# 224 x 224 RGB


if __name__ == "__main__":
    vgg = VGG11()
# run test
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        vgg.fit(sess, one, train_labels[0:2])
        res = vgg.call_evaluate(sess, one)