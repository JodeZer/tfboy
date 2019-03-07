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
        
        self.l1 = self._convLayer(self.inputs, [3,3,1, 64])
        
        self.maxp1 = tf.nn.max_pool(self.l1, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        
        self.l2 = self._convLayer(self.maxp1, [3, 3, 64, 128])
        
        self.maxp2 = tf.nn.max_pool(self.l2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        
        self.l3 = self._convLayer(self.maxp2, [3, 3, 128, 256])
        
        self.l4 = self._convLayer(self.l3, [3, 3, 256, 256])
        
        self.maxp3 = tf.nn.max_pool(self.l4, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        
        self.l5 = self._convLayer(self.maxp3, [3, 3, 256, 512])
        
        self.l6 = self._convLayer(self.l5, [3, 3, 512, 512])
        
        self.maxp4 = tf.nn.max_pool(self.l6, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        
        self.l7 = self._convLayer(self.maxp4, [3, 3, 512, 512])
        
        self.l8 = self._convLayer(self.l7, [3, 3, 512, 512])
        
        self.maxp5 = tf.nn.max_pool(self.l8, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        

         # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten( self.maxp5)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 4096)

        fc1 = tf.layers.dense(fc1, 4096)

        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=0.25, training=True)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, 10)
        # flatten layer
        # self.fcInput = tf.reshape(self.maxp5, [-1, 7*7*512])
        
        # # fully connected layer
        # self.fcl1 = tf.contrib.layers.fully_connected(self.fcInput, 4096)
        
        # self.fcl2 = tf.contrib.layers.fully_connected(self.fcl1, 4096)
        
        # # output layer
        # out = tf.contrib.layers.fully_connected(self.fcl2, 10)
        
        
        ## ?? weight decay?
        # softmaxlayer and loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.outputs, logits=out))

        #opt = tf.train.MomentumOptimizer(0.01,0.9)
        opt = tf.train.GradientDescentOptimizer(0.001)
        self.train = opt.minimize(self.loss)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.outputs, 1), tf.argmax(out, 1)), tf.float32))

        self.evaluate = tf.argmax(out, 1)
                
    def fit(self,sess, X, y):
        sess.run(self.train, feed_dict={self.inputs:X, self.outputs:y})

    def call_accuracy(self,sess, X, y):
        return sess.run(self.accuracy, feed_dict={self.inputs:X, self.outputs:y})
    
    def call_evaluate(self,sess, X):
        return sess.run(self.evaluate, feed_dict={self.inputs:X})

    def call_loss(self,sess, X, y):
        return sess.run(self.loss, feed_dict={self.inputs:X, self.outputs:y})

# 224 x 224 RGB

class NaiveCNN():
    ## 3*3 conv
    ## maxp
    ## 3*3 conv
    ## maxp
    ## fc
    ## fc
    def __init__(self):
        self._initNet()
        
    def _convLayer(self, inputs, filterSize):
        w, b = tf.Variable(tf.random_normal(filterSize)), tf.Variable(tf.random_normal([filterSize[-1]]))
        l = tf.nn.conv2d(inputs, w,[1,1,1,1], padding = "SAME")
        z = tf.nn.bias_add(l, b)
        return tf.nn.relu(z)

    def _initNet(self):
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
        self.outputs = tf.placeholder(dtype=tf.float32, shape = [None, 10])
        
        conv1 = tf.layers.conv2d(self.inputs, 32, 3, activation=tf.nn.relu)
        
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        
        conv1 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        
        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv1)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=0.25, training=True)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, 10)
        
        ## ?? weight decay?
        # softmaxlayer and loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.outputs, logits=out))

        #opt = tf.train.MomentumOptimizer(0.01,0.9)
        opt = tf.train.GradientDescentOptimizer(0.1)

        self.train = opt.minimize(self.loss)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.outputs, 1), tf.argmax(out, 1)), tf.float32))

        self.evaluate = tf.argmax(out, 1)
                
    def fit(self,sess, X, y):
        sess.run(self.train, feed_dict={self.inputs:X, self.outputs:y})

    def call_accuracy(self,sess, X, y):
        return sess.run(self.accuracy, feed_dict={self.inputs:X, self.outputs:y})
    
    def call_evaluate(self,sess, X):
        return sess.run(self.evaluate, feed_dict={self.inputs:X})

    def call_loss(self,sess, X, y):
        return sess.run(self.loss, feed_dict={self.inputs:X, self.outputs:y})

if __name__ == "__main__":
    vgg = VGG11()
# run test
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        vgg.fit(sess, one, train_labels[0:2])
        res = vgg.call_evaluate(sess, one)