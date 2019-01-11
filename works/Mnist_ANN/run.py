import tensorflow as tf 
import tensorflow.examples.tutorials.mnist as mnist

DataSize = 512
Batch = 32
# 1.prepare data
mnist_data = mnist.input_data.read_data_sets("MNIST_data", one_hot=True)

class DataCollection:
    def __init__(self, train, test):
        self.train = train
        self.test = test
        pass
    
data = DataCollection((mnist_data.train.images[:DataSize],mnist_data.train.labels[:DataSize]), (mnist_data.train.images[DataSize:DataSize+50], mnist_data.train.labels[DataSize:DataSize+50]))
    
modelInput = tf.placeholder(dtype=tf.float32,
    shape=(None, data.train[0].shape[1]))
modelOutput = tf.placeholder(dtype=tf.float32, shape=(None, data.train[1].shape[1]))

# 2.define graph

#graph = tf.Graph()
#with graph.as_default():
    # hidden layer
l1 = tf.layers.dense(modelInput, 16, activation=tf.nn.relu)
# output layer
l2 = tf.layers.dense(l1, 10)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=modelOutput, logits=l2))

evaluation = tf.equal(tf.argmax(modelOutput, 1), tf.argmax(l2, 1))

opt = tf.train.GradientDescentOptimizer(learning_rate=0.5)

train_op = opt.minimize(loss)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

accuracy = tf.reduce_mean(tf.cast(evaluation, tf.float32))
# 3.run session/training
with tf.Session() as sess:
    sess.run(init_op) 
    # SGD
    for _ in range(10):
        for i in range(0, DataSize//Batch):
            sess.run([train_op], {modelInput: data.train[0][i:Batch*(1+i)], modelOutput: data.train[1][i:Batch*(1+i)]})
    #GD
#    for i in range(0, 512):
#        sess.run([train_op], {modelInput: data.train[0], modelOutput: data.train[1]})
#    
    print(sess.run([accuracy], {modelInput:data.train[0], modelOutput:data.train[1]}))

    print(sess.run([accuracy], {modelInput:data.test[0], modelOutput:data.test[1]}))

# 4. save model 
    
    
