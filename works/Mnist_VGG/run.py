import utils
import model
import tensorflow.examples.tutorials.mnist as mnist
import tensorflow as tf

# define
SampleSize = 1000 #  all data
batchSize = 16
# prepare data
mnist_data = mnist.input_data.read_data_sets("MNIST_data", one_hot=True)

train_imageData = mnist_data.train.images.reshape(mnist_data.train.images.shape[0], 28, 28)

print("start expand data")
train_imageData = utils.BatchImageExpand(train_imageData[:SampleSize], 224)
print("expand data done")
train_labels = mnist_data.train.labels[:SampleSize]
train_imageData = train_imageData.reshape(-1, 224,224,1)

dataset = utils.DataSet(train_imageData, train_labels, testRate= 0.05, batchSize=batchSize)
learnCurve = utils.learnCurve()
vgg = model.VGG11()

#data = dataset.getNextBatch()

with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    for ep in range(1, 11):
        data = dataset.getNextBatch()
        while data != None:
            print("start epoch:{}, iter:{}".format(ep, dataset.iter))
            vgg.fit(sess, data[0], data[1])
            batchAcc = vgg.call_accuracy(sess, data[0], data[1])
            print("fit epoch:{}, iter:{}, batchAcc:{}".format(ep, dataset.iter, batchAcc))
            #trainAcc = vgg.call_accuracy(sess, dataset.trainX, dataset.trainY)
            testAcc = vgg.call_accuracy(sess, dataset.testX, dataset.testY)
            print("done epoch:{}, iter:{},{},{} ".format(ep, dataset.iter, batchAcc,testAcc))
            learnCurve.append(0, batchAcc, testAcc)
            data = dataset.getNextBatch()
        dataset.nextEpoch()


#learnCurve.plot()



