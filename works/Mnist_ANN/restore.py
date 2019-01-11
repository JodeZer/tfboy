import tensorflow.examples.tutorials.mnist as input_data
from PIL import Image
import os
import numpy as np 


mnist = input_data.input_data.read_data_sets("MNIST_data/", one_hot=True)

save_dir = 'MNIST_data/raw/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

# 保存前20张图片
for i, image in enumerate(mnist.train.images):
    if i > 99:
        break
    # 请注意，mnist.train.images[i, :]就表示第i张图片（序号从0开始）
    image_array = image
    # TensorFlow中的MNIST图片是一个784维的向量，我们重新把它还原为28x28维的图像。
    #im_data = np.array(np.reshape(mnist.train.images[1], (28, 28)) * 255, dtype=np.int8)
    image_array = np.array(np.abs(image_array.reshape(28, 28)*255-255), dtype=np.int8)
    # 保存文件的格式为 mnist_train_0.jpg, mnist_train_1.jpg, ... ,mnist_train_19.jpg
    filename = save_dir + 'mnist_train_%d.jpg' % i
    # 将image_array保存为图片
    # 先用scipy.misc.toimage转换为图像，再调用save直接保存。
    Image.fromarray(image_array, mode="L").save(filename)

def restoreOne(image, filename):
    image_array = np.array(image.reshape(28, 28)*255, dtype=np.int8)
    # 保存文件的格式为 mnist_train_0.jpg, mnist_train_1.jpg, ... ,mnist_train_19.jpg
    filename = save_dir + 'mnist_%s.jpg' % filename
    # 将image_array保存为图片
    # 先用scipy.misc.toimage转换为图像，再调用save直接保存。
    Image.fromarray(image_array, mode="L").save(filename)