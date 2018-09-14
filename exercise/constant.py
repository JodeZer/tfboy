import tensorflow as tf

def main():
    a = tf.constant(2)
    b = tf.constant(3)
    result = a * b
    with tf.Session() as sess:
        print(type(sess.run(result)))
        print(type(result))
        print(result)

if __name__ == "__main__":
    main()
