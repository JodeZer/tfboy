import tensorflow as tf


def main():
    v1 = tf.Variable(10)
    v2 = tf.Variable(5)
    result = v1 + v2
    result2 = v1 * v2
    print(type(v1))
    print(type(v2))

    with tf.Session() as sess:
        # variables must be initialized first.
        tf.global_variables_initializer().run(session=sess)
        # 用session跑node
        print(sess.run(v1+v2))  # Output => 15
        # 塞session进node跑
        print(result.eval(session=sess))
        # 这样也能跑，用的default session是哪个呢..
        print(result2.eval())
        # 这样也行
        print(tf.add(v1,v2).eval())
    # 这样不行，no default session
    #tf.add(v1,v2).eval()


if __name__ == "__main__":
    main()
