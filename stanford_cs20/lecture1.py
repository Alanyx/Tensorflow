"""
Created on 2018/7/25
@author: AlanYx
"""
import tensorflow as tf


if __name__ == '__main__':
    a = tf.add(3,5)
    # 不会获得a的值
    # print(a)

    ##  How to get the value of a
    # 方式1
    a = tf.add(3, 5)
    sess = tf.Session()
    print(sess.run(a))
    sess.close()

    # 方式2
    a = tf.add(3, 5)
    sess = tf.Session()
    with tf.Session() as sess:
        print(sess.run(a))