"""
Created on 2018/10/27
@author: AlanYx
"""
import tensorflow as tf

if __name__ == '__main__':
    #####################################################
    """
    tensorflow设置日志级别
    """
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    hello = tf.constant('Hello, TensorFlow!')
    with tf.Session() as sess:
        print(sess.run(hello))
    #####################################################
