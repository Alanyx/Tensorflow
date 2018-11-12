"""
Created on 2018/11/2
@author: AlanYx
"""

# Eager execution
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
import numpy as np


if __name__ == '__main__':
    ##########################################################
    # # 静态图
    # x = tf.placeholder(tf.float32,shape=[1,1])
    # m = tf.matmul(x,x)
    #
    # print("m:",m)
    #
    # with tf.Session() as sess:
    #     m_out = sess.run(m,feed_dict={x:[[2.]]})
    #     print("m_out:",m_out)

    # # Eager execution
    # x2 = [[2.]]     # 不需要placeholders!
    # m2 = tf.matmul(x2,x2)
    #
    # print("m2:",m2) # 不需要sessions!
    ##########################################################

    ##########################################################
    # #  Lazy Loading(静态图)
    # x = tf.radom_uniform([2,2])
    #
    # with tf.Session() as sess:
    #     for i in range(x.shape[0]):
    #         for j in range(x.shape[1]):
    #             print(sess.run(x[i,j]))

    # # Eager execution
    # x = tf.random_uniform([2,2])
    #
    # for i in range(x.shape[0]):
    #     for j in range(x.shape[1]):
    #         print(x[i][j])
    ##########################################################


    ##########################################################
    # # tensors act like numpy arrays
    # x = tf.constant([1.0,2.0,3.0])
    #
    # assert type(x.numpy())==np.ndarray
    # squared = np.square(x)
    #
    # # tensors are iterable!
    # for i in x:
    #     print(i)
    ##########################################################


    ##########################################################
    # # gradients
    x = tfe.Variable(2.0)
    def loss(y):
        return (y-x**2)**2

    grad = tfe.implicit_gradients(loss)

    print(loss(7.))
    print(grad(loss(7.)))
    ##########################################################


    ##########################################################
    # 什么时候使用eager execution模式？
    # 1.你是一个研究者并想要灵活的框架
    # 2.开发一个新模型
    # 3.刚接触tf(允许你以python的方式探索tf的API)

    # 缺点:
    # 单一的GPU，resnet的基准性能与图相似
    # 在较小的操作上开销较大
    # 分布式支持还在开发中
    # 不兼容所有的tf的API
    ##########################################################








