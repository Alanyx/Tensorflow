"""
Created on 2018/7/31
@author: AlanYx
"""

import tensorflow as tf

if __name__ == '__main__':
    a = tf.constant([1.0,2.0],name="a")
    b = tf.constant([2.0,3.0],name="b")
    result = tf.add(a,b,name="add")
    print(result)
    # =========================================================================

    # 通过a.graph可以查看张量所属的计算图。由于没有特意指定，
    # 所以这个计算图应该等于当前默认的计算图。下面输出为True
    # print(a.graph is tf.get_default_graph())
    # =========================================================================

    # g1 = tf.Graph()
    # with g1.as_default():
    #     # 在计算图g1中定义变量"v"，并设置初始值为0
    #     v = tf.get_variable(
    #         "v", shape=[1], initializer=tf.zeros_initializer)
    #
    # g2 = tf.Graph()
    # with g2.as_default():
    #     # 在计算图g2中定义变量"v"，并设置初始值为1
    #     v = tf.get_variable(
    #         "v", shape=[1], initializer=tf.ones_initializer)
    #
    # # 在计算图g1中读取变量"v"的取值
    # with tf.Session(graph=g1) as sess:
    #     tf.global_variables_initializer().run()
    #     with tf.variable_scope("",reuse=True):
    #         # 在计算图g1中，变量"v"的取值应为0，所以下面输出[0.]
    #         print(sess.run(tf.get_variable("v")))
    #
    # # 在计算图g2中读取变量"v"的取值
    # with tf.Session(graph=g2) as sess:
    #     tf.global_variables_initializer().run()
    #     with tf.variable_scope("", reuse=True):
    #         # 在计算图g1中，变量"v"的取值应为1，所以下面输出[1.]
    #         print(sess.run(tf.get_variable("v")))
    # =========================================================================






















