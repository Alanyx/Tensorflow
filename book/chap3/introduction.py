"""
Created on 2018/7/31
@author: AlanYx
"""

import tensorflow as tf

if __name__ == '__main__':
    ### 3.1计算图
    # a = tf.constant([1.0,2.0],name="a")
    # b = tf.constant([2.0,3.0],name="b")
    # result = tf.add(a,b,name="add")
    # # print(result)
    # with tf.Session() as sess:
    #     print(sess.run(result))
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

    ### 3.2 Tensorflow数据模型---张量
    # # 使用张量记录中间结果
    # a = tf.constant([1.0,2.0],name="a")
    # b = tf.constant([2.0,3.0],name="b")
    # result = a + b
    #
    # # 直接计算向量的和，这样可读性比较差
    # result = tf.constant([1.0,2.0],name="a")+tf.constant([2.0,3.0],name="b")
    # # =========================================================================

    ### 3.3 TensorFlow运行模型——会话
    # # 创建一个会话
    # sess=tf.Session()
    # # 使用创建好的会话得到运算结果
    # sess.run(...)
    # # 关闭会话释放资源
    # sess.close()
    #
    # # 创建一个会话,通过Python中的上下文管理器管理这个会话
    # With tf.Session() as sess:
    #     # 使用创建好的会话得到运算结果
    #     sess.run(...)
    # # 不需要调用sess.close()关闭会话
    # # 当上下文退出时会话关闭和资源释放自动完成
    # # =========================================================================

    # sess =tf.Session()
    # with sess.as_default():
    #     print(result.eval())
    #
    # sess = tf.Session()
    # print(sess.run(result))
    # print(result.eval(session=sess))
    # # =========================================================================

    # sess = tf.InteractiveSession()
    # print(result.eval())
    # sess.close()
    # # =========================================================================

    config = tf.ConfigProto(allow_soft_placement=True,
                           log_device_placement=True)
    sess1 = tf.InteractiveSession(config=config)
    sess2 = tf.Session(config=config)
























