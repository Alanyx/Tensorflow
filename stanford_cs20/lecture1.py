"""
Created on 2018/7/25
@author: AlanYx
"""
import tensorflow as tf

if __name__ == '__main__':
    ##########################################################
    # a = tf.add(3,5)
    # 不会获得a的值
    # print(a)
    ##########################################################

    ##########################################################
    # ##  How to get the value of a
    # # 方式1
    # a = tf.add(3, 5)
    # sess = tf.Session()
    # print(sess.run(a))
    # sess.close()
    #
    # # 方式2(推荐)
    # a = tf.add(3, 5)
    # sess = tf.Session()
    # with tf.Session() as sess:
    #     print(sess.run(a))
    ##########################################################

    ##########################################################
    # # 子图
    # x = 2
    # y = 3
    # add_op = tf.add(x,y)
    # mul_op = tf.multiply(x,y)
    # useless = tf.multiply(x,add_op)
    # pow_op = tf.pow(add_op,mul_op)
    # with tf.Session() as sess:
    #     z = sess.run(pow_op)
    #     print("z:",z)
    ##########################################################


    ##########################################################
    # 分布式计算
    # create a graph
    with tf.device("/gpu:2"):
        a = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0],name="a")
        b = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0],name="b")
        c = tf.multiply(a,b)

    # create a session with log_device_placement set to True
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=True))
    print(sess.run(c))
    ##########################################################


    ##########################################################
    # tf.Graph()
    g = tf.Graph()
    with g.as_default():
        x = tf.add(3,5)
    sess = tf.Session(graph=g) # 指定图
    with tf.Session() as sess:
        sess.run(x)
    ##########################################################


    ##########################################################
    #错误示范:创建多个图(方式不够好,最好不要超过一张图)
    g1 = tf.get_default_graph()
    g2 = tf.Graph()

    # 给默认图增加操作
    with g1.as_default():
        a = tf.constant(3)

    # 给用户创建图增加操作
    with g2.as_default():
        b =tf.constant(2)
    ##########################################################



