"""
Created on 2018/10/27
@author: AlanYx
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    ##########################################################
    # # log日志设置 import os
    # a = tf.constant(2)
    # b = tf.constant(3)
    # x = tf.add(a,b)
    #
    # with tf.Session() as sess:
    #     print(sess.run(x))
    ##########################################################

    ##########################################################
    # # tensorboard
    # a = tf.constant(2,name="a")
    # b = tf.constant(3,name="b")
    # x = tf.add(a, b,name="add")
    #
    # writer = tf.summary.FileWriter("./graphs",tf.get_default_graph())
    # with tf.Session() as sess:
    #     # writer = tf.summary.FileWriter("./graphs", tf.get_default_graph())
    #     print(sess.run(x))
    # # close writer when you're done using it
    # writer.close()
    #
    # """
    # 可视化操作:
    # python3 lecture2.py
    # tensorboard --logdir="./graphs" -- port 6006  (其他端口也ok)
    #
    # 然后打开浏览器，查看:http://localhost:6006/
    # """
    ##########################################################

    ##########################################################
    # # Constant
    # a = tf.constant([2,2],name="a")
    # b = tf.constant([[0,1],[2,3]],name="b")
    # x = tf.multiply(a, b,name="mul")
    #
    # with tf.Session() as sess:
    #     print(sess.run(x))

    ##########################################################
    # # 指定特别的值
    # input_tensor = tf.constant([[0, 1], [2, 3], [4, 5]])
    # a = tf.zeros([2, 3], dtype=tf.int32, name="a")
    # b = tf.zeros_like(input_tensor, name="b")
    #
    # c = tf.ones([2, 3], dtype=tf.int32, name="c")
    # d = tf.ones_like(input_tensor, name="d")
    #
    # e = tf.fill([2,3],8,name="e")
    #
    # with tf.Session() as sess:
    #     print("a:", sess.run(a))
    #     print("b:", sess.run(b))
    #     print("c:", sess.run(c))
    #     print("d:", sess.run(d))
    #     print("e:", sess.run(e))
    ##########################################################

    ##########################################################
    # # Constants as sequences(函数说明可参见function.py)
    # #tf.lin_space(start, stop, num, name=None)
    # a = tf.lin_space(10.0,13.0,4)
    #
    # #tf.range(start, limit=None, delta=1, dtype=None, name='range')
    # # 注意与numpy range()不相同，tf.range()是不可以循环迭代的！！！
    # b =tf.range(3,18,3)
    #
    # with tf.Session() as sess:
    #     print("a:", sess.run(a))
    #     print("b:", sess.run(b))
    ##########################################################

    ##########################################################
    # Randomly Generated Constants(函数说明可参见function.py)
    # tf.random_normal
    # tf.truncated_normal
    # tf.random_uniform
    # tf.random_shuffle
    # tf.random_crop
    # tf.multinomial
    # tf.random_gamma
    ##########################################################

    ##########################################################
    # # 算术操作
    # # tf.abs
    # # tf.negative
    # # tf.sign
    # # tf.reciprocal   # 求倒数
    # # tf.square
    # # tf.round        # 求各元素各自距离最近的整数；若在中间，则取偶数值
    # # tf.sqrt
    # # tf.rsqrt
    # # tf.pow
    # # tf.exp
    # a = tf.constant([2, 2], name="a")
    # b = tf.constant([[0, 1], [2, 3]], name="b")
    # with tf.Session() as sess:
    #     print(sess.run(tf.div(b, a)))  # 对应元素 相除
    #     print(sess.run(tf.divide(b, a)))  # 对应元素 相除
    #     print(sess.run(tf.truediv(b, a)))  # 对应元素 相除
    #     print(sess.run(tf.floordiv(b, a)))  # 对应元素 地板除
    #     # print(sess.run(tf.realdiv(b,a)))    # Error: only works for real value
    #     print(sess.run(tf.truncatediv(b, a)))  # 对应元素 截断除 取余
    #     print(sess.run(tf.floor_div(b, a)))  # 对应元素 地板除
    ##########################################################

    ##########################################################
    # t_0 = 19
    # a = tf.zeros_like(t_0)
    # b = tf.ones_like(t_0)
    #
    # t_1 = [b"apple", b"peach", b"graph"]
    # c = tf.zeros_like(t_1)
    # # d = tf.ones_like(t_1)    # Expected string, got 1 of type 'int' instead.
    #
    # t_2 = [[True, False, False],
    #        [False, False, True],
    #        [False, True, False]]
    # e = tf.zeros_like(t_2)
    # f = tf.ones_like(t_2)
    #
    # with tf.Session() as sess:
    #     print(sess.run(a))
    #     print(sess.run(b))
    #     print(sess.run(c))
    #     # print(sess.run(d))
    #     print(sess.run(e))
    #     print(sess.run(f))
    ##########################################################

    ##########################################################
    # #  TF vs NP(numpy) Data Types
    # a = (tf.int32 ==np.int32)  # =>True
    #
    # b = tf.ones([2,2],dtype=np.float32)
    #
    # with tf.Session() as sess:
    #     print("a:",sess.run(a))
    #     # print(type(b))
    #     # b2 = sess.run(b)
    #     # print(type(b2))

    ##########################################################

    ##########################################################
    # 建议:尽可能地使用tf的dtype
    # 1.tf不得不转换python类型
    # 2。numpy与gpu不兼容
    ##########################################################

    ##########################################################
    # #  Print out the graph def
    # # 常量存储在图定义中(会占用内存)
    # my_const = tf.constant([1.0,2.0],name="my_const")
    # with tf.Session() as sess:
    #     print(sess.graph.as_graph_def())
    ##########################################################

    ##########################################################
    # #  Variables
    # # 1.用tf.Variables创建变量
    # s1 = tf.Variable(2,name="scalar1")
    # m1 = tf.Variable([[0,1],[2,3]],name="matrix1")
    # W1 = tf.Variable(tf.zeros([784,10]),name="big_matrix1")
    #
    # # s1.initializer # init op
    # # s1.value()     # read op
    # # s1.assign()    # write op
    # # s1.assign_add() # and more op
    #
    # # 2.用tf.get_variable创建变量【推荐使用】（因为tf.Variables是一个有很多操作的类，而tf.constant是一个操作）
    # s2 = tf.get_variable("scalar2",initializer=tf.constant(2))
    # m2 = tf.get_variable("matrix2",initializer=tf.constant([[0,1],[2,3]]))
    # W2 = tf.get_variable("big_matrix2",shape=(784,10),initializer=tf.zeros_initializer())
    #
    # with tf.Session() as sess:
    #     # print(sess.run(W2)) # FailedPreconditionError: Attempting to use uninitialized value big_matrix2 必须先初始化变量
    #     # 方式一:The easiest way is initializing all variables at once
    #     sess.run(tf.global_variables_initializer())
    #     print("s2:",sess.run(s2))
    #     print("m2:",sess.run(m2))
    #     print("W2:",sess.run(W2))
    #     # 方式二:Initialize only a subset of variables
    #     # sess.run(tf.variables_initializer([s2,m2,W2]))
    #     # 方式三:Initialize a single variable
    #     # sess.run(W1.initializer)
    ##########################################################

    ##########################################################
    # # Eval() a variable (评估变量的值，就是计算)
    # #  W is a random 700 x 100 variable object
    # W = tf.Variable(tf.truncated_normal([700,10]))
    #
    # with tf.Session() as sess:
    #     sess.run(W.initializer)
    #     print(W.eval())         # Similar to print(sess.run(W))
    ##########################################################

    ##########################################################
    # # tf.Variable.assign()
    # W = tf.Variable(10)
    # W.assign(100)
    # with tf.Session() as sess:
    #     sess.run(W.initializer)
    #     print("1:",W.eval())            # 10，不是100，W.assign(100)操作需要在一个会话中执行才会产生效果
    #
    # W = tf.Variable(10)
    # assign_op = W.assign(100)
    # with tf.Session() as sess:
    #     sess.run(W.initializer)
    #     sess.run(assign_op)
    #     print("2:",W.eval())             # 100
    #
    #
    # # 更多细节
    # my_var = tf.Variable(2)
    #
    # my_var_two_times = my_var.assign(2 * my_var)
    # with tf.Session() as sess:
    #     sess.run(my_var.initializer)
    #     print("1:",sess.run(my_var_two_times))      # 4
    #     print("2:", sess.run(my_var_two_times))     # 8
    #     print("3:", sess.run(my_var_two_times))     # 16
    ##########################################################

    ##########################################################
    # assign_add() and assign_sub()
    # my_var = tf.Variable(10)
    # with tf.Session() as sess:
    #     sess.run(my_var.initializer)
    #
    #     # increment by 10
    #     print("assign_add:",sess.run(my_var.assign_add(10)))    # 20
    #
    #     # decrement by 2
    #     print("assign_sub:",sess.run(my_var.assign_sub(2)))  # 18
    ##########################################################

    ##########################################################
    # #  Each session maintains its own copy of variables
    # W = tf.Variable(10)
    #
    # sess1 = tf.Session()
    # sess2 = tf.Session()
    #
    # sess1.run(W.initializer)
    # sess2.run(W.initializer)
    #
    # print("sess1:", sess1.run(W.assign_add(10)))  # 20
    # print("sess2:", sess2.run(W.assign_sub(2)))   # 8
    #
    # print("sess1:",sess1.run(W.assign_add(100)))  # 120
    # print("sess2:",sess2.run(W.assign_sub(7)))    # 1
    #
    # sess1.close()
    # sess2.close()
    ##########################################################

    ##########################################################
    # #  Control Dependencies(控制依赖)
    # # 你的图有5个ops(操作):a,b,c,d,e
    # g = tf.get_default_graph()
    # a = []
    # b = []
    # c = []
    # with g.control_dependencies([a,b,c]):
    #     # 变量g依赖a,b,c之后，d,e只有在a,b,c执行完成之后才能运行
    #     d =...
    #     e =...
    ##########################################################

    ##########################################################
    # # placeholders
    # # create a placeholder for a vector of 3 elements ,type tf.float32
    # a = tf.placeholder(tf.float32,shape=[3])
    #
    # b = tf.constant([5,5,5],tf.float32)
    #
    # # 当你想要一个常量或者变量时，使用这个占位符
    # c = a + b
    #
    # with tf.Session() as sess:
    #     # print(sess.run(c))          # You must feed a value for placeholder tensor 'Placeholder' with dtype float and shape [3]
    #
    #     # 用一个字典给占位符赋值(注意下面的a是一个键，不是一个字符"a")
    #     print(sess.run(c,feed_dict={a:[1,2,3]}))
    ##########################################################

    ##########################################################
    # # 将值喂给tf操作
    # a = tf.add(2,5)
    # b = tf.multiply(a,3)
    #
    # with tf.Session() as sess:
    #     # 当a被给定时会覆盖上面的a值
    #     print(sess.run(b))
    #     print(sess.run(b, feed_dict={a:15}))
    ##########################################################

    ##########################################################
    # normal loading  (节点”Add”在图定义中只添加一次)
    x = tf.Variable(10, name="x")
    y = tf.Variable(20, name="y")
    z = tf.add(x, y)  # 在执行图之前创建这个节点

    writer = tf.summary.FileWriter("./graphs/normal_loading", tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(10):
            print("normal loading:", sess.run(z))
    writer.close()

    # lazy loading (节点”Add”在图定义中添加10次(与计算z的次数一样多))【非常消耗资源，不推荐使用】
    x2 = tf.Variable(10, name="x2")
    y2 = tf.Variable(20, name="y2")

    writer = tf.summary.FileWriter("./graphs/lazy_loading", tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(10):
            print("lazy loading:", sess.run(tf.add(x2, y2)))  # someone decide to be clever to save one line of code
    writer.close()
    ##########################################################
