"""
Created on 2018/7/27
@author: AlanYx
"""
import tensorflow as tf

if __name__ == '__main__':
    ##### [Check installation and version]
    """
    查看安装tensorflow版本命令
    import tensorflow as tf
    tf.__version__ (两个下划线)
    """
    # =========================================================================


    ## [Computational Graph]
    """
    *****计算的3个步骤：
    (1)Build graph (tensors) using TensorFlow operations
    (2) feed data and run graph (operation) sess.run (op)
    (3) update variables in the graph (and return values)
    """
    # (1)
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)
    node3 = tf.add(node1, node2)


    # print("node1:", node1, "node2:", node2)
    # print("node3:", node3)
    # tf在一个会话中启动图，直接打印无法得到想要的结果

    # (2)(3)
    sess = tf.Session()
    # print("sess.run(node1,node2):", sess.run([node1, node2]))
    # print("sess.run(node3):", sess.run(node3))
    # =========================================================================


    ## [Placeholder:占位符/位置标志符]
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    # provide a shortcut(捷径) for tf.add(a,b)
    adder_node = a + b

    # print(sess.run(adder_node, feed_dict= {a:3,b:4.5}))
    # print(sess.run(adder_node, feed_dict= {a:[1,3],b:[2,4]}))
    # =========================================================================


    ## [Everything is Tensor]
    """
    3                           # a rank 0 tensor; this is a scalar with shape[]
    [1., 2., 3.]                # a rank 1 tensor; this is a vector with shape[3]
    [[1.,2.,3.],[4.,5.,6.]]     # a rank 2 tensor; a matrix with shape[2,3]
    [[[1.,2.,3.]],[[4.,5.,6.]]] # a rank 3 tensor with shape[2,1,3] 
    """
    t = tf.constant([1.,2.,3.])
    print(sess.run(t))
    # =========================================================================


    ## [Tensor Ranks, Shapes, and Types]
    # 见图片:Tensor Ranks, Shapes, and Types
    # =========================================================================


    ##### Machine Learning Basics
    ## [Linear Regression]
    # X and Y data
    x_train =[1,2,3]
    y_train =[1,2,3]

    W = tf.Variable(tf.random_normal([1]), name="weight")
    b = tf.Variable(tf.random_normal([1]), name="bais")

    # our hypotheis XW+b
    hypothesis = x_train * W + b

    # cost/loss function
    # t = [1.,2.,3.,4.]  tf.reduce_mean(t) ===>2.5
    cost = tf.reduce_mean(tf.square(hypothesis - y_train))

    # Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)

    # Launch the graph in a session.
    sess = tf.Session()
    # Initializes global variables in the graph.
    









