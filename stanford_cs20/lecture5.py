"""
Created on 2018/11/12
@author: yinxing
"""

import tensorflow as tf

if __name__ == '__main__':
    ##########################################################
    # tf中词嵌入
    # 方式1:one-hot
    # 方式2:word embedding
    # tf.nn.embedding_lookup(params=params,ids=ids,partition_strategy="mod",name=None,
    #                        validate_indices=True,max_norm=None)

    # NCE loss（噪声对比估计损失函数）
    tf.nn.nce_loss(weights=w, biases=b, labels=l, inputs=inputs, num_sampled=ns,
                   num_classes=nc, num_true=1, sampled_values=None,
                   remove_accidental_hits=False, partition_strategy="mod", name="nce_loss")
    ##########################################################

    ##########################################################
    # 结构化你的tensorflow模型
    """
    阶段一：构建图
        1.import data(with tf.data or placeholder)
        2.define the weight
        3.define the inference model
        4.define loss function
        5.define optimizer
    阶段二：计算
        1.初始化模型参数
        2.输入train data
        3.在train data上执行接口模型
        4.计算loss
        5.调节模型参数（重复2-5）
    """


    ##########################################################

    ##########################################################
    # 可复用的模型
    ##########################################################

    ##########################################################
    # 词嵌入可视化 04word2vec_visualize.py
    ##########################################################

    ##########################################################
    # 变量共享
    ##########################################################

    ##########################################################
    # Name scope(命名作用域)
    # with tf.name_scope(name_of_that_scope):
    #     # declare op_1
    #     # declare op_2
    #     # declare op_3
    #     # .....

    # with tf.name_scope("data"):
    #     iterator = dataset.make_initializable_iterator()
    #     center_words,target_words = iterator.get_next()
    #
    # with tf.name_scope("optimizer"):
    #     optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize()
    ##########################################################

    ##########################################################
    # variable scope(比上面这种更便于变量共享)
    def two_hidden_layers(x):
        assert x.shape.as_list == [200, 100]
        # 关键是使用tf.get_variable
        w1 = tf.get_variable(name="h1_weights", shape=[100, 50], initializer=tf.random_normal_initializer())
        b1 = tf.get_variable(name="h1_biases", shape=[50], initializer=tf.constant_initializer(0.0))
        h1 = tf.matmul(x,w1)+b1
        assert h1.shape.as_list()==[200,50]
        w2 = tf.get_variable(name="h2_weights", shape=[50, 10], initializer=tf.random_normal_initializer())
        b2 = tf.get_variable(name="h2_biases", shape=[10], initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(h1,w2)+b2
        return logits

    x1 = tf.get_variable(shape=[200,10],initializer=tf.random_normal_initializer())
    x2 = tf.get_variable(shape=[200,10],initializer=tf.random_normal_initializer())
    with tf.variable_scope("two_layers") as scope:
        logits1 = two_hidden_layers(x1)
        # 复用，不用创建两个除输入，其他都相同的图
        scope.reuse_variables()
        logits2 = two_hidden_layers(x2)


    # 进一步优化上面的代码
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################

    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################
