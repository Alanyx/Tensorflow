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
    # # variable scope(比上面这种更便于变量共享)
    # def two_hidden_layers(x):
    #     assert x.shape.as_list == [200, 100]
    #     # 关键是使用tf.get_variable
    #     w1 = tf.get_variable(name="h1_weights", shape=[100, 50], initializer=tf.random_normal_initializer())
    #     b1 = tf.get_variable(name="h1_biases", shape=[50], initializer=tf.constant_initializer(0.0))
    #     h1 = tf.matmul(x,w1)+b1
    #     assert h1.shape.as_list()==[200,50]
    #     w2 = tf.get_variable(name="h2_weights", shape=[50, 10], initializer=tf.random_normal_initializer())
    #     b2 = tf.get_variable(name="h2_biases", shape=[10], initializer=tf.constant_initializer(0.0))
    #     logits = tf.matmul(h1,w2)+b2
    #     return logits
    #
    # x1 = tf.get_variable(shape=[200,10],initializer=tf.random_normal_initializer())
    # x2 = tf.get_variable(shape=[200,10],initializer=tf.random_normal_initializer())
    # with tf.variable_scope("two_layers") as scope:
    #     logits1 = two_hidden_layers(x1)
    #     # 复用，不用创建两个除输入，其他都相同的图
    #     scope.reuse_variables()
    #     logits2 = two_hidden_layers(x2)
    #
    #
    # # 进一步优化上面的代码
    # def fully_connected(x,output_dim,scope):
    #     with tf.variable_scope(scope,reuse=tf.AUTO_REUSE,) as scope:
    #         w = tf.get_variable("weight",[x.shape[1],output_dim],initializer=tf.random_normal_initializer())
    #         b = tf.get_variable("biases",[output_dim],initializer=tf.constant_initializer(0.0))
    #         return tf.matmul(x,w)+b
    #
    # def two_hidden_layers(x):
    #     h1 = fully_connected(x,50,"h1")
    #     h2 = fully_connected(h1,10,"h2")
    #
    # # 如果变量已存在则直接获取，否则创建新变量
    # with tf.variable_scope("two_layers") as scope:
    #     logits1 = two_hidden_layers(x1)
    #     logits2 = two_hidden_layers(x2)
    ##########################################################

    ##########################################################
    # tf.train.Saver
    # tf.train.Saver.save(sess=sess,save_path=sp,global_step=None)
    # tf.train.Saver.restore(sess=sess,save_path=sp)

    # 1000步之后保存参数
    # # 定义模型
    # model = SkipGramModel(params)
    # # create a saver object
    # saver = tf.train.Saver()
    # with tf.Session() as sess:
    #     for step in range(training_steps):
    #         sess.run([optimizer])
    #
    #         # save model every 1000 steps
    #         if(step+1)% 1000 ==0:
    #             # 指定哪一步保存
    #             saver.save(sess,"checkpoint_directory/modle_name",global_step=step)
    ##########################################################


    ##########################################################
    # # Global step
    # global_step = tf.variable(0,dtype=tf.int32,trainable=False,name="global_step")
    #
    # optimizer = tf.train.AdamOptimizer(lr).minimize(loss,global_step=global_step)
    ##########################################################


    ##########################################################
    # # tf.train.Saver 【注意：只保存变量，不保存值】
    # v1 = tf.variable(name="v1",...)
    # v2 = tf.variable(name="v2",...)
    #
    # # 3种方式保存变量
    # saver = tf.train.Saver({"v1":v1,"v2":v2})
    #
    # saver = tf.train.Saver([v1,v2])
    #
    # saver = tf.train.Saver({v.op.name : v for v in [v1,v2]})
    #
    # # 还原变量
    # saver.restore(sess,"checkpoint/name_of_the_checkpoint")
    ##########################################################


    ##########################################################
    # # tf.summary(在训练过程中可视化总结统计)
    # # tf.summary.scalar
    # # tf.summary.histogram
    # # tf.summary.image
    #
    # # step1:create summaries
    # with tf.name_scope("summaries"):
    #     tf.summary.scalar("loss",self.loss)
    #     tf.summary.scalar("accuracy",slef.accuracy)
    #     tf.summary.histogram("histogram_loss",slef.loss)
    #     summary_op = tf.summary.merge_all()         #将它们全部合并到一个概要的op中，使得更容易管理
    #
    # # step2: run them
    # loss_batch,_,summary = sess.run([loss,optimizer,summary_op])
    #
    # # step3: write summaries to file
    # writer.add_summary(summary,global_step=step)
    ##########################################################


    ##########################################################
    # 综合示例
    tf.summary.scalar("loss", self.loss)
    tf.summary.histogram("histogram loss", self.loss)
    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()  # defaults to saving all variables

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        writer = tf.summary.FileWriter('./graphs', sess.graph)
        for index in range(10000):
            ...

            if (index + 1) % 1000 == 0:
                saver.save(sess, 'checkpoints/skip-gram', index)
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
