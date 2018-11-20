"""
Created on 2018/10/30
@author: AlanYx
"""

# Basic Models in TensorFlow
import tensorflow as tf


if __name__ == '__main__':
    ##########################################################
    # Linear Regression in TensorFlow(见讲义lecture3)
    # 具体实现见lecture3目录下的py文件
    ##########################################################

    ##########################################################
    """
    TF 控制流操作：
    * Control Flow Ops:
        tf.group, tf.count_up_to, tf.cond, tf.case, tf.while_loop, ...
        
    * Comparison Ops:
        tf.equal, tf.not_equal, tf.less, tf.greater, tf.where, ...
        
    * Logical Ops:
        tf.logical_and, tf.logical_not, tf.logical_or, tf.logical_xor
        
    * Debugging Ops:
        tf.is_finite, tf.is_inf, tf.is_nan, tf.Assert, tf.Print, ...
    """
    ##########################################################



    ##########################################################
    # # tf.data(以下为示意代码，无法运行)
    # """
    # tf.data.Dataset
    # tf.data.Iterator
    # """
    # # tf.data.Dataset
    # # 方式一:Store data in tf.data.Dataset : tf.data.Dataset.from_tensor_slices((features,labels))
    # dataset = tf.data.Dataset.from_tensor_slices((data[:,0],data[:,1]))
    # print(dataset.output_types)         # >> (tf.float32, tf.float32)
    # print(dataset.output_shapes)        # >> (TensorShape([]), TensorShape([]))
    #
    # # 方式二:Can also create Dataset from files
    # tf.data.TextLineDataset(filenames)
    # tf.data.FixedLengthDataset(filenames)
    # tf.data.TFRecordDataset(filenames)
    #
    # # tf.data.Iterator  (Create an iterator to iterate through samples in Dataset)
    # # 方式一:一次遍历数据集，无需初始化
    # iterator1 = dataset.make_one_shot_iterator()
    # X,Y = iterator1.get_next()
    # with tf.Session() as sess:
    #     print(sess.run([X,Y]))          # 迭代打印
    # # 方式二:按照需要多次迭代数据集，每次需要初始化
    # iterator2 = dataset.make_initializable_iterator()
    # .....
    # for i in range(100):
    #     sess.run(iterator2.initializer)
    #     total_loss = 0
    #     try:
    #         while True:
    #             sess.run([optimizer])
    #     except tf.errors.OutOgRangeError:
    #         pass
    ##########################################################



    ##########################################################
    # # tf中处理数据
    # dataset =[]
    # dataset = dataset.shuffle(1000)
    #
    # dataset = dataset.repeat(100)
    #
    # dataset = dataset.batch(1000)
    #
    # dataset = dataset.map(lambda x: tf.one_hot(x,10))   # 将数据集中每一个元素转换成one_hot向量
    ##########################################################


    ##########################################################
    # # Optimizer
    # # 会话照看所有可训练的变量，损失函数依赖并更新这些变量(如下例)
    # with tf.Session() as sess:
    #     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
    #     _, loss = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
    #
    # # 常见优化器列表
    # tf.train.GradientDescentOptimizer
    # tf.train.AdagradOptimizer
    # tf.train.MomentumOptimizer
    # tf.train.AdamOptimizer
    # tf.train.FtrlOptimizer
    # tf.train.RMSPropOptimizer
    # ...
    ##########################################################

