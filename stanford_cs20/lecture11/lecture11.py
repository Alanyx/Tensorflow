"""
Created on 2018/11/14
@author: AlanYx
"""

import tensorflow as tf

# RNNs专题
if __name__ == '__main__':
    ##########################################################
    # #  Cell Support (tf.nn.rnn_cell)
    # """
    # ● BasicRNNCell: The most basic RNN cell.
    # ● RNNCell: Abstract object representing
    # an RNN cell.
    # ● BasicLSTMCell: Basic LSTM recurrent
    #   network cell.
    # ● LSTMCell: LSTM recurrent network cell.
    # ● GRUCell: Gated Recurrent Unit cell
    # """
    # cell = tf.nn.rnn_cell.GRUCell(hidden_size)
    ##########################################################

    ##########################################################
    # # 堆叠多个细胞单元
    # layers = [tf.nn.rnn_cell.GRUCell(size) for size in hidden_size]
    # cells = tf.nn.rnn_cell.MultiRNNCell(layers)
    # output,out_state = tf.nn.dynamic_rnn(cell,seq,length,initial_state) # 这里的问题在于大多数序列的长度不相同
    #
    # # # 构建rnn
    # # tf.nn.dynamic_rnn
    # # tf.nn.bidirectional_dynamic_rnn
    ##########################################################

    ##########################################################
    # 处理变量序列的长度
    """
    * 用零向量填充所有的序列和所有标签
    * 目前大多数模型不能处理长度大于120个令牌的序列，所以通常有一个固定的max_length(最大长度).，我们将序列截断为max_length
    """
    # 填充或者截断序列长度问题： 填充的标签改变了总的损失，这将影响梯度结果
    # 方法1:
    """
    ● Maintain a mask (True for real, False for padded tokens)
    ● Run your model on both the real/padded tokens (model will predict labels
    for the padded tokens as well)
    ● Only take into account the loss caused by the real elements
    """
    full_loss = tf.nn.softmax_cross_entropy_with_logits(preds,labels)
    loss = tf.reduce_mean(tf.boolean_mask(full_loss,mask))

    # 方法2:
    """
    ● Let your model know the real sequence length so it only predict the labels for the real tokens
    """
    cell = tf.nn.rnn_cell.GRUCell(hidden_size)
    rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cell]*num_layers)
    tf.reduce_sum(tf.reduce_max(tf.sign(seq),2),1)
    output, out_state = tf.nn.dynamic_rnn(cell, seq, length, initial_state)
    ##########################################################

    ##########################################################
    #  Tips and Tricks
    # 1.梯度消失
    """
    Use different activation units:
    ● tf.nn.relu 
    ● tf.nn.relu6 
    ● tf.nn.crelu 
    ● tf.nn.elu
    
    In addition to:
    ● tf.nn.softplus
    ● tf.nn.softsign
    ● tf.nn.bias_add 
    ● tf.sigmoid
    ● tf.tanh
    """

    # 2.梯度爆炸
    # 用tf.clip_by_global_norm裁剪梯度

    # 对所有可训练的变量求取代价的梯度
    gradients = tf.gradients(cost,tf.trainable_variables())
    # 通过预定义的max norm裁剪梯度
    clipped_gradients,_ = tf.clip_by_global_norm(gradients,max_grad_norm)
    # 将裁剪的梯度加到优化器中 
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(gradients,trainables))

    # 3.磨练学习率
    # 优化器同时可以接受标量和张量作为学习率 
    learning_rate = tf.train.exponential_decay(init_lr,global_step,decay_steps,staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # 4.过拟合
    # 方式1：通过对所有的细胞用tf.nn.dropout或者DropoutWrapper使用随机失活
    # tf.nn.dropout
    hidden_layer = tf.nn.dropout(hidden_layer,keeep_prob)

    # 方式2：DropoutWrapper
    cell = tf.nn.rnn_cell.GRUCell(hidden_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=keep_prob)


    ##########################################################

    ##########################################################
    #  Language Modeling
    # 1.word-level：n-grams

    # 2.character-level

    # 3.subword-level:介于以上两种情况

    ##########################################################
    ##########################################################
