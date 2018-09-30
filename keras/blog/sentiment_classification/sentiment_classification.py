"""
Created on 2018/9/6
@author: AlanYx
"""
# 导入3.x的特征函数
from __future__ import absolute_import
from __future__ import print_function
import pandas as pd
import numpy as np
# 导入结巴分词
import jieba

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU



if __name__ == '__main__':

    # 读取训练语料完毕
    neg = pd.read_excel("neg.xls", header=None, index=None)
    pos = pd.read_excel("pos.xls", header=None, index=None)

    # 给训练语料贴上标签
    pos["mark"] = 1
    neg["mark"] = 0

    # 合并语料
    pn = pd.concat([pos, neg], ignore_index=True)
    neglen = len(neg)

    # 计算语料数目
    poslen = len(pos)

    # 定义分词函数
    cw = lambda x: list(jieba.cut(x))
    pn["words"]=pn[0].apply(cw)

    #读入评论内容
    comment = pd.read_excel("sum.xls")
    #comment = pd.read_csv('a.csv', encoding='utf-8')

    #仅读取非空评论
    comment = comment[comment["rateContent"].notnull()]

    #评论分词
    comment["words"] = comment["rateContent"].apply(cw)

    d2v_train = pd.concat([pn["words"],comment["words"]],ignore_index=True)

    # 将所有词语整合在一起
    w = []
    for i in d2v_train:
        w.extend(i)

    # 统计词的出现次数
    dict = pd.DataFrame(pd.Series(w).value_counts())
    del d2v_train
    dict["id"] = list(range(1,len(dict)+1))

    get_sent = lambda x:list(dict["id"][x])
    #速度太慢
    pn["sent"]= pn["words"].apply(get_sent)

    maxlen =50

    print("Pad sequences (samples x time)")
    pn["sent"] = list(sequence.pad_sequences(pn["sent"],maxlen=maxlen))

    #训练集
    x = np.array(list(pn["sent"]))[::2]
    y = np.array(list(pn["mark"]))[::2]
    #测试集
    x_t = np.array(list(pn["sent"]))[1::2]
    y_t = np.array(list(pn["mark"]))[1::2]
    # 全集
    x_a = np.array(list(pn["sent"]))
    y_a = np.array(list(pn["mark"]))

    print("Build model ...")
    model = Sequential()
    model.add(Embedding(input_dim=len(dict) + 1, output_dim=256, input_length=maxlen))
    # try using a GRU instead, for fun
    model.add(LSTM(128))
    # model.add(LSTM((256,128)))
    # model.add(GRU())
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss="binary_crossentropy",optimizer="adam")

    model.fit(x,y,batch_size=16,epochs=10)

    classes = model.predict(x_t)
    accuracy = np_utils.accuracy(classes,y_t)
    print("test accuracy:",accuracy)


