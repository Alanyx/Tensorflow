"""
Created on 2018/9/6
@author: AlanYx
"""

"""
这段代码的作用是:通过深度学习方法实现10分类问题的训练及测试过程。具体来讲，就是生成一个1000条训练样例，每个样例20维，
    再使用同样20维的测试数据100条，通过一个序列模型使用SGD优化算法进行训练，其模型层数不多，3个全连接层和2个放弃层。
    就这样一个简单的深度学习（算不上深度）模型就搭建完毕了。麻雀虽小，五脏俱全。
"""

## 2.1引用包
# 引入序列模型
from keras.models import Sequential
# 引入全连接层、放弃层、激活层（激活层没有直接用到，但是在全连接层里间接用到了。）
from keras.layers import Dense, Dropout, Activation
# 引入SGD优化算法
from keras.optimizers import SGD
# 引入了metrics评估模块
from keras import metrics
# 引入了keras
import keras
# 使用numpy来模拟生成数据
import numpy as np

if __name__ == '__main__':

    ## 2.2生成数据
    # Generate dummy data
    # 生成一个1000*20维的向量
    x_train = np.random.random((1000, 20))
    # 生成一个1000*10维的向量
    # keras.utils.to_categorical:将一个类型的容器（整型）的转化为二元类型矩阵。比如用来计算多类别交叉熵来使用的。
    # 先通过np生成一个1000*1维的其值为0-9的矩阵，然后再通过```keras.utils.to_categorical```方法获取成一个1000*10维的二元矩阵
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    x_test = np.random.random((100, 20))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)


    ## 2.3构建模型
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    # 第一层为全连接层，隐含单元数为64，激活函数为relu，在第一层中一定要指明输入的维度。
    model.add(Dense(64, activation="relu", input_dim=20))
    # 随机失活层，将在训练过程中每次更新参数时随机断开一定百分比（rate）的输入神经元，Dropout层用于防止过拟合。这里是断开50%的输入神经元。
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    # 实例化优化算法为sgd优化算法
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=[metrics.categorical_accuracy, metrics.mae])

    ## 2.4训练模型
    model.fit(x_train, y_train, epochs=20, batch_size=128)

    ## 2.5评估模型
    score = model.evaluate(x_test, y_test, batch_size=128)
    print("score:" + str(score))
