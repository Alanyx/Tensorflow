"""
Created on 2018/10/27
@author: AlanYx
"""
import tensorflow as tf

"""
tf.multiply与tf.matmul的区别:
"""
#####################################################
if __name__ == '__main__':
    # 两个矩阵对应的元素各自相乘
    x0 = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    y0 = tf.constant([[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0]])
    # 注意这里的x,y必须是相同的类型，否则会因为数据类型不匹配出错
    z0 = tf.multiply(x0, y0)

    # 两个数相乘
    x1 = tf.constant(1)
    y1 = tf.constant(2)
    z1 = tf.multiply(x1, y1)

    # 数与矩阵相乘
    x2 = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    y2 = tf.constant(2.0)
    z2 = tf.multiply(x2, y2)

    # 两个矩阵相乘
    x3 = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    y3 = tf.constant([[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0]])
    # 注意这里的x,y要满足矩阵相乘的格式要求
    z3 = tf.matmul(x3, y3)

    with tf.Session() as sess:
        print("z0:", sess.run(z0))
        print("z1:", sess.run(z1))
        print("z2:", sess.run(z2))
        print("z3:", sess.run(z3))

#####################################################
