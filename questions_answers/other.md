其他类型疑问与解答
=======

##tensorflow设置日志级别
###1.log 日志级别设置
~~~
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error 
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error 
--------------------- 
作者：姜兴琪 
来源：CSDN 
原文：https://blog.csdn.net/jxq0816/article/details/78699523 
版权声明：本文为博主原创文章，转载请附上博文链接！
~~~
###2.示例


## tensorboard学习使用
todo