# -*- coding: utf-8 -*-
import tensorflow as tf
import input_data
mnist=input_data.read_data_sets('datasets/',one_hot=True)
x=tf.placeholder('float',(None,784))#每一个图片拉成784长度的向量，x是一个占位符，
#类似于形参，在运行的时候再定义具体值
#shape的第一个参数是None代表是可以任意长度的，无限大
W=tf.Variable(tf.zeros((784,10)))
b=tf.Variable(tf.zeros((10)))
#一个Variable代表一个可以修改的张量
#因为要学习W和b的值，所以初始值可以随意设置
y=tf.nn.softmax(tf.matmul(x,W)+b)
#tf.nn是modle,上述一行代码定义了模型
'''
以上是设置变量和定义模型
'''
y_=tf.placeholder('float',[None,10])#定义一个占位符用于输入正确值
cross_entropy=-tf.reduce_sum(y_*tf.log(y))
#tensorflow可以自动运行反向传播，因为有描述各个计算单元的图
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#tf会在后台计算图里添加一系列新的计算操作单元用于实现反向传播和梯度下降
init=tf.global_variables_initializer()
#初始化变量
sess=tf.Session()
sess.run(init)
#上述，启动模型，并且初始化变量
'''
开始训练,1000次
'''
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
#每次循环随机抓取100个批处理数据点，然后我们用这些数据带你作为参数替换之前
#的占位符来运行train_step
#上述称为随机梯度下降训练，所有数据训练需要很大开销
，所以每一次训练使用不同
#的数据子集，既可以减少开销，又可以学习到数据集的总体特性
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#上述是一组布尔值
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))
#求平均准确率
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
sess.close()