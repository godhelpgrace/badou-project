#!/usr/bin/env python
# -*-coding:utf-8-*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]  # 创建200个-0.5到0.5之间的随机数值序列，生成二维数组,作为输入数据
noise = np.random.normal(0, 0.02, x_data.shape)  # 生成和x_data形状相同的服从正态分布的随机数值
y_data = np.square(x_data) + noise  # y_data=x_data^2+noise

x = tf.placeholder(tf.float32, [None, 1])  # 定义占位符存放数据（1列，行不确定）
y = tf.placeholder(tf.float32, [None, 1])  # 定义占位符存放数据（1列，行不确定）

'''定义神经网络的隐含层'''
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))  # 定义权重
biases_L1 = tf.Variable(tf.zeros([1, 10]))  # 定义偏置项
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)  # 加入激活函数，把值压缩到 -1～1 之间

'''定义神经网络的输出层'''
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))  # 定义权重
biases_L2 = tf.Variable(tf.zeros([1, 1]))  # 定义偏置项
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)  # 加入激活函数，把值压缩到 -1～1 之间

loss = tf.reduce_mean(tf.square(y - prediction))  # 定义损失函数（均方差函数）
# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # minimize() 函数处理了梯度计算和参数更新两个操作

with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 训练网络
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
    # 预测
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

# 绘制图形
plt.figure()  # 新建一个图像
plt.scatter(x_data, y_data, c="k", marker="x")  # 生成真实值的散点图
plt.plot(x_data, prediction_value, "g-", lw=3)  # 生成预测曲线
plt.show()
