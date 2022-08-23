#!/usr/bin/env python
# -*-coding:utf-8-*-

import numpy
import scipy.special


class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 初始化网络，设置输入层，隐藏层，和输出层节点数,这样就能决定网络的形状和大小。
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 设置学习率
        self.lr = learningrate
        '''
        初始化权重矩阵，有两个权重矩阵，一个是wih表示输入层和中间层节点间链路权重形成的矩阵;一个是who,表示中间层和输出层间链路权重形成的矩阵.
        由于权重不一定都是正的，它完全可以是负数，因此我们在初始化时，把所有权重初始化为-0.5到0.5之间
        '''
        # self.wih=numpy.random.rand(self.hnodes,self.inodes)-0.5
        # self.who=numpy.random.rand(self.onodes,self.hnodes)-0.5
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), size=(self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), size=(self.onodes, self.hnodes))
        print(self.wih)
        print(self.who)
        '''
        设置激活函数，每个节点执行激活函数，得到的结果将作为信号输出到下一层。scipy.special.expit对应的是sigmod函数.
        lambda是Python关键字，类似C语言中的宏定义.调用self.activation_function(x)时，编译器会把其转换为spicy.special_expit(x)。
        '''
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        '''
        把inputs_list, targets_list转换成二维矩阵，.T表示做矩阵的转置
        '''
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # 计算信号经过输入层后产生的信号量
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 隐藏层神经元对输入的信号做激活函数后得到输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层接收来自隐藏层的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 输出层对信号量进行激活函数后得到最终输出信号
        final_outputs = self.activation_function(final_inputs)
        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr * numpy.dot(output_errors * final_outputs * (1 - final_outputs),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), numpy.transpose(inputs))
        # pass

    def query(self, inputs):
        # 根据输入数据计算并输出答案
        # 计算中间层从输入层接收到的信号量
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层接收到的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs


# 初始化网络
'''
一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点
'''
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 读入训练数据
# open函数里的路径根据数据存储的路径来设定
training_data_file = open('dataset/mnist_train.csv', mode="r")  # 创建一个file对象
training_data_list = training_data_file.readlines()  # 读取全部行
training_data_file.close()  # 关闭文件

# 设定网络的训练循环次数epocs
epocs = 50
for i in range(epocs):
    # 把数据根据','区分，并分别读入
    for record in training_data_list:
        all_values = record.split(",")
        '''
        我们需要做的是将数据“归一化”，也就是把所有数值全部转换到0.01到1.0之间。由于表示图片的二维数组中，每个数大小不超过255，由此我们只要把所有数组除以255，就能让数据全部落入到0和1之间。
        有些数值很小，除以255后会变为0，这样会导致链路权重更新出问题。所以我们需要把除以255后的结果先乘以0.99，然后再加上0.01，这样所有数据就处于0.01到1之间。第一个值对应的是图片的表示的数字
        '''
        inputs = numpy.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
        # 设置图片与数值的对应关系，将图片数字对应的索引设置数值最大
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

# 读入测试数据
# open函数里的路径根据数据存储的路径来设定
testing_data_file = open('dataset/mnist_test.csv', mode="r")  # 创建一个file对象
testing_data_list = testing_data_file.readlines()  # 读取全部行
testing_data_file.close()  # 关闭文件
'''
把所有测试图片都输入网络，看看它检测的效果如何
'''
scores = []
for record in testing_data_list:
    all_values = record.split(",")  # 把数据根据','区分，并分别读入
    # 第一个值对应的是图片的表示的数字
    correct_number = int(all_values[0])
    print("该图片对应的数字是：", correct_number)
    '''
    我们需要做的是将数据“归一化”，也就是把所有数值全部转换到0.01到1.0之间。由于表示图片的二维数组中，每个数大小不超过255，由此我们只要把所有数组除以255，就能让数据全部落入到0和1之间。
    有些数值很小，除以255后会变为0，这样会导致链路权重更新出问题。所以我们需要把除以255后的结果先乘以0.99，然后再加上0.01，这样所有数据就处于0.01到1之间。
    '''
    inputs = numpy.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
    outputs = n.query(inputs)  # 让网络判断图片对应的数字
    label = numpy.argmax(outputs)  # 找到数值最大的神经元对应的编号
    print("网络判断的图片数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
    print("result:", scores)

# 计算图片判断的成功率
scores_array = numpy.asarray(scores)
print("perfermance=", scores_array.sum() / scores_array.size)
