# -*- coding=utf-8  -*-

import numpy as np
import scipy.special


class NeuralNetWork:
    # 初始话
    def __init__(self, input_nodes, hide_nodes, output_nodes, learning_rate):
        # 输入层、隐藏层、输出层节点、学习率、激活函数
        self.input_nodes = input_nodes
        self.hide_nodes = hide_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # 初始化权重
        self.input_hide_weight = np.random.normal(0.0, pow(self.hide_nodes, -0.5), (self.hide_nodes, self.input_nodes))
        self.hide_out_weight = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hide_nodes))

        # 激活函数
        self.activate_function = lambda x: scipy.special.expit(x)

    # 训练模型

    def train(self, inputs, target):
        # 转化输入和输出的数据格式
        input_array = np.array(inputs, ndmin=2).T
        target_array = np.array(target, ndmin=2).T
        # 计算输出
        hide_input = np.dot(self.input_hide_weight, input_array)
        hide_output = self.activate_function(hide_input)
        out_input = np.dot(self.hide_out_weight, hide_output)
        out_output = self.activate_function(out_input)
        # 计算反向传播
        out_loss = target_array - out_output
        hide_loss = np.dot(self.hide_out_weight.T, out_loss*out_output*(1-out_output))
        # 更新参数
        self.hide_out_weight += self.learning_rate*np.dot(out_loss*out_output*(1-out_output),
                                                          np.transpose(hide_output))
        self.input_hide_weight += self.learning_rate*np.dot(hide_loss*hide_output*(1-hide_output),
                                                            np.transpose(input_array))
        pass

    # 推断

    def query(self, inputs):
        # 计算各层的数据
        hide_input = np.dot(self.input_hide_weight, inputs)
        hide_output = self.activate_function(hide_input)
        out_input = np.dot(self.hide_out_weight, hide_output)
        out_output = self.activate_function(out_input)
        return out_output


if __name__ == "__main__":
    # input_nodes = 784
    # hidden_nodes = 200
    output_nodes = 10
    # learning_rate = 0.1
    epochs = 5
    n = NeuralNetWork(input_nodes=784, hide_nodes=200, output_nodes=10, learning_rate=0.1)

    # 训练集
    with open(r"../data/mnist_train.csv") as f:
        train_data_list = f.readlines()
    for i in range(epochs):
        for line in train_data_list:
            feature = np.asfarray(line.split(","))[1:]/255*0.99+0.01
            label = np.zeros(output_nodes) + 0.01
            label[int(line.split(",")[0])] = 0.99
            n.train(feature, label)

    # 测试集
    with open(r"../data/mnist_test.csv") as f:
        test_data_list = f.readlines()
    confidence_list = list()
    for line in test_data_list:
        feature = np.asfarray(line.split(","))[1:] / 255 * 0.99 + 0.01
        label = int(line.split(",")[0])
        label_hat = np.argmax(n.query(feature))
        if label_hat == label:
            confidence_list.extend([1])
        else:
            confidence_list.extend([0])
    #  计算得分
    accuracy_ratio = sum(confidence_list)/len(confidence_list)
    print(accuracy_ratio)
