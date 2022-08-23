#!/bin/usr/python
# encoding=utf-8

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout


def AlexNet(input_shape=(224, 224, 3), output_shape=2):
    # 描述各层网络,每一个数据处理层串联起来
    model = Sequential()
    # 使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层，输出的shape为(55,55,96)；
    # 所建模型后输出为48特征层
    model.add(Conv2D(filters=48, kernel_size=(11, 11), strides=(4, 4), padding="valid", input_shape=input_shape,
                     activation="relu"))
    # 会使用当前批次数据的均值和标准差做归一化
    model.add(BatchNormalization())
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,96)
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
    # 使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(27,27,256)；
    # 所建模型后输出为128特征层
    model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu"))
    # 会使用当前批次数据的均值和标准差做归一化
    model.add(BatchNormalization())
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(13,13,256)；
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    # 所建模型后输出为192特征层
    model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    # 所建模型后输出为192特征层
    model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(13,13,256)；
    # 所建模型后输出为128特征层
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(6,6,256)；
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

    # 用于将输入层的数据压成一维的数据，一般用在卷积层和全连接层之间（因为全连接层只能接收一维数据，而卷积层可以处理二维数据，就是全连接层处理的是向量，
    # 而卷积层处理的是矩阵）
    model.add(Flatten())
    model.add(Dense(units=1024, activation="relu"))  # 全连接层 # 缩减为1024
    model.add(Dropout(rate=0.25))  # 应用于输入Dropout层在训练期间的每一步中将输入单位随机设置为0，频率为速率，这有助于防止过拟合。

    model.add(Dense(units=1024, activation="relu"))  # 全连接层
    model.add(Dropout(rate=0.25))  # 应用于输入Dropout层在训练期间的每一步中将输入单位随机设置为0，频率为速率，这有助于防止过拟合。

    model.add(Dense(units=output_shape, activation="softmax"))  # 两个全连接层，最后输出为1000类,这里改为2类
    return model
