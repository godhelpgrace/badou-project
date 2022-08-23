#!/usr/bin/env python
# -*-coding:utf-8-*-
# -------------------------------------------------------------#
#   InceptionV3的网络部分
# -------------------------------------------------------------#
from keras import layers
from keras.layers import Conv2D, BatchNormalization, Activation, Input, MaxPooling2D
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, Dense
from keras import Model
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import decode_predictions


# from keras.applications.imagenet_utils import preprocess_input

def conv2d_bn(x, filters, num_row, num_col, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False, name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def inception_module_block1(input, branch_pool_conv_filters, name):
    branch1x1 = conv2d_bn(input, 64, 1, 1)

    branch5x5 = conv2d_bn(input, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(input, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_pool = conv2d_bn(branch_pool, branch_pool_conv_filters, 1, 1)
    # 64+64+96+branch_pool_conv_filters
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name=name)
    return x


def inception_module_block2(input, branch7x7_filters, name):
    branch1x1 = conv2d_bn(input, 192, 1, 1)

    branch7x7 = conv2d_bn(input, branch7x7_filters, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, branch7x7_filters, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(input, branch7x7_filters, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, branch7x7_filters, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, branch7x7_filters, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, branch7x7_filters, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    # 192+192+192+192
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name=name)
    return x


def InceptionV3(input_shape=[299, 299, 3], classes=1000):
    img_input = Input(shape=input_shape)  # 构建网络输入层
    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # --------------------------------#
    #   Block1 35x35
    # --------------------------------#
    # Block1 part1
    # 35 x 35 x 192 -> 35 x 35 x 256
    x = inception_module_block1(x, 32, 'mixed0')

    # Block1 part2
    # 35 x 35 x 256 -> 35 x 35 x 288
    x = inception_module_block1(x, 64, 'mixed1')

    # Block1 part3
    # 35 x 35 x 288 -> 35 x 35 x 288
    x = inception_module_block1(x, 64, 'mixed2')

    # --------------------------------#
    #   Block2 17x17
    # --------------------------------#
    # Block2 part1
    # 35 x 35 x 288 -> 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed3')

    # Block2 part2
    # 17 x 17 x 768 -> 17 x 17 x 768
    x = inception_module_block2(x, 128, 'mixed4')

    # Block2 part3 and part4
    # 17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768
    for i in range(2):
        x = inception_module_block2(x, 160, name='mixed' + str(i + 5))

    # Block2 part5
    # 17 x 17 x 768 -> 17 x 17 x 768
    x = inception_module_block2(x, 192, 'mixed7')

    # --------------------------------#
    #   Block3 8x8
    # --------------------------------#
    # Block3 part1
    # 17 x 17 x 768 -> 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')

    # Block3 part2 part3
    # 8 x 8 x 1280 -> 8 x 8 x 2048 -> 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed' + str(9 + i))

    # 全局平均值池化。求四维数据（图片）的每个通道值c的平均，最后结果没有了宽（w）高（h）维度，最后返回的tensor是[batch_size, channels]两个维度的。
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # 全连接
    x = Dense(units=classes, activation='softmax', name='predictions')(x)

    # inputs=img_input
    model = Model(img_input, x, name='inceptionV3')
    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = InceptionV3()
    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")  # 仅读取权重
    model.summary()  # 输出模型各层的参数状况
    img_path = 'elephant.jpg'
    # img_path = 'bike.jpg'
    # 指定图像路径读取图像。只是加载了一个文件，没有形成numpy数组。
    img = image.load_img(img_path, target_size=(299, 299))
    # 把numpy矩阵中的整数转换成浮点数
    x = image.img_to_array(img)
    # 在指定轴axis上增加数组a的一个维度
    x = np.expand_dims(x, axis=0)
    # 数据预处理,能够提高算法的运行效果
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    preds = model.predict(x)
    # 返回一个预测的列表值。
    print("Predicted:", decode_predictions(preds))
