#!/usr/bin/env python
# -*-coding:utf-8-*-
# -------------------------------------------------------------#
#   ResNet50的网络部分
# -------------------------------------------------------------#
from keras.layers import Conv2D, BatchNormalization, Activation, Input, Dense
from keras.layers import ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten
from keras import layers
from keras.models import Model
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, kernel_size=(1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])  # 直接对张量求和
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])  # 直接对张量求和
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=[224, 224, 3], classes=1000):
    img_input = Input(shape=input_shape)  # 构建网络输入层
    # 表示将上一层的输出上下左右补充3行（3列）,行数+6,列数+6
    x = ZeroPadding2D((3, 3))(img_input)

    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # 全局平均值池化。AveragePooling2D最后返回的tensor是 [batch_size, channels, pooled_rows, pooled_cols] 4个维度的。
    x = AveragePooling2D(pool_size=(7, 7), name='avg_pool')(x)
    x = Flatten()(x)  # 将输入层的数据压成一维的数据，一般用在卷积层和全连接层之间
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')
    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")  # 仅读取权重

    return model


# def preprocess_input(x):
#     x /= 255.
#     x -= 0.5
#     x *= 2.
#     return x


if __name__ == '__main__':
    model = ResNet50()
    model.summary()  # 输出模型各层的参数状况
    image_path = 'elephant.jpg'
    # image_path = 'bike.jpg'
    # 指定图像路径读取图像。只是加载了一个文件，没有形成numpy数组。
    img = image.load_img(path=image_path, target_size=(224, 224))
    # 把numpy矩阵中的整数转换成浮点数
    x = image.img_to_array(img)
    # 在指定轴axis上增加数组a的一个维度
    x = np.expand_dims(x, axis=0)
    # 数据预处理,能够提高算法的运行效果,常用的预处理包括数据归一化和白化（whitening）。
    x = preprocess_input(x)
    print("Input image shape:", x.shape)
    preds = model.predict(x)
    # 返回一个预测的列表值。
    print("Predicted:", decode_predictions(preds))
