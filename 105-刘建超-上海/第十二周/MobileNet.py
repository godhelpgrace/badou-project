#!/usr/bin/env python
# -*-coding:utf-8-*-
# -------------------------------------------------------------#
#   MobileNet的网络部分
# -------------------------------------------------------------#
from keras.layers import Conv2D, BatchNormalization, Activation, DepthwiseConv2D, Input, GlobalAveragePooling2D
from keras.layers import Reshape, Dropout, Activation
from keras import backend as k
from keras.models import Model
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import decode_predictions


# from keras.applications.imagenet_utils import preprocess_input


def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel_size=kernel, strides=strides, padding='same', use_bias=False, name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    x = DepthwiseConv2D((3, 3), strides=strides, padding='same', depth_multiplier=depth_multiplier, use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1), strides=(1, 1), padding='same', use_bias=False,
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def relu6(x):
    return k.relu(x, max_value=6)


def MobileNet(input_shape=[224, 224, 3], depth_multiplier=1, dropout=1e-3, classes=1000):
    img_input = Input(shape=input_shape)

    # 224,224,3 -> 112,112,32
    x = _conv_block(img_input, 32, strides=(2, 2))

    # 112,112,32 -> 112,112,64
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)

    # 112,112,64 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)

    # 56,56,128 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    # 56,56,128 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4)

    # 28,28,256 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    # 28,28,256 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6)

    # 14,14,512 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 14,14,512 -> 7,7,1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    # 7,7,1024 -> 1,1,1024
    # 全局平均值池化。求四维数据（图片）的每个通道值c的平均，最后结果没有了宽（w）高（h）维度，最后返回的tensor是[batch_size, channels]两个维度的。
    x = GlobalAveragePooling2D()(x)
    # “重塑张量形状层”。 Reshape层用来将输入shape转换为特定的shape。
    x = Reshape(target_shape=(1, 1, 1024), name='reshape_1')(x)
    x = Dropout(rate=dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    inputs = img_input
    model = Model(inputs, x, name='MobileNet_1_0_224_tf')
    model.load_weights('mobilenet_1_0_224_tf.h5')  # 仅读取权重。
    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = MobileNet(input_shape=[224, 224, 3])
    model.summary()  # 输出模型各层的参数状况
    img_path = 'elephant.jpg'
    # 指定图像路径读取图像。只是加载了一个文件，没有形成numpy数组。
    img = image.load_img(img_path, target_size=(224, 224))
    # 把numpy矩阵中的整数转换成浮点数
    x = image.img_to_array(img)
    # 在指定轴axis上增加数组a的一个维度
    x = np.expand_dims(x, axis=0)
    # 数据预处理,能够提高算法的运行效果
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    preds = model.predict(x)
    # 返回数组沿着某一条轴最大值的索引。
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds, top=1))  # 只显示top1
