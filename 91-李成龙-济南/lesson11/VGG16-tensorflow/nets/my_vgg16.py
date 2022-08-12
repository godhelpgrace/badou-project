import tensorflow as tf


# 创建slim对象
slim = tf.contrib.slim
"""
contrib库非官方视觉相关人员提供的库，非原生。可以被贡献者随时更改。也有可能会被收收录到原生库。
slim是一个使构建，训练，评估神经网络变得简单的库。它可以消除原生tensorflow里面很多重复的模板
性的代码，让代码更紧凑，更具备可读性。另外slim提供了很多计算机视觉方面的著名模型（VGG, AlexNet
等），我们不仅可以直接使用，甚至能以各种方式进行扩展。
"""


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):


    with tf.variable_scope(scope, "vgg_16", [input]):  # tf.variable_scope()用来指定变量的作用域，作为变量名的前缀，支持嵌套
        # 建立vgg_16的神经网络
        # conv1两次[3,3]卷积网络，输出特征层为64，输出为（224，224，64）
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope="conv1")
        # 2*2最大池化，输出为（112,112,64）
        net = slim.max_pool2d(net, [2,2], scope="pool1")
        # conv两次[3,3]卷积网络，输出特征层为128，输出为（112，112，128）
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope="conv2")
        # 2*2最大池化，输出为（56，56，128）
        net = slim.max_pool2d(net, [2,2], scope="pool2")
        # conv3三次[3，3]卷积网络，输出特征层为256，输出为（56，56，256）
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope="conv3")
        # 2*2最大池化，输出为（28，28，256）
        net = slim.max_pool2d(net, [2,2], scope="pool3")
        # conv3三次[3,3]卷积网络，输出特征层为256，输出为（28，28，512）
        net = slim.repeat(net, 3, slim.conv2d, 512, [3,3], scope="conv4")
        # 2*2最大池化，输出为（14，14，512）
        net = slim.max_pool2d(net, [2,2], scope="pool4")
        # conv3三次[3,3]卷积网络，输出特征层为256，输出为（14，14，512）
        net = slim.repeat(net, 3, slim.conv2d, 512, [3,3], scope="conv5")
        #   # 2*2最大池化，输出为（7，7，512）
        net = slim.max_pool2d(net, [2, 2], scope="pool5")
        # 利用卷积的方式模拟全连接，效果等同，输出为（1，1，4096）
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope="fc6")
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope="dropout6")
        #  # 利用卷积的方式模拟全连接，效果等同，输出为（1，1，4096）
        net = slim.conv2d(net, 4096, [1, 1], scope="fc7")
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope="dropout7")
        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1000)
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope="fc8")
        # 由于用卷积的方式模拟全连接层，所以输出需要平铺。
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name="fc8/squeezd")
            # tf.squeeze用于从张量形状中移除大小为1的维度，设置为[1, 2]
            # 则表示在1、2纬度方向判断是否值是否为1，为1则删除
        return net

