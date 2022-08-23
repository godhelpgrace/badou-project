#!/usr/bin/env python
# -*-coding:utf-8-*-

import Utils
import tensorflow as tf
from nets import VGG16

# 读取图像(图片修剪成中心的正方形)
img = Utils.load_image("./test_data/dog.jpg")
# 对输入的图片进行resize，使其shape满足(-1,224,224,3)
inputs = tf.placeholder(tf.float32, [None, None, 3])  # 占位符，[None,3]，表示列是3，行不一定
resized_img = Utils.resize_image(inputs, (224, 224))
# 建立网络结构
prediction = VGG16.vgg_16(resized_img)
# 启动默认图
sess = tf.Session()
# TensorFlow的模型文件
ckpt_filename = "./model/vgg_16.ckpt"
# 初始化全局所有变量
sess.run(tf.global_variables_initializer())
# 创建tf.train.Saver类的对象
saver = tf.train.Saver()
# 恢复以前保存的变量
saver.restore(sess, ckpt_filename)
# 最后结果进行softmax预测
pro = tf.nn.softmax(prediction)
pre = sess.run(pro, feed_dict={inputs: img})
# 打印预测结果
print("result:")
Utils.print_prob(pre[0], "./synset.txt")
