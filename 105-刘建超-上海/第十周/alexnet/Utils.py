#!/usr/bin/python
# -*-coding:utf-8-*-

import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import cv2


def load_image(path):
    # 读取图片RGB
    img = mpimg(path)
    # 将图片修剪成中心的正方形(从中间裁剪成高、宽一样)
    short_edge = min(img.shape[:2])  # 获取宽、高中的最小值
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_image = img[yy:yy + short_edge, xx:xx + short_edge]
    return crop_image


def resize_image(image, size):
    with tf.name_scope("resize_image"):  # 命名空间,就是给几个变量包一层名字，方便变量管理;解决命名冲突问题。
        images = []
        for img in image:
            img = cv2.resize(img, size)
            images.append(img)
        images = np.array(images)
        return images


def print_answer(argmax):
    with open("./data/model/index_word.txt", "r", encoding="utf-8") as f:
        synset = [l.split(";")[1][:-1] for l in f.readlines()]

    print(synset[argmax])
    return synset[argmax]
