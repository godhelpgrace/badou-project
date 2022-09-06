#!/usr/bin/env python
# -*-coding:utf-8-*-

from Alexnet import AlexNet
import cv2
import numpy as np
import Utils
from keras import backend as k

k.set_image_dim_ordering("tf")  # 设置图像的维度顺序

if __name__ == "__main__":
    # 创建AlexNet模型
    model = AlexNet()
    # 读取权重
    model.load_weights("./logs/last1.h5")
    img = cv2.imread("./Test.jpg")
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_nor = img_RGB / 255
    # 扩展数组的形状
    img_nor = np.expand_dims(img_nor, axis=0)
    img_resize = Utils.resize_image(img_nor, (224, 224))
    Utils.print_answer(np.argmax(model.predict(img_resize)))
    cv2.imshow("animal", img)
    cv2.waitKey()
