#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2022/7/21 23:27
@Author  : luoshixiang
@Email   : just_rick@163.com
@File    : Histogram_equalization.py
@effect  : 直方图均衡化
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

def histogram_equalization_one_channel(src_img, ):
    # 单通道-直方图均衡化
    # 获取灰度图像
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("image_gray", gray)

    # 灰度图像直方图均衡化
    '''
    equalizeHist—直方图均衡化
    函数原型： equalizeHist(src, dst=None)
    src：图像矩阵(单通道图像)
    dst：默认即可
    '''
    dst = cv2.equalizeHist(gray)

    # 直方图
    '''
    cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]])
        imaes:输入的图像
        channels:选择图像的通道， 0，1，2
        mask:掩模，是一个大小和image一样的np数组，其中把需要处理的部分指定为1，不需要处理的部分指定为0，一般设置为None，表示处理整幅图像 mask必须是一个8位（CV_8U）
        histSize:使用多少个bin(柱子)，一般为256
        ranges:像素值的范围，一般为[0,255]表示0~255
        后面两个参数基本不用管。  hist：直方图计算的输出值  accumulate=false：在多个图像时，是否累积计算像素值的个数
        注意，除了mask，其他四个参数都要带[]号。
    '''
    hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
    # plt.figure()
    # plt.hist(dst.ravel(), 256)
    # plt.show()

    return gray, dst



def histogram_equalization_3_channel(src_img, ):
    # 彩色图像直方图均衡化, 需要分解通道 对每一个通道均衡化
    (b, g, r) = cv2.split(src_img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    return result



if __name__ == '__main__':
    image_file = 'lenna.png'
    src_img = cv2.imread(image_file)

    cv2.imshow("src", src_img)
    cv2.waitKey(0)

    gray, dst = histogram_equalization_one_channel(src_img, )
    cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
    cv2.waitKey(0)

    result = histogram_equalization_3_channel(src_img, )
    cv2.imshow("dst_rgb", result)
    cv2.waitKey(0)

