#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2022/7/17 21:41
@Author  : luoshixiang
@Email   : just_rick@163.com
@File    : Nearest_interpolation.py
@effect  : 最邻近插值法
        将scrH,scrW 四舍五入选取最接近的整数，得到该点的像素值（该整数坐标在原图上的像素值）。
        这样的做法会导致像素的变化不连续，在新图中会产生锯齿。
"""

import cv2
import numpy as np

def nearest_interpolation(src_img, dst_height, dst_width):
    """
    最邻近插值法
        将scrH,scrW 四舍五入选取最接近的整数，得到该点的像素值（该整数坐标在原图上的像素值）。
        这样的做法会导致像素的变化不连续，在新图中会产生锯齿。
    @param src_img: 来源图像矩阵
    @param dst_height: 生成图像的长度
    @param dst_width: 生成图像的宽度
    @return: 生成图像矩阵
    """
    src_height, src_width, channels = src_img.shape
    dst_img = np.zeros((dst_height, dst_width, channels), np.uint8)  # 创建空白图层
    scale_height = dst_height / src_height
    scale_width = dst_width / src_width

    for i in range(dst_height):
        for j in range(dst_width):
            # 计算新图像素坐标在原图上的映射坐标
            x = round(i / scale_height)
            y = round(j / scale_width)
            dst_img[i, j] = src_img[x, y]
    return dst_img

if __name__ == '__main__':
    image_file = 'lenna.png'

    src_img = cv2.imread(image_file)
    dst_img = nearest_interpolation(src_img, dst_height=800, dst_width=800)
    print(src_img.shape, dst_img.shape)

    cv2.imshow('near', dst_img)
    cv2.imshow('img', src_img)
    cv2.waitKey(0)


