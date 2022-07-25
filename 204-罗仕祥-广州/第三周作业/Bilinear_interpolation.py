#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2022/7/17 22:47
@Author  : luoshixiang
@Email   : just_rick@163.com
@File    : Bilinear_interpolation.py
@effect  : 双线性插值

"""

import cv2
import numpy as np

def bilinear_interpolation(src_img, dst_height, dst_width):
    """
    双线性插值
        在两个方向分别进行一次线性插值
    @param src_img: 来源图像矩阵
    @param dst_height: 生成图像的长度
    @param dst_width: 生成图像的宽度
    @return: 生成图像矩阵
    """
    src_height, src_width, channels = src_img.shape
    if src_height == dst_height and src_width == dst_width:
        return src_img.copy()
    dst_img = np.zeros((dst_height, dst_width, channels), np.uint8)  # 创建空白图层
    # 计算对应比例值
    scale_x = src_width / dst_width
    scale_y = src_height / dst_height

    # 通过dst像素点的坐标对应到src图像当中的坐标；然后通过双线性插值的方法算出src中相应坐标的像素值
    for c in range(channels):   # 按channels循环遍历每一个坐标点
        for h in range(dst_height):
            for w in range(dst_width):
                # # 按比例对应
                # src_x = h * scale_x
                # src_y = w * scale_y
                # 按几何中心对应,即目标在源上的坐标
                src_x = (w + 0.5)*scale_x - 0.5
                src_y = (h + 0.5) * scale_y - 0.5

                # 找到相邻的四个邻近点坐标, 不能超过src的边界同时坐标必须是整型
                src_x0 = int(np.floor(src_x))
                src_y0 = int(np.floor(src_y))
                src_x1 = min(src_x0 + 1, src_width - 1)
                src_y1 = min(src_y0 + 1, src_height - 1)

                # 双线性插值
                # 在x方向做插值，分母取1
                value_x1 = (src_x1 - src_x) * src_img[src_y0, src_x0, c] + (src_x - src_x0) * src_img[src_y0, src_x1, c]
                value_x2 = (src_x1 - src_x) * src_img[src_y1, src_x0, c] + (src_x - src_x0) * src_img[src_y1, src_x1, c]
                # 在y方向做插值
                dst_img[h, w, c] = (src_y1 - src_y) * value_x1 + (src_y - src_y0) * value_x2
    return dst_img


if __name__ == '__main__':

    image_file = 'lenna.png'

    src_img = cv2.imread(image_file)
    dst_img = bilinear_interpolation(src_img, dst_height=700, dst_width=700)
    print(src_img.shape, dst_img.shape)

    cv2.imshow('bilinear interpolation', dst_img)
    cv2.waitKey(0)

