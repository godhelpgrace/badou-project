# utils 自定义的包或文件，用到什么函数或者方法直接导入此包调用即可。
# 图像预处理，排序等操作在此py文件中
# utils里是一些工具代码（工具人），包括载入图片、图片大小更改、打印预测结果等.
import matplotlib.image as mpimg  # 导入读取图像的包
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops

def load_image(path):
    # 读取图片，rgb
    img = mpimg.imread(path)
    # 将图片修剪成中心的正方形
    short_edge = min(img.shape[:2])  # img.shape[:2] 取彩色图片的长、宽。如果img.shape[:3] 则取彩色图片的长、宽、通道.
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img

def resize_image(image, size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):   # 调用tensorflow的包作图像预处理工作。
    with tf.name_scope('resize_image'):
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size,
                                       method, align_corners)
        image = tf.reshape(image, tf.stack([-1, size[0], size[1], 3]))
        return image

def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]  # 按行读取文件
    # 将概率从大到小排列的结果的序号存入pred
    pred = np.argsort(prob)[::-1]
    # 取最大的1个、5个。
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1



