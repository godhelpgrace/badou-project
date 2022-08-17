import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np


def load_image(path):
    # 读取图像，rgb
    img = mpimg.imread(path)
    # 将图片修剪成中心的正方形(从中间裁剪成高、宽一样)
    short_edge = min(img.shape[:2])  # 获取宽、高中的最小值
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    img_crop = img[yy:yy + short_edge, xx:xx + short_edge]
    return img_crop


def resize_image(image, size, method=tf.image.ResizeMethod.BILINEAR, align_corners=False):
    with tf.name_scope("resize_image"):  # 命名空间,就是给几个变量包一层名字，方便变量管理;解决命名冲突问题。
        # 用于给函数增加维度
        image = tf.expand_dims(image, 0)
        # 通过指定的方法method来调整输入图像images为指定的尺寸size。
        image = tf.image.resize_images(image, size, method, align_corners)
        # 改变张量（tensor）的形状,如果沿着axis ==0 进行拼接，那么拼接后的输入的tensor的shape为（N,A,B,C）
        image = tf.reshape(image, tf.stack([-1, size[0], size[1], 3]))
        return image


def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    # 将概率从大到小排列的结果下标的序号存入pred
    pred = np.argsort(prob)[::-1]
    # 取最大的1个。
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # 取最大的5个。
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1
