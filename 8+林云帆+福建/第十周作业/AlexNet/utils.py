import cv2
import numpy as np
import tensorflow as tf

def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i,size)
            images.append(i)
        images = np.array(images)
        return images


def print_answer(argmax):
    with open('./data/model/index_word.txt','r',encoding='utf-8') as f:
        synset = [l.split(';')[1][:-1]for l in f.readlines()]
    print(synset[argmax])
    return synset[argmax]