#!/usr/bin/env python
# -*-coding:utf-8-*-

import numpy as np
from Alexnet import AlexNet
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
import cv2
import Utils
from keras.utils import np_utils
from keras import backend as k

k.set_image_dim_ordering("tf")  # 设置图像的维度顺序


def generate_arrays_from_file(lines, batch_size):
    # 获取txt总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i == 0:
                # 打乱顺序
                np.random.shuffle(lines)
            # 获取图像名字
            name = lines[i].split(";")[0]
            # 从文件中读取图像
            # img = cv2.imread(r".\data\image\train" + '/' + name)
            img = cv2.imread(r"./data/image/train/" + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            X_train.append(img)
            Y_train.append(lines[i].split(";")[1])
            # 读完整个数据集后重新开始
            i = (i + 1) % n
        # 处理图像
        X_train = Utils.resize_image(X_train, (224, 224))
        X_train = X_train.reshape(-1, 224, 224, 3)
        # 将整型的类别标签转为onehot编码
        Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=2)
        yield (X_train, Y_train)


if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"
    # 打开、读取数据集的txt
    with open("./data/dataset.txt", "r") as f:
        lines = f.readlines()
    # 这个txt主要用于帮助读取数据来训练,打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    # 90%用于训练，10%用于估计
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val
    # 创建AlexNet模型
    model = AlexNet()

    # 保存的方式，3世代保存一次。该回调函数将在每个epoch后保存模型到filepath
    checkpoint_period1 = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                         monitor="acc", save_best_only=True, save_weights_only=False, period=3)
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练。当评价指标不在提升时，减少学习率。
    reduce_lr = ReduceLROnPlateau(monitor="acc", factor=0.5, patience=3, verbose=1)
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止。当监测值不再改善时，该回调函数将中止训练。
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1)
    # 配置训练模型。1e-3=0.001
    model.compile(optimizer=Adam(lr=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    # 一次训练的训练集大小
    batch_size = 128
    print("Train on {} samples, val on {} samples, with batch size {}.".format(num_train, num_val, batch_size))
    # 开始训练。利用Python的生成器，逐个生成数据的batch并进行训练。生成器与模型将并行执行以提高效率。
    # 例如，该函数允许我们在CPU上进行实时的数据提升，同时在GPU上进行模型训练。函数返回一个History对象。
    model.fit_generator(generator=generate_arrays_from_file(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        epochs=50,
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_val // batch_size),
                        initial_epoch=0,
                        callbacks=[checkpoint_period1, reduce_lr, early_stopping])
    # 保存模型参数(并没有保存模型的图结构)
    model.save_weights(log_dir + "last1.h5")
