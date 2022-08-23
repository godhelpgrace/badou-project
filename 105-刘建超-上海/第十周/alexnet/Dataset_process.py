#!/usr/bin/env python
# coding=utf-8

import os

photos = os.listdir("./data/image/train/")  # 返回目录下的所有文件名
with open("data/dataset.txt", "w") as f:
    for photo in photos:
        name = photo.split(".")[0]
        if name == "cat":
            f.write(photo + ";0\n")
        elif name == "dog":
            f.write(photo + ";1\n")
f.close()
