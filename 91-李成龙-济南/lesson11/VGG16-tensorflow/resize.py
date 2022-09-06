import matplotlib.image as mpimg  # 导入读取图像的包
import matplotlib.pyplot as plt  # 导入读取图像的包
import cv2 as cv
import numpy as np
path = "./test_data/dog.jpg"
img = mpimg.imread(path)
print("img:", img.shape)
plt.imshow(img)   # 打印出来是 h, w, c
# 将图片修剪成中心的正方形
short_edge = min(img.shape[:2])  # img.shape[:2] 取彩色图片的长、宽。如果img.shape[:3] 则取彩色图片的长、宽、通道.
yy = int((img.shape[0] - short_edge) / 2)
xx = int((img.shape[1] - short_edge) / 2)
crop_img = img[yy: yy + short_edge, xx: xx + short_edge]    # 将图像裁剪成中心正方形？疑问这样是不是裁掉了部分特征信息。
# print(crop_img)
print("crop_img:", crop_img.shape)
cv.imshow("crop_img", crop_img)
cv.waitKey()
cv.destroyAllWindows()

