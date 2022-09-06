# -*- coding=utf-8 -*-


# 导入必要包
import os
import numpy as np
import pandas as pd
import torch
import torch.onnx
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools import convert_sparkml
import onnxruntime
import onnx

import matplotlib.pyplot as plt

# 配置环境和超参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_works = 5
batch_size = 256
epochs = 200
lr = 1e-4

# 数据加载与预处理
img_size = 28
data_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(img_size), transforms.ToTensor()])


class FMnistDataSet(Dataset):

    def __init__(self, df_data, transforms=None):
        self.df_data = df_data
        self.transforms = transforms

        self.imgs = df_data.iloc[:, 1:].values.astype(np.uint8)
        self.label = df_data.iloc[:, 0].values

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx].reshape(28, 28, 1)
        label = int(self.label[idx])
        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = torch.tensor(img / 255., torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return img, label


# 网络设计
class FMnistNet(nn.Module):

    def __init__(self):
        super(FMnistNet, self).__init__()
        # input: 1*28*28

        self.conv = nn.Sequential(nn.Conv2d(1, 32, 5, padding=0, stride=1),
                                  # floor(28 - 5 + 0 + 1) = 24
                                  # 1*28*28 -> 32*24*24
                                  nn.ReLU(),
                                  # floor((24 - 2 + 2) / 2) = 12
                                  # 32*24*24 -> 32*12*12
                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                  nn.Dropout(0.3),
                                  # floor(12 - 5 + 0 + 1) = 8
                                  # 32*12*12 -> 64*8*8
                                  nn.Conv2d(32, 64, 5, padding=0, stride=1),
                                  nn.ReLU(),
                                  # floor((8 - 2 + 2)/2)
                                  # 64*8*8 -> 64*4*4
                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                  nn.Dropout(0.3)
                                  )
        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 512),
                                nn.ReLU(),
                                nn.Linear(512, 10))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        return x


# 训练模型
def train(epoch):
    model.train()
    train_loss = 0
    for data, label in train_dataloader:
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    train_loss = train_loss / len(train_dataloader.dataset)
    print("Epoch : {} \t train loss: {:.6f}".format(epoch, train_loss))


# 验证模型
def val(epoch):
    model.eval()
    val_loss = 0
    true_list = list()
    predict_list = list()
    with torch.no_grad():
        for data, label in test_dataloader:
            data, label = data.cuda(), label.cuda()

            out = model(data)
            label_hat = torch.argmax(out, 1)
            true_list.extend([label.cpu().data.numpy()])
            predict_list.extend([label_hat.cpu().data.numpy()])
            loss = criterion(out, label)
            val_loss += loss.item() * data.size(0)
    val_loss = val_loss / len(test_dataloader.dataset)
    true_labels, predict_labels = np.concatenate(true_list), np.concatenate(predict_list)
    acc = np.sum(true_labels == predict_labels) / len(predict_labels)
    print("Epoch : {} \t Validation Loss: {:.6f} \t Accuracy : {:.6f}".format(epoch, val_loss, acc))


# 预测数据
def inference(test_data, pth_model_path):
    dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=num_works)
    model = FMnistNet().to(device)
    model.load_state_dict(torch.load(pth_model_path))
    model.eval()
    predict_score_list = list()
    for batch, data in enumerate(dataloader_test):
        x = data[0]
        x = x.to(device)
        predict_score = model(x)
        predict_score_list.extend([predict_score.cpu().detach().numpy()])
    score = np.concatenate(predict_score_list, axis=0)
    print(score)
    print(np.argmax(score, axis=1))


if __name__ == "__main__":
    df_train = pd.read_csv(r"../data/fm/fashion-mnist_train.csv")
    df_test = pd.read_csv(r"../data/fm/fashion-mnist_test.csv")

    train_dataset = FMnistDataSet(df_train, data_transform)
    test_dataset = FMnistDataSet(df_test, data_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=num_works)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_works)
    image, label = next(iter(train_dataloader))
    print(image.shape, label.shape)
    model = FMnistNet()
    model = model.cuda()
    # 指定损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for i in range(epochs + 1):
        train(i)
        val(i)

    # 保存模型
    torch.save(model.state_dict(), r"./model/fm_torch.pth")
    # 转化模型格式(onnx)
    model = FMnistNet().to(device)
    model.load_state_dict(torch.load(r"./model/fm_torch.pth"))
    model.eval()
    input_sample = torch.randn(1, 1, 28, 28).to(device)
    torch.onnx.export(model, input_sample, r"./model/fm_torch.onnx")
    print("Export fm_torch.pth model to onnx model has been successful")
    # 验证 onnx 模
    fm_model = onnxruntime.InferenceSession(r"./model/fm_torch.onnx")
    input_name = fm_model.get_inputs()[0].name
    output_name = fm_model.get_outputs()[0].name
    input_sample = np.random.rand(1, 1, 28, 28)
    input_sample = input_sample.astype(np.float32)
    pre_test = fm_model.run([output_name], {input_name: input_sample})
    print(pre_test, pre_test[0].shape, pre_test[0].argmax(axis=1))

    inference(test_dataset, r"./model/fm_torch.pth")
