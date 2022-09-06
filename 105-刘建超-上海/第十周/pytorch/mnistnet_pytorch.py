#!/usr/bin/env python
# coding:utf-8

import torch
import torchvision


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()  # 初始化网络结构，首先找到MnistNet的父类Model，然后运行父类Model的__init__初始化函数
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 返回一个有相同数据但不同大小的Tensor,reshape成28*28列，行不确定
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.softmax(self.fc3(x), dim=1)
        return x


def mnist_load_data():
    transform = torchvision.transforms.Compose(  # 图片变换
        [torchvision.transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
         torchvision.transforms.Normalize([0, ], [1, ])])  # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
    # 通过torchvision下载图像mnist
    trainset=torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    # 数据加载器，结合了数据集和取样器，并且可以提供多个线程处理数据集。
    # trainloader=torch.utils.data.DataLoader(dataset=trainset,batch_size=32,shuffle=True,num_works=2)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=32, shuffle=True)
    # 通过torchvision下载图像mnist
    testset=torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    # 数据加载器，结合了数据集和取样器，并且可以提供多个线程处理数据集。
    # testloader=torch.utils.data.DataLoader(dataset=testset,batch_size=32,shuffle=True,num_works=2)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=32, shuffle=True)
    return trainloader,testloader

class Model:
    def __init__(self,net,cost,optimist):
        self.net=net
        self.cost=self.creste_cost(cost)
        self.optimizer=self.create_optimizer(optimist)
        # pass

    def creste_cost(self,cost):
        support_cost={
            "CROSS_ENTROPY":torch.nn.CrossEntropyLoss(),
            "MSE":torch.nn.MSELoss()
        }
        return support_cost[cost]

    def create_optimizer(self,optimist,**rests):
        support_optim={
            "SGD":torch.optim.SGD(params=self.net.parameters(),lr=0.1,**rests),
            "ADAM":torch.optim.Adam(self.net.parameters(),lr=0.01,**rests),
            "RMSP":torch.optim.RMSprop(params=self.net.parameters(),lr=0.001,**rests)
        }
        return support_optim[optimist]

    def train(self,train_loader,epoches=3):
        for epoch in range(epoches):
            running_loss=0.0
            for i,data in enumerate(train_loader,start=0):  #获得索引和值
                inputs,labels=data
                self.optimizer.zero_grad()  #清空上一次的梯度记录
                outputs=self.net(inputs)    #前向传播
                loss=self.cost(outputs,labels)
                loss.backward() #反向传播,计算梯度
                self.optimizer.step()   #执行一次优化步骤，更新参数的值
                running_loss+=loss.item()  #损失
                if i%100==0:
                    print("[epoch %d,%.2f%%]loss: %.3f"
                          %(epoch+1,(i+1)*100./len(train_loader),running_loss/100))
                    running_loss=0.0
        print("Finished Training")

    def evaluate(self,test_loader):
        print("evaluating ...")
        correct=0
        total=0
        with torch.no_grad():   #推理预测不计算梯度
            for data in test_loader:
                images,labels=data
                outputs=self.net(images)    #前向传播
                predicted=torch.argmax(input=outputs,dim=1)  #返回指定维度最大值的序号
                total+=labels.size(0)   # 更新测试图片的数量
                correct+=(predicted==labels).sum().item()   # 更新正确分类的图片的数量
                # correct += (predicted == labels).sum()
        print("Accuracy of the network on the test images: %d %%"% (100*correct/total))

if __name__=="__main__":
    net=MnistNet()
    model=Model(net,"CROSS_ENTROPY","RMSP")
    train_loader,test_loader=mnist_load_data()
    model.train(train_loader,epoches=3)
    model.evaluate(test_loader)
