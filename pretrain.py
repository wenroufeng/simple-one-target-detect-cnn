import numpy as np
import matplotlib.image as imgg
from PIL import Image as pil
import os
import torch.nn as nn
import torch
import torchvision
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms as tt
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import time
from torch.autograd import Variable
import json
import torch.nn.functional as F
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


class Mydata(Dataset):
    def __init__(self, t):
        self.ts = tt.Compose([tt.Resize((300, 300)), tt.ToTensor()])
        self.type_label = {
            'train': './shape_dataset/train.txt',
            'test': './shape_dataset/val.txt',
        }
        with open(self.type_label[t], 'rb') as f:
            self.img_sign = [line.split() for line in f.readlines()]

    def __getitem__(self, idx):
        img = pil.open(f'{self.img_sign[idx][0].decode()}')
        sign = np.array(int(self.img_sign[idx][1]))
        img = self.ts(img)
        return [img, sign]

    def __len__(self):
        return len(self.img_sign)


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)

        self.preprocess = nn.Sequential()

        if use_1x1conv:
            conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
            self.preprocess.add_module('1x1conv', conv3) 

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y += self.preprocess(X)
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals,
                first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*self.resnet_block(64, 128, 2))
        b4 = nn.Sequential(*self.resnet_block(128, 256, 2))
        b5 = nn.Sequential(*self.resnet_block(256, 512, 2))
        self.common_b = nn.Sequential(b1, b2, b3, b4, )
        self.net = nn.Sequential(b5, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 3))

    def forward(self, X):
        Y = self.common_b(X)
        return self.net(Y)

    def resnet_block(self, input_channels, num_channels, num_residuals,
                first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk


def create_data():
    BATCHSIZE = 100
    train_data = Mydata('train')
    train_dataloader = DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True)

    test_data = Mydata('test')
    test_dataloader = DataLoader(dataset=test_data, batch_size=BATCHSIZE, shuffle=True)

    train_data_size = len(train_data)
    test_data_size = len(test_data)

    return {"train_dataloader": train_dataloader, "test_dataloader": test_dataloader, 
            "train_data_size": train_data_size, "test_data_size": test_data_size}


def do_train(train_dataloader):
    net.train()
    loss = None
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = net(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss


def do_test(test_dataloader):
    net.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()

            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    return total_test_loss, total_accuracy


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = create_data()
    train_dataloader = data['train_dataloader']
    test_dataloader = data['test_dataloader']
    train_data_size = data['train_data_size']
    test_data_size = data['test_data_size']

    print("训练数据集的长度为：{}".format(train_data_size))
    print("测试数据集的长度为：{}".format(test_data_size))

    net = ResNet().to(device)
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    # 训练的轮数
    epoch = 20
    os.makedirs('shape_models', exist_ok=True)
    for epoch_witch in range(1, epoch + 1):
        print("-------第 {} 轮训练开始-------".format(epoch_witch))

        # 训练步骤开始
        loss = do_train(train_dataloader)
        print("训练集Loss: {}".format(loss.item()), flush=True)

        total_test_loss, total_accuracy = do_test(test_dataloader)
        print("整体测试集上的Loss: {}".format(total_test_loss), flush=True)
        print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size), flush=True)

        torch.save(net, "shape_models/model_{}.pth".format(epoch_witch))
        print("模型已保存", flush=True)
