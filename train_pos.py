import numpy as np
import matplotlib.image as imgg
from PIL import Image as pil
import os
import torch.nn as nn
import torch
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
            'train': './dataset/train.txt',
            'test': './dataset/val.txt',
        }
        with open(self.type_label[t], 'rb') as f:
            self.img_sign = [line.split() for line in f.readlines()]

    def __getitem__(self, idx):
        img = pil.open(f'{self.img_sign[idx][0].decode()}')
        sign = np.array(int(self.img_sign[idx][1]))
        left = np.array(int(self.img_sign[idx][2]))
        top = np.array(int(self.img_sign[idx][3]))
        right = np.array(int(self.img_sign[idx][4]))
        bottom = np.array(int(self.img_sign[idx][5]))
        img = self.ts(img)
        return [img, sign, left, top, right, bottom]

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

        self.common_b = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), )

        self.left = nn.Sequential(nn.Linear(512, 21))
        self.top = nn.Sequential(nn.Linear(512, 21))
        self.right = nn.Sequential(nn.Linear(512, 21))
        self.bottom = nn.Sequential(nn.Linear(512, 21))

        self.classfication = nn.Sequential(nn.Linear(21, 2))
    
    def forward(self, X):
        Y = self.common_b(X)
        left = self.left(Y)
        return left, self.top(Y), self.right(Y), self.bottom(Y), self.classfication(left)

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
    loss1, loss2, loss3, loss0, total_loss = None, None, None, None, None
    for data in train_dataloader:
        img, sign, left, top, right, bottom = data
        img = img.to(device)
        sign = sign.to(device)
        left = left.to(device)
        top = top.to(device)
        right = right.to(device)
        bottom = bottom.to(device)
        outputs = net(img)

        loss0 = loss_fn(outputs[0], left)
        loss1 = loss_fn(outputs[1], top)
        loss2 = loss_fn(outputs[2], right)
        loss3 = loss_fn(outputs[3], bottom)
        loss4 = loss_fn(outputs[4], sign)
        total_loss = loss0 + loss1 + loss2 + loss3 + loss4
        # 优化器优化模型
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    return total_loss


def do_test(test_dataloader):
    net.eval()
    total_left_test_loss = 0
    total_top_test_loss = 0
    total_right_test_loss = 0
    total_bottom_test_loss = 0
    total_sign_test_loss = 0

    total_accuracy_left = 0
    total_accuracy_top = 0
    total_accuracy_right = 0
    total_accuracy_bottom = 0
    total_accuracy_sign = 0
    with torch.no_grad():
        for data in test_dataloader:
            img, sign, left, top, right, bottom = data
            img = img.to(device)
            sign = sign.to(device)
            left = left.to(device)
            top = top.to(device)
            right = right.to(device)
            bottom = bottom.to(device)
            outputs = net(img)
            loss0 = loss_fn(outputs[0], left)
            loss1 = loss_fn(outputs[1], top)
            loss2 = loss_fn(outputs[2], right)
            loss3 = loss_fn(outputs[3], bottom)
            loss4 = loss_fn(outputs[4], sign)
            total_left_test_loss = total_left_test_loss + loss0.item()
            total_top_test_loss = total_top_test_loss + loss1.item()
            total_right_test_loss = total_right_test_loss + loss2.item()
            total_bottom_test_loss = total_bottom_test_loss + loss3.item()
            total_sign_test_loss = total_sign_test_loss + loss4.item()

            accuracy0 = (outputs[0].argmax(1) == left).sum()
            accuracy1 = (outputs[1].argmax(1) == top).sum()
            accuracy2 = (outputs[2].argmax(1) == right).sum()
            accuracy3 = (outputs[3].argmax(1) == bottom).sum()
            accuracy4 = (outputs[4].argmax(1) == sign).sum()

            total_accuracy_left += accuracy0
            total_accuracy_top += accuracy1
            total_accuracy_right += accuracy2
            total_accuracy_bottom += accuracy3
            total_accuracy_sign += accuracy4

    return (total_left_test_loss, total_top_test_loss, total_right_test_loss, total_bottom_test_loss, total_sign_test_loss,
            total_accuracy_left, total_accuracy_top, total_accuracy_right, total_accuracy_bottom, total_accuracy_sign)
    

if __name__ == '__main__':
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

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
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # 训练的轮数
    epoch = 50
    os.makedirs('models', exist_ok=True)

    for epoch_witch in range(1, epoch + 1):
        print("-------第 {} 轮训练开始-------".format(epoch_witch))

        # 训练步骤开始
        loss = do_train(train_dataloader)
        print("训练集Loss: {}".format(loss.item()))

        total_left_test_loss, total_top_test_loss, total_right_test_loss, total_bottom_test_loss, total_sign_test_loss,\
        total_accuracy_left, total_accuracy_top, total_accuracy_right, total_accuracy_bottom, total_accuracy_sign = do_test(test_dataloader)

        print("整体测试集上的Loss_left: {}，Loss_top: {}，Loss_right: {}，Loss_bottom: {}，Loss_sign: {}".format(
            total_left_test_loss, total_top_test_loss, total_right_test_loss, total_bottom_test_loss, total_sign_test_loss))
        print("整体测试集上的 left正确率: {}, top正确率: {}, right正确率: {}, bottom正确率: {}, sign正确率: {}".format(
            total_accuracy_left / test_data_size, total_accuracy_top / test_data_size, 
            total_accuracy_right / test_data_size, total_accuracy_bottom / test_data_size, 
            total_accuracy_sign / test_data_size
            ))
        
        torch.save(net, "models/model_{}.pth".format(epoch_witch))
        print("模型已保存", flush=True)
