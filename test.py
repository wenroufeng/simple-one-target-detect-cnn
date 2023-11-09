# import torch
# from train_shape import Residual

# load_dict = torch.load('./shape_models/model_7.pth')
# # print(load_dict)
# model_dict = load_dict.state_dict()
# # print(model_dict)
# print(model_dict.keys())
# print('随机初始化权重第一层：',model_dict['0.0.weight'])

# ========================
import torch
import torch.nn as nn
import math

# 创建一个简单的多分类模型
# model = nn.Sequential(
#     nn.Linear(2, 3),  # 三分类问题，输出三个值
# )

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 创建示例数据
x = torch.tensor([
    [0.3, 0.3, 0.4], 
    [0.3, 0.4, 0.3], 
    [0.1, 0.2, 0.7]
    ], dtype=torch.float32)

y = torch.tensor([2, 1, 0], dtype=torch.long)

# print(x.softmax(1))
p = torch.tensor([-torch.log(x.softmax(1))[i, y[i]] for i in range(3)])
print(p)
print(torch.sum(p) / 3)
# 计算损失
# outputs = model(X)
# print(outputs)
loss = criterion(x, y)

# 在这个简单的示例中，我们的模型和数据非常适合，损失可以为 0。
print("CrossEntropyLoss:", loss.item())

