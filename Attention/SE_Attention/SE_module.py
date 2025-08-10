import torch
import torch.nn as nn


class SE_module(nn.Module):
    def __init__(self, in_c, scale=16):
        super(SE_module, self).__init__()
        # 全局平均池化，[H,W,C]---->[1,1,C]
        self.fsq = nn.AdaptiveAvgPool2d((1, 1))
        linear_c = in_c // scale
        self.W1 = nn.Linear(in_c, linear_c)
        self.W2 = nn.Linear(linear_c, in_c)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入特征图
        b, c, h, w = x.size()
        # squeeze操作
        y = self.fsq(x).view(b, c)
        y = self.relu(self.W1(y))
        y = self.sigmoid(self.W2(y))  # 输出[b,c]
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


# input = torch.rand([8, 3, 480, 480])
# se_model = SE_module(in_c=3, scale=3)
# output = se_model(input)
# print(output.shape)
