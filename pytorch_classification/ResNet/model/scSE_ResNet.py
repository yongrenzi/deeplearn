import torch
import torch.nn as nn
from Attention.scSE_Attention import sSE_module, scSE_module, cSE_module


# 18-layers,34-layers
class BasicBlock(nn.Module):
    expansion = 1

    # in_channel=3,out_channel=64
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, Attention="scSE"):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if Attention == "scSE":
            self.attention = scSE_module(out_channel)
        elif Attention == "sSE":
            self.attention = sSE_module(out_channel)
        else:
            self.attention = cSE_module(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Attention_Module
        atten_out = self.attention(x)
        out = atten_out + identity
        out = self.relu(out)

        return out


# 50-layers,101-layers,152-layers
class Bottleneck(nn.Module):
    expansion = 4

    # in_channel=3,out_channel=64,stride=2主要对应的是conv3，conv4，conv5的第一层
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, padding=1, stride=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = x + identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    # 这里的in_channel=64
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, Attention="scSE"):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Attention = Attention
        self.layer1 = self._make_layers(block, 64, blocks_num[0])
        self.layer2 = self._make_layers(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layers(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layers(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layers(self, block, channel, block_num, stride=1):
        downsample = None
        # 18层和34层的第一层会跳过这个结构
        # 使用到这个结构的层：18层和34层的conv3x，conv4x，conv5x，50层，101层和152层的conv2x，conv3x，conv4x，conv5x
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []

        # 先把conv2x加入到；layers中去
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride, Attention=self.Attention))
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, Attention=self.Attention))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


def scSE_resnet34(num_classes=1000, include_top=True, Attention="scSE"):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, Attention=Attention)


def scSE_resnet50(num_classes=1000, include_top=True, Attention="scSE"):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, Attention=Attention)
