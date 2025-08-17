import torch
import torch.nn as nn


# 通道注意力，保留通道的特征信息
class Channel_Atten(nn.Sequential):
    def __init__(self, in_c, reduction_ratio=16):
        super(Channel_Atten, self).__init__()
        self.inter_channels = in_c // reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.MaxPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Conv2d(in_c, self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            nn.Conv2d(self.inter_channels, in_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_c)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # [b,c,h,w]
        avg_y = self.avg_pool(x)  # [b,c,1,1]
        max_y = self.max_pool(x)  # [b,c,1,1]
        mlp_avg = self.mlp(avg_y)
        mlp_max = self.mlp(max_y)
        f = mlp_max + mlp_avg
        out_c = self.sigmoid(f)
        return out_c


class Spatial_Atten(nn.Module):
    def __init__(self):
        super(Spatial_Atten, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿着通道方向
        avg_y = torch.mean(x, dim=1, keepdim=True)
        max_y = torch.max(x, dim=1, keepdim=True)[0]
        f = torch.cat([avg_y, max_y], dim=1)
        out = self.sigmoid(self.conv(f))
        return out


class CBAM_module(nn.Module):
    def __init__(self, in_c, reduction_ratio):
        super(CBAM_module, self).__init__()
        self.channel_atten = Channel_Atten(in_c, reduction_ratio)
        self.spatial_atten = Spatial_Atten()

    def forward(self, x):
        # 通道注意力
        channel = self.channel_atten(x)
        x = channel * x

        # 空间注意力
        spatial = self.spatial_atten(x)
        x = spatial * x
        return x


# input = torch.rand([8, 3, 480, 480])
# cbam = CBAM_module(3, 3)
# output = cbam(input)
# print(output.shape)
