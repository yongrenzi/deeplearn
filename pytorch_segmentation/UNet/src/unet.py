import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConvBNRelu(nn.Module):
    def __init__(self, in_c, out_c, mid_c=None):
        if mid_c is None:
            mid_c = out_c
        super(DoubleConvBNRelu, self).__init__()
        self.conv1 = nn.Conv2d(in_c, mid_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super(Down, self).__init__()
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv = DoubleConvBNRelu(in_c, out_c)

    def forward(self, x):
        return self.maxpool(self.conv(x))


class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dobleconvbnrelu = DoubleConvBNRelu(in_c, out_c, in_c // 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.dobleconvbnrelu(x)
        return x


class UNet(nn.Module):
    def __init__(self, base_c: int = 3, num_classes: int = 2):
        super(UNet, self).__init__()
        self.doubleconv = DoubleConvBNRelu(base_c, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)

        self.conv1 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Down
        x1 = self.doubleconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Up
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.conv1(x)

        return {"out": logits}
        # return logits


# input = torch.rand([1, 3, 480, 480])
#
# unet = UNet()
# out = unet(input)
# print(out.shape)
