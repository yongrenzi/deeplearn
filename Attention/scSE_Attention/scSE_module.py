import torch
import torch.nn as nn


class cSE_module(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(cSE_module, self).__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.cSE(x)


class sSE_module(nn.Module):
    def __init__(self, in_channels):
        super(sSE_module, self).__init__()
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.sSE(x)+x


class scSE_module(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(scSE_module, self).__init__()
        self.cSE = cSE_module(in_channels, reduction)
        self.sSE = sSE_module(in_channels)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        return self.alpha * self.cSE(x) + (1 - self.alpha) * self.sSE(x)

# input = torch.rand([8, 3, 480, 480])

# sSE_model = sSE_module(in_channels=3)
# cSE_model = cSE_module(in_channels=3,reduction=3)
# scSE_model = scSE_module(in_channels=3,reduction=3)

# sSE_out = sSE_model(input)
# cSE_out = cSE_model(input)
# scSE_out = scSE_model(input)

# print(sSE_out.shape)
# print(cSE_out.shape)
# print(scSE_out.shape)
