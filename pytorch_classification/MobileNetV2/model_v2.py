from torch import nn
import torch


# ch=9,new_ch=8
# ch=12,new_ch=16
# 找到与ch最接近8的new_ch
def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
#         padding = (kernel_size - 1) // 2
#         super(ConvBNReLU, self).__init__(
#             nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups,bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU6(inplace=True)
#         )

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = expand_ratio * in_ch
        self.shortcut = stride == 1 and in_ch == out_ch

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_ch, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_neareat=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        in_ch = _make_divisible(32 * alpha, divisor=round_neareat)
        last_ch = _make_divisible(1280 * alpha, divisor=round_neareat)

        inverted_residual_setting = [
            # t ,c ,n ,s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        features = []
        features.append(ConvBNReLU(3, in_ch, stride=2))

        for t, c, n, s in inverted_residual_setting:
            out_ch = _make_divisible(c * alpha, round_neareat)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(in_ch, out_ch, stride=stride, expand_ratio=t))
                in_ch = out_ch
        # build last several layers
        features.append(ConvBNReLU(in_ch, last_ch, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_ch, num_classes)
        )
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

