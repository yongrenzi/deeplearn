import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Union


class ConvBnRelu(nn.Module):
    def __init__(self, in_c, out_c, kernel: int = 3, dilation: int = 1):
        super(ConvBnRelu, self).__init__()
        padding = kernel // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_c, out_c, kernel, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


# class DownConvBnRelu(nn.Module):
#     def __init__(self, in_c, out_c, kernel: int = 3, dilation: int = 1, flag: bool = True):
#         super(DownConvBnRelu, self).__init__()
#         self.down_flag = flag
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         padding = kernel // 2 if dilation == 1 else dilation
#         self.conv = nn.Conv2d(in_c, out_c, kernel, padding=padding, dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(out_c)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.down_flag:
#             x = self.maxpool(x)
#         return self.relu(self.bn(self.conv(x)))

class DownConvBnRelu(ConvBnRelu):
    def __init__(self, in_c, out_c, kernel: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_c, out_c, kernel, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        return self.relu(self.bn(self.conv(x)))


class UpConvBnRelu(ConvBnRelu):
    def __init__(self, in_c, out_c, kernel: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_c, out_c, kernel, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2], dim=1)
        return self.relu(self.bn(self.conv(x)))


class RSU(nn.Module):
    def __init__(self, height, in_c, mid_c, out_c):
        super(RSU, self).__init__()
        assert height >= 2
        encoder_list = [ConvBnRelu(in_c, out_c), ConvBnRelu(out_c, mid_c)]
        decoder_list = [UpConvBnRelu(mid_c * 2, mid_c, flag=False)]
        for i in range(height - 2):
            encoder_list.append(DownConvBnRelu(mid_c, mid_c))
            decoder_list.append(UpConvBnRelu(mid_c * 2, mid_c if i < height - 3 else out_c))
        encoder_list.append(ConvBnRelu(mid_c, mid_c, dilation=2))
        self.encoder_modules = nn.ModuleList(encoder_list)
        self.decoder_modules = nn.ModuleList(decoder_list)

    def forward(self, x):
        encoder_out = []
        for encoder in self.encoder_modules:
            x = encoder(x)
            encoder_out.append(x)

        x = encoder_out.pop()
        for decoder in self.decoder_modules:
            x2 = encoder_out.pop()
            x = decoder(x, x2)

        return x + encoder_out.pop()


class RSU4F(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super().__init__()
        self.conv1 = ConvBnRelu(in_c, out_c)
        self.encoder_modules = nn.ModuleList([
            ConvBnRelu(out_c, mid_c),
            ConvBnRelu(mid_c, mid_c, dilation=2),
            ConvBnRelu(mid_c, mid_c, dilation=4),
            ConvBnRelu(mid_c, mid_c, dilation=8),
        ])
        self.decoder_modules = nn.ModuleList([
            ConvBnRelu(mid_c * 2, mid_c, dilation=4),
            ConvBnRelu(mid_c * 2, mid_c, dilation=2),
            ConvBnRelu(mid_c * 2, out_c),
        ])

    def forward(self, x):
        x_in = self.conv1(x)
        x = x_in
        encoder_outputs = []
        for encoder in self.encoder_modules:
            x = encoder(x)
            encoder_outputs.append(x)
        x = encoder_outputs.pop()
        for decoder in self.decoder_modules:
            x2 = encoder_outputs.pop()
            x = decoder(torch.cat([x, x2],dim=1))
        return x + x_in


class U2Net(nn.Module):
    def __init__(self, cfg: dict, out_ch: int = 1):
        super().__init__()
        assert "encoder" in cfg
        assert "decoder" in cfg
        self.encoder_num = len(cfg["encoder"])

        encoder_list = []
        side_list = []
        for c in cfg["encoder"]:
            # c:[height,in_ch,mid_ch,out_ch,RSU4F,side]
            assert len(c) == 6
            encoder_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.encoder_modules = nn.ModuleList(encoder_list)

        decoder_list = []
        for c in cfg["decoder"]:
            assert len(c) == 6
            decoder_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.decoder_modules = nn.ModuleList(decoder_list)
        self.side_modules = nn.ModuleList(side_list)
        self.out_conv = nn.Conv2d(self.encoder_num * out_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w = x.shape

        # collect encoder outputs
        encoder_outputs = []
        for i, encoder in enumerate(self.encoder_modules):
            x = encoder(x)
            encoder_outputs.append(x)
            if i != self.encoder_num - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        # collect decoder outputs
        x = encoder_outputs.pop()
        decoder_outputs = [x]
        for decoder in self.decoder_modules:
            x2 = encoder_outputs.pop()
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
            x = decoder(torch.concat([x, x2], dim=1))
            decoder_outputs.insert(0, x)

        # collect side outputs
        side_outputs = []
        for side in self.side_modules:
            x = decoder_outputs.pop()
            x = F.interpolate(side(x), size=[h, w], mode='bilinear', align_corners=False)
            side_outputs.insert(0, x)

        x = self.out_conv(torch.concat(side_outputs, dim=1))

        if self.training:

            return [x] + side_outputs
        else:
            return torch.sigmoid(x)


def u2net_full(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encoder": [[7, 3, 32, 64, False, False],  # En1
                   [6, 64, 32, 128, False, False],  # En2
                   [5, 128, 64, 256, False, False],  # En3
                   [4, 256, 128, 512, False, False],  # En4
                   [4, 512, 256, 512, True, False],  # En5
                   [4, 512, 256, 512, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decoder": [[4, 1024, 256, 512, True, True],  # De5
                   [4, 1024, 128, 256, False, True],  # De4
                   [5, 512, 64, 128, False, True],  # De3
                   [6, 256, 32, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }

    return U2Net(cfg, out_ch)


def u2net_lite(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encoder": [[7, 3, 16, 64, False, False],  # En1
                   [6, 64, 16, 64, False, False],  # En2
                   [5, 64, 16, 64, False, False],  # En3
                   [4, 64, 16, 64, False, False],  # En4
                   [4, 64, 16, 64, True, False],  # En5
                   [4, 64, 16, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decoder": [[4, 128, 16, 64, True, True],  # De5
                   [4, 128, 16, 64, False, True],  # De4
                   [5, 128, 16, 64, False, True],  # De3
                   [6, 128, 16, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }

    return U2Net(cfg, out_ch)


# input = torch.rand([1, 3, 480, 480])
#
# u2net_full = u2net_full()
# out1 = u2net_full(input)
# u2net_lite = u2net_lite()
# out2 = u2net_lite(input)
#
# print(out1)
# print(out2.shape)
