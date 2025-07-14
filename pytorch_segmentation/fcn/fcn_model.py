from collections import OrderedDict

from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from backbone import resnet50, resnet101


class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channel = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channel, channels, 1)
        ]
        super(FCNHead, self).__init__(*layers)


class FCN(nn.Module):
    def __init__(self, backbone, classifier, aux_classifier):
        super(FCN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        feature = self.backbone(x)
        # print(f"backbone================================>{self.backbone}")
        result = OrderedDict()
        # print(f"feature================================>{feature}")
        x = feature["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result['out'] = x

        if self.aux_classifier is not None:
            x = feature["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


def fcn_resnet50(aux, num_classes=21, pretrain_backbone=False):
    # 'resnet50_imagenet': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    # 'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth'
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])
    if pretrain_backbone:
        backbone.load_state_dict(torch.load("resnet.pth", map_location='cpu'))

    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {"layer4": "out"}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    aux_classifier = None
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)
    classifier = FCNHead(out_inplanes, num_classes)
    model = FCN(backbone, classifier, aux_classifier)
    return model


def fcn_resnet101(aux, num_classes=21, pretrain_backbone=False):
    # 'resnet101_imagenet': 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
    # 'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth'
    backbone = resnet101(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # 载入resnet101 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet101.pth", map_location='cpu'))

    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = FCNHead(out_inplanes, num_classes)

    model = FCN(backbone, classifier, aux_classifier)

    return model
