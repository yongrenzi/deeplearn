import numpy as np
import torch
from torch import nn


# 自己实现
def ln_process(feature):
    feature_shape = feature.shape
    # 遍历batch通道
    for i in range(feature_shape[0]):
        # 计算第i channel 的均值和方差
        feature_t = feature[i, :, :, :]
        mean_t = feature_t.mean()
        # 总体标准差
        std_t1 = feature_t.std(unbiased=False)

        # bn_process
        feature[i, :, :, :] = (feature[i, :, :, :] - mean_t) / np.sqrt(std_t1 ** 2 + 1e-5)

    return feature


# [batch,channel,height,weight]
feature = torch.randn(2, 2, 2, 2)


feature1 = ln_process(feature)

print("手动实现的：", feature1)
# 官方实现
bn = nn.LayerNorm(
    normalized_shape=[2, 2, 2],  # 归一化维度（C, H, W）
    eps=1e-5,  # 与你的代码中 1e-5 对应
    elementwise_affine=False  # 禁用可学习的缩放和偏移参数（你的代码未实现此功能）
)
feature2 = bn(feature)
print("官方实现的：", feature2)
