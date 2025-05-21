import numpy as np
import torch
from torch import nn


# 自己实现
def bn_process(feature, mean, var):
    feature_shape = feature.shape
    # 遍历channel通道
    for i in range(feature_shape[1]):
        # 计算第i channel 的均值和方差
        feature_t = feature[:, i, :, :]
        mean_t = feature_t.mean()
        # 总体标准差
        std_t1 = feature_t.std(unbiased=False)
        # 样本标准差
        std_t2 = feature_t.std(unbiased=True)

        # bn_process
        feature[:, i, :, :] = (feature[:, i, :, :] - mean_t) / np.sqrt(std_t1 ** 2 + 1e-5)
        mean[i] = mean[i] * 0.9 + mean_t * 0.1
        var[i] = var[i] * 0.9 + (std_t2 ** 2) * 0.1
    return feature


# [batch,channel,height,weight]
feature = torch.randn(2, 2, 2, 2)

# 初始化统计均值和方差
calculate_mean = [0.0, 0.0]
calculate_var = [1.0, 1.0]

feature1 = bn_process(feature, calculate_mean, calculate_var)

print("手动实现的：", feature1)
# 官方实现
bn = nn.BatchNorm2d(2, eps=1e-5)
feature2 = bn(feature)
print("官方实现的：", feature2)
