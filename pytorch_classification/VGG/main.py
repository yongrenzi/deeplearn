import numpy as np
import torch

from model import VGG16

if __name__ == '__main__':
    random_data = np.random.rand(1, 3, 224, 224)  # numpy数组
    random_data_tensor = torch.from_numpy(random_data.astype(np.float32))  # 将numpy数组转化为tensor类型

    print("输入数据形状：", random_data_tensor.shape)
    print("输入数据：", random_data_tensor)
    vgg16 = VGG16()
    output = vgg16(random_data_tensor)
    print("输出数据形状：", output.shape)
    print("输出数据：", output)
