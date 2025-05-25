## AlexNet
### 1 AlexNet 网络结构

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250522135554710.png" alt="image-20250522135554710" style="zoom:60%;" />

```py
conv1  224 x 224 x 3   kernel_size=11   stride=4    padding=[1,2]     55 x 55 x 96
pool1  55 x 55 x 96    kernel_size=3    stride=2                      27 x 27 x 96
conv2  27 x 27 x 96    kernel_size=5    stride=1    padding=[2,2]     27 x 27 x 256
pool2  27 x 27 x 256   kernel_size=3    stride=2                      13 x 13 x 256
conv3  13 x 13 x 256   kernel_size=3    stride=1    padding=1         13 x 13 x 384
conv4  13 x 13 x 384   kernel_size=3    stride=1    padding=1         13 x 13 x 384
conv5  13 x 13 x 384   kernel_size=3    stride=1    padding=1         13 x 13 x 256
pool3  13 x 13 x 256   kernel_size=3    stride=2                       6 x 6 x 256
```
<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250522171920961.png" alt="image-20250522171920961" style="zoom:60%;" />

### 2 AlexNet 的优点

- 首次利用GPU进行网络加速训练
- 使用了RELU激活函数，而不是传统的sigmoid 激活函数以及Tanh 激活函数
- 使用了LRN局部响应归一化
- 在全连接层的前两层中使用了Dropout 随机失活神经元操作，以减少过拟合