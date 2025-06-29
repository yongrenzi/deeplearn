### FCN

论文《Fully Convolutional Networks for Semantic Segmentation》

首个端对端的针对像素级预测的全卷积网络

#### 模型图
![image1.png](image%2Fimage1.png)



膨胀（空洞）卷积：

- 增大感受野
- 保持原输入特征图的W 和 H

![image2.png](image%2Fimage2.png)

![image3.png](image%2Fimage3.png)

空洞卷积最初的提出是为了解决图像分割的问题而提出的,常见的图像分割算法通常使用池化层和卷积层来增加感受野(Receptive Filed),同时也缩小了特征图尺寸(resolution),然后再利用上采样还原图像尺寸,特征图缩小再放大的过程造成了精度上的损失,因此需要一种操作可以在增加感受野的同时保持特征图的尺寸不变,从而代替下采样和上采样操作,在这种需求下,空洞卷积就诞生了

[一文看懂膨胀（空洞）卷积（含代码）_膨胀卷积-CSDN博客](https://blog.csdn.net/qq_46073783/article/details/128383220)

![image4.png](image%2Fimage4.png)