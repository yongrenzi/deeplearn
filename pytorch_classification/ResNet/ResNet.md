## ResNet

### 1 ResNet 网络结构

![image1.png](image%2Fimage1.png)

**34-layers模型简图**

![image2.png](image%2Fimage2.png)

### 2 ResNet 的优点

ResNet网络亮点：

- 提出residual结构（残差结构），搭建超深的网络结构（突破1000层）
- 使用**Batch Normalization** 加速训练（丢弃dropout）
- 解决了深层网络梯度消失和梯度爆炸的问题
- 解决了深度网络退化的问题

#### 2.1 残差结构

下图是论文中两种残差结构，左边的残差结构针对的是层数较少的网络，右边的残差结构指的是层数较深的网路。对于一个channel为256的特征矩阵，右侧残差结构需要的参数明显小于左侧残差结构。

![image3.png](image%2Fimage3.png)

另外对于网络中的虚线曲线：

- 浅层网络：

![image4.png](image%2Fimage4.png)

- 深层网络：

![image5.png](image%2Fimage5.png)