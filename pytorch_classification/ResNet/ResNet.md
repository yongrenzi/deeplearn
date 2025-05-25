## ResNet

### 1 ResNet 网络结构

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250521155107617.png" alt="image-20250521155107617" style="zoom:60%;" />

**34-layers模型简图**

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250521154944774.png" alt="image-20250521154944774" style="zoom:60%;" />



### 2 ResNet 的优点

ResNet网络亮点：

- 提出residual结构（残差结构），搭建超深的网络结构（突破1000层）
- 使用**Batch Normalization** 加速训练（丢弃dropout）
- 解决了深层网络梯度消失和梯度爆炸的问题
- 解决了深度网络退化的问题

#### 2.1 残差结构

下图是论文中两种残差结构，左边的残差结构针对的是层数较少的网络，右边的残差结构指的是层数较深的网路。对于一个channel为256的特征矩阵，右侧残差结构需要的参数明显小于左侧残差结构。

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250521160139743.png" alt="image-20250521160139743" style="zoom:60%;" />

另外对于网络中的虚线曲线：

- 浅层网络：

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250521160821067.png" alt="image-20250521160821067" style="zoom:60%;" />

- 深层网络：

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250521160859913.png" alt="image-20250521160859913" style="zoom:60%;" />