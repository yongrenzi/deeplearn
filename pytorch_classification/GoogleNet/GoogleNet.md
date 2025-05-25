## GoogleNet

### 1 GoogleNet 模型结构

![image1.png](image%2Fimage1.png)


![image2.png](image%2Fimage2.png)

![image3png](image%2Fimage3.png)
### 2 GoogleNet 网络的亮点

- 引入了Inception结构（融合不同尺度的特征信息）
- 使用1x1的卷积核进行降维以及映射处理
- 添加两个辅助分类器帮助训练，一共有三个输出层
- 丢弃全连接层，使用平均池化层（大大减少模型的参数）