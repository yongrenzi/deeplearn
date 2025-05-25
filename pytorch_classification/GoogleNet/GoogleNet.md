## GoogleNet

### 1 GoogleNet 模型结构

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250524164819458.png" alt="image-20250524164819458" style="zoom:60%;" />



<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250524160853967.png" alt="image-20250524160853967" style="zoom:60%;" />



<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250524165016631.png" alt="image-20250524165016631" style="zoom:60%;" />

### 2 GoogleNet 网络的亮点

- 引入了Inception结构（融合不同尺度的特征信息）
- 使用1x1的卷积核进行降维以及映射处理
- 添加两个辅助分类器帮助训练，一共有三个输出层
- 丢弃全连接层，使用平均池化层（大大减少模型的参数）