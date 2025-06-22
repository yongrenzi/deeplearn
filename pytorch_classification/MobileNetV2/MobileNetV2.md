### MobileNet v1

#### 模型结构

![image1.png](image%2Fimage1.png)

传统卷积神经网络，内存需求大、运算量大导致无法在移动设备以及嵌入式设备上运行

#### 网络亮点：

- DW（Depthwise Conv）卷积

传统卷积：卷积核的channel等于输入特征图的channel，输出特征值的channel等于卷积核个数

DW卷积：卷积核channel为1，输入特征图channel=卷积核个数=输出特征图channel

PW卷积：和传统卷积一样，只不过卷积核大小为1

Depthwise Separable Conv：先使用DW卷积，然后再使用PW卷积（因为DW卷积每个输出通道仅由输入的一个通道得来，缺乏了输入通道之间的信息交互，所以通常在后面加一个1 x 1的卷积来实现通道之间的信息交换，就是PW卷积）

![image2.png](image%2Fimage2.png)

- 增加超参数

---

### MobileNet v2

实验发现在MobileNetv1中，深度卷积核的参数较多为0，也就是其卷积核没有发挥提取特征作用。那么作者先通过1\*1卷积将维度上升，再使用深度卷积，深度卷积的输入输出通道数更高，就能够提取更多的信息。相较于mobilenetv1，准确率更高，参数量更少。

论文名称：MobileNetV2:Inverted Residuals and Linear Bottlenecks

### 模型结构

![image3.png](image%2Fimage3.png)

其中，t为扩展因子，c是输出特征矩阵深度channel，n为bottleneck的重复次数，s为步距（针对第一层，其他为1）

#### 网络亮点

- 采用倒残差结构（Inverted Residuals）
- 
![image4.png](image%2Fimage4.png)

![image5.png](image%2Fimage5.png)

![image6.png](image%2Fimage6.png)

**注意**：当stride=1且输入特征矩阵与输出特征矩阵shape相同时才有shortcut连接

使用到的激活函数为Relu6，公式如下：
$$
\mathrm{Re}Lu6\left(x \right)=\min \left(\max \left(x,0 \right),6 \right)
$$

- Linear Bottlenecks

ReLU激活函数对低维特征信息照成大量损失

---

### MobileNet v3

#### 模型结构
![image7.png](image%2Fimage7.png)

#### 网络亮点

- 更新block（bneck）

加入了SE注意力机制，另外更新了激活函数
![image8.png](image%2Fimage8.png)

- 使用NAS搜索参数
- 重新设计耗时层结构

减少第一个卷积层个数，精简last stage
![image9.png](image%2Fimage9.png)

![image10.png](image%2Fimage10.png)
