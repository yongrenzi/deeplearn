### Swin Transformer

论文名称：Swin Transformer:Hierarchical Vision Transformer using Shifted Windows

Swin Transformer使用了类似卷积神经网络中的层次化构建方法，比如特征图尺寸中有对图像下采样4倍的，8倍的以及16倍的，这样的backbone有助于在此基础上构建目标检测，实例分割等任务

在Swin Transformer中使用了Windows Multi-Head Self-Attention(W-MSA)的概念，比如在下图的4倍下采样和8倍下采样中，将特征图划分成了多个不相交的区域（Window），并且Multi-Head Self-Attention只在每个窗口（Window）内进行。相对于Vision Transformer中直接对整个（Global）特征图进行Multi-Head Self-Attention，这样做的目的是能够减少计算量的，尤其是在浅层特征图很大的时候。这样做虽然减少了计算量但也会隔绝不同窗口之间的信息传递，所以在论文中作者又提出了 Shifted Windows Multi-Head Self-Attention(SW-MSA)的概念，通过此方法能够让信息在相邻的窗口中进行传递。

##### Swin-Transformer与Vision-Transformer的不同

![image1.png](image%2Fimage1.png)


- Swin采用层次化构建方法，特征图尺寸对图形下采样4倍，8倍，16倍，这样有助于特征图特征提取，而Vision从一开始就是下采样16倍率
- Swin中将特征图划分成了多个不想交的区域`Windows Multi-Head Self-Attention(W-MSA)`，并且Multi-Head-Self-Attention都是在每个窗口中进行，窗口之间不进行，这样相较于Vision减少了计算量。但是这样会隔绝不同窗口之间的信息传递，所以作者提出了 `Shifted Windows Multi-Head Self-Attention(SW-MSA)`，通过此方法能够让信息在相邻的窗口之间进行传递。

### 模型图
![image2.png](image%2Fimage2.png)


- 首先将图片输入到Patch Partition模块中进行分块，即每4x4相邻的像素为一个Patch，然后在channel方向展平（flatten）。假设输入的是RGB三通道图片，那么每个patch就有4x4=16个像素，然后每个像素有R、G、B三个值所以展平后是16x3=48，所以通过Patch Partition后图像shape由 `[H, W, 3]`变成了 `[H/4, W/4, 48]`。然后在通过Linear Embeding层对每个像素的channel数据做线性变换，由48变成C，即图像shape再由 `[H/4, W/4, 48]`变成了 `[H/4, W/4, C]`。其实在源码中Patch Partition和Linear Embeding就是直接通过一个卷积层实现的，和之前Vision Transformer中讲的 Embedding层结构一模一样。
- 然后就是通过四个Stage构建不同大小的特征图，除了Stage1中先通过一个Linear Embeding层外，剩下三个stage都是先通过一个Patch Merging层进行下采样（后面会细讲）。然后都是重复堆叠Swin Transformer Block注意这里的Block其实有两种结构，如图(b)中所示，这两种结构的不同之处仅在于一个使用了W-MSA结构，一个使用了SW-MSA结构。而且这两个结构是成对使用的，先使用一个W-MSA结构再使用一个SW-MSA结构。所以你会发现堆叠Swin Transformer Block的次数都是偶数（因为成对使用）。
- 最后对于分类网络，后面还会接上一个Layer Norm层、全局池化层以及全连接层得到最终输出。图中没有画，但源码中是这样做的。

上述`Patch Partition`和`Linear Embedding`的联合作用与`Patch Merging`相同
![image3.png](image%2Fimage3.png)

#### Patch Merging
![image4.png](image%2Fimage4.png)

#### W-MSA



#### SW-MSA