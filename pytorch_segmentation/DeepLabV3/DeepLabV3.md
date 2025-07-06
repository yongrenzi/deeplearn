### DeepLab-V1

#### 语义分割任务中存在的问题：

- 下采样导致空间分辨率降低

- 空间不敏感

#### 解决方案：

- 使用膨胀卷积
- fully-connected CRF(Conditional Random Field)

#### 网络优势：

- 速度更快，论文中说是采用了膨胀卷积的原因，但fully-connected CRFs很耗时

- 准确率更高，相比之前最好的网络提升了7.2个点

- 模型结构简单，主要是由DCNNs和CRFs联级构成

#### 模型结构：

![image1.png](image%2Fimage1.png)

![image2.png](image%2Fimage2.png)




### DeepLab-V2

相比v1，v2更换了backbone，由vgg更换为Resnet，引入了一个新的结构ASPP

语义分割出现的问题及解决方案：

1. 分辨率被降低（主要是因为stride>1导致的），一般将Maxpooling层的stride设置为1（如果是通过卷积下采样的，比如resnet，同样是将stride设置为1即可），配合膨胀卷积使用
2. 目标的多尺度问题，最简单的方法是将图像缩放到多个尺度，分别通过网络进行推理，最后将多个结果融合，这样做虽有用，但是计算量太大了，为了解决这个问题，v2中提出了ASPP模块
3. DCNNs的不变性会降低定位精度，通过CREs解决

#### 模型结构

![image3.png](image%2Fimage3.png)

### DeepLab-V3

#### 网络改进：

1. 引入了Multi-grid
2. 改进了ASPP结构
3. 移除CRFs后处理

#### 模型结构

![image4.png](image%2Fimage4.png)

#### 常见的几种多尺度方法

![image5.png](image%2Fimage5.png)

图a 将原图缩放至不同的尺寸，然后提取特征

图b 编码器解码器架构

图c 最后几个下采样的布局设置为1，然后引入膨胀卷积增大感受野V1

图d ASPP结构

![image6.png](image%2Fimage6.png)