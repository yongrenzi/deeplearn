### vision Transformer

#### 模型图

![image1.png](image%2Fimage1.png)

![image2.png](image%2Fimage2.png)

#### Embedding层

在Embedding层中，直接通过一个卷积层来实现，以ViT-B/16为例，使用卷积核大小为16 x 16 ，stride为16，卷积核个数为768， [224,224,3]--->[14,14,768]--->[196,768]，

拼接[class] token之后 [196,768]--->[197,768]

然后叠加position Embedding  [197,768]--->[197,768]