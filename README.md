## 前言
&emsp;&emsp;CV岗面试中经常会问到一些经典神经网络问题，其中具备划时代卷积神经网络VGG为“典中典”,参考面试题如下所示：

面试官：VGG使用3x3卷积核的优势是什么?\
参考答案：2个x3的卷积核串联和5x5的卷积核有相同的感知野，前者**拥有更少的参数**。多个3x3的卷积核比一个较大尺寸的卷积核有更多层的非线性函数**增加了非线性表达,使判决函数更具有判决性**。

&emsp;&emsp; 在这里我将以实验数据来证实面试者的回答

## 实验准备
&emsp;&emsp;  为了确保实验数据的准确性，实验硬件设施保持一致，在同一台电脑上进行的实验，笔者实验数据如下表所述：

| 数据属性 | 数据量 |  | 网络参数 | 参数值 |
| --- | --- |---| --- | --- |
| 图像类别数 | 10 | | Lr | 0.01 |
| 图像通道数 | 1 | | Train_num:Test_num | 350:150 |
| 图像像素 | 20 x 20 | | maxEpochs | 50 |
| 单类图像数 | 500 | | ValidationFrequency | 5 |
| 数据属性 | 数据量 | |  优化器 | sgdm |

&emsp;&emsp; **这里采用LeNet-5作为母版结构进行实验：**

&emsp;&emsp; 在VGG问世以前网络结构基本是LeNet-5、AletNet和ZFNet，然而AletNet和ZFNet结构基本相似，可以认为一致，但他们的卷积核大小为11x11 和3x3混合实验使用，这里不利于做实验数据比对，无法证实：2个**3x3**的卷积核串联和**5x5**的卷积核的差异。

&emsp;&emsp;  实验使用LeNet-5作母版的话，结构简单便于开发，不会出现母版不需要更改就出现过拟合现象。

&emsp;&emsp; **为了实验更加具备说服力，避免异常情况出现，每次实验数据为实验三次后取的平均值。**

## 网络搭建
&emsp;&emsp; 由于实验CNN母版是LeNet-5,在对该网络更改的时也将是LeNet-5中 **convolution2dLayer**的卷积核大小5x5替换为两个 **convolution2dLayer**的3x3卷积核。其结构如下：

**LeNet-5网络结构**
```
layers = [
    imageInputLayer([20 20 1],"Name","imageinput")
    convolution2dLayer([5 5],6,"Name","conv_1","Padding",[1 1 1 1],"Stride",[4 4])
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same","Stride",[3 3])
    convolution2dLayer([5 5],16,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same","Stride",[2 2])
    convolution2dLayer([5 5],120,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")
    fullyConnectedLayer(10,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
```
    
**改LeNet-5网络结构**
```
layers = [
    imageInputLayer([20 20 1],"Name","imageinput")
    groupedConvolution2dLayer([3 3],6,1,"Name","conv1","BiasLearnRateFactor",4,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same","Stride",[3 3])
    groupedConvolution2dLayer([3 3],16,1,"Name","conv2","BiasLearnRateFactor",4,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same","Stride",[2 2])
    groupedConvolution2dLayer([3 3],120,1,"Name","conv3","BiasLearnRateFactor",4,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")
    fullyConnectedLayer(10,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
```
&emsp;&emsp;细心的朋友会发现，我这边改变的仅仅是卷积核的大小，其他层例如池化核激活函数之类的并没有改变，就连卷积层中的Padding我也同LeNet-5保持一致

## 实验结果：

**改LeNet-5第一轮实验：**
重要实验结果：
1. 验证准确度：97.27% 
2. 历时：1分50秒(110秒)
![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d28fb1eb81554b6da68edfaca1afda8f~tplv-k3u1fbpfcp-watermark.image?)

**改LeNet-5第二轮实验：**
重要实验结果：
1. 验证准确度：96.53% 
2. 历时：1分54秒（114秒）
![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/efce649486f740879fa1681525036b66~tplv-k3u1fbpfcp-watermark.image?)

**改LeNet-5第三轮实验：**
重要实验结果：
1. 验证准确度：95.80% 
2. 历时：1分52秒（112秒）
![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/036b81e3ffbc44368ebfaa9483933dcd~tplv-k3u1fbpfcp-watermark.image?)

**LeNet-5第一轮实验：**
重要实验结果：
1. 验证准确度：83.67% 
2. 历时：1分52秒（112秒）
![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c642ae2154d84583a7cbf19818725f3c~tplv-k3u1fbpfcp-watermark.image?)

**LeNet-5第二轮实验：**
重要实验结果：
1. 验证准确度：84.80% 
2. 历时：1分54秒（114秒）
![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4ba73aeefb3e4ce2a1106d4614acb1c2~tplv-k3u1fbpfcp-watermark.image?)

**LeNet-5第三轮实验：**
重要实验结果：
1. 验证准确度：85.60% 
2. 历时：1分52秒（112秒）
![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/9fc66f64b51f4188b27121f9e0723a6f~tplv-k3u1fbpfcp-watermark.image?)

## 结果分析

综合上述可得：\
耗时方面：&emsp;&emsp;&emsp;LeNet-5（112.0）  **>** 改LeNet-5(112.6)\
验证准确度方面：LeNet-5（84.69）**<** 改LeNet-5(96.53%)

&emsp;&emsp;我们可以通过耗时来说明：2个3x3的卷积核串联比单个5x5的卷积核拥有更少的拥有更少的参数；在只有卷积核大小不同的情况能够影响网络计算耗时的也就是内部计算参数了。

&emsp;&emsp;同理可以通过验证准确度易得2个3x3的卷积核串联比单个5x5的卷积核增加了非线性表达,使判决函数更具有判决性

