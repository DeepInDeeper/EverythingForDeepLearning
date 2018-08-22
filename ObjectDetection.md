#### [DL目标检测技术演进（通俗易懂，看一遍能记一辈子！）][Title-0000-01] + 七月在线实验室
Object Detection 主要解决两个问题：位置+分类，即表述为哪里有哪种物体。  
**目前学术和工业界的目标检测方法分为3类**
1. 传统的目标检测算法：`Cascade + HOG/DPM +Haar/SVM`等以及相关改进、融合以及优化
2. 候选窗 + 深度学习分类：`R-CNN(Selective Search + CNN +SVM)`,`SPP-net(ROI Pooling)`,`Fast R-CNN(Selective Search + CNN + ROI)`,`Faster R-CNN(RPN + CNN + ROI)`,`R-FCN`等系列方法；
3. 基于深度学习的回归方法：`YOLO/SSD/DenseBox等`,`结合RNN的RRC detection`,`结合DPM的Deformable CNN等`

几个典型的算法概要：  

![img][img-0000-0101]

传统的检测算法流程：  
1. 选择区域(传统方式主要通过穷举策略，即通过滑窗模式，设置不同大小，长宽比进行遍历)
2. 特征提取(SIFT、HOG等，但对于形态，光照，背景等复杂问题常表现为鲁棒性差)
3. 分类器（主要有SVM、Adaboost等）

对于一个物体检测任务而言，可以分解为classification + regression 问题进行分析：  

![img][img-0000-0102]
分类问题还是比较好解决的，关键是对于 regression
比较难以解决，主要是训练参数收敛时间要长很多，所以讲regression问题如何转换为分类问题是当时考虑的一个观点：即通过对产生的滑窗进行分类判别，取其中分类最好的边框作为最后的框（当然考虑NMS，即非极大值抑制）即：
![img][img-0000-0103]    

传统目标的主要问题：
1. 基于滑动窗口的区域的策略没有针对性，时间复杂性搞，窗口冗余
2. 手工设计的特征不具有很好的鲁棒性  

**R-CNN** 横空出世来解决bug啦~~  
通过选择性搜索方式（[[pdf]][pdf-0000-0101]）预先找出 Region Proposal,其中利用图像的纹理、边缘、颜色等信息，使得产生的候选框在较小的窗口仍能保持较高的召回率。当然也有通过EdgeBoxes{TODO}：了解一下。正是建立在region proposal + CNN 框架下，RBG在2014年设计了R-CNN，使得目标检测取得巨大突破  

R-CNN的步骤：
1. 输入测试图像
2. 利用selective search从图像得到 2k 左右的Region Proposal
3. 对取出的候选框进行缩放到 227 * 277 大小并输入到CNN，fc7 作为输出特征
4. 对fc7 输入到SVM 进行分类
?5. 使用回归其精细修正候选框位置，对于每一个类，训练一个线性回归模型去判定这个框是否完美。  
这样处理很慢，因为每一个region proposal 都要进行CNN + SVM是十分耗时的，一张图可能需要 47S。  
![img][img-0000-0104] 

那有没有问题可以解决啊  
答案是有的，从R-CNN处理的方法来看，其实是可以其中在图像上做一次CNN，然后对region proposal的位置映射到特征图上，这样就只需要进行一次CNN。然而对于不同尺度大小的proposal如何处理为一样呢？这里就不得不提Kaiming He等提出的SPP(空间金字塔池化)，即解决了尺寸不归一导致无法全连接输出，也是提议区域的全局内容未经裁剪压缩送到后续处理。（R-CNN为了统一大小，不得不做出裁剪或者压缩）  
这样网络的结构变为：  
![img][img-0000-0105]   

特点：
1. 结合空间金字塔方法实现CNNs的多尺度输入
2. 只对原图提取一次卷积特征   

**Fast R-CNN**  
![img][img-0000-0106]  

在Fast R-CNN上就采纳了SPP net的方法，对其进行了改进。通过对比，可以发现在处理过程有两处不同，一是引入了ROI pooling（即精简版的SPP net，不同的是只对得到layer进行下采样得到`7*7`的feature map，同时vgg16的conv5_3对应512个feature map，即全连接的特征向量为`7*7*512`）;而是损失函数使用了多任务损失函数，将回归问题直接加入到CNN 网络训练(bbox regression 放入训练)  

**Faster R-CNN** wtf，还可以提高？  
推荐[Tensorflow 版本 Faster RCNN 代码解读][link-0000-0101]
是的，还可以改进。其实分析可以发现前两版为了生成较好的候选框，使用了selective search，这个过程是非常耗时的。那能不能找到一个更为高效方式来代替呢?  
Region Proposal Network(RPN),具体怎么操作呢？将RPN放在最后一个卷积层后面，RPN直接训练得到候选区域。  
![img][img-0000-0107]  
{TODO}:理解RPN简介：
1. 在feature map上滑动窗口
2. 建一个神经网络用分类+框位置的回归  
3. 滑动窗口位置提供物体大体位置信息
4. 框的回归提供了框更精确的位置
四个损失函数：  
* RPN calssification(anchor good.bad)
* RPN regression(anchor->propoasal)
* Fast R-CNN classification(over classes)
* Fast R-CNN regression(proposal ->box)  
![img][img-0000-0108]   
![img][img-0000-0109]  



#### [从YOLOv1到YOLOv3，目标检测的进化之路][Title-0608-01] + AI科技大本营

> yolo 的核心思想在于通过将图片分成 S*S 网格进行回归预测bounding box 的位置和所属类别
* YOLO  
直接在卷积网络全连接层后面一次全连接，后对连接层进行 reshape 到 (S,S,B*5+C) 用来对边界框进行回归预测。强烈推荐观看该[PPT][PPT-0608-0101] 学习yolo的整个运行流程,讲解的非常的清楚。  
整体的损失函数描述：  
![img-0608-01-01][img-0608-01-01]
一些问题：
	* localization error 和 classification· error的权值不应该同等
	* 若网格没有object，则应当将网络的 box 的 confidence push 到0
解决方法：
	* coord 权值设为5，noobje 设为0.5，confidence loss 和 weight loss 设为1
	* 对于大小框偏移 loss 不应当同等看待的问题，作者采用将 box 的width 和 height 采用平方根的形式。

1. YOLO 对相互靠的很近的物体，还有很小的群体检测效果不好，这是因为一个网格中只预测了两个框，并且只属于一类。
2. 同一类物体出现的新的不常见的长宽比和其他情况时，泛化能力偏弱。
3. 由于损失函数的问题，定位误差是影响检测效果的主要原因。尤其是大小物体的处理上，还有待加强。

* YOLOv2
> 代表着目前业界最先进物体检测水平，他的速度也是毋庸置疑的快，且YOLO9000 这个网络可以实时检测超过 9000 类物体，这归功于使用了 WordTree。  

YOLOv2里面的小trick：
* trick
	* Batch Normalization：每一个卷积层增加了批次归一化，使得mAP提高了2%，使用了BN，去掉了Dropout
	* High resolution classifier：工业界普遍用的是 256*256，而现在YOLOv2将分辨率从224*224 提高到448*448，具体做法是先在448*448分辨率在ImageNet进行微调，调优用于检测的Resulting Network，使得mAP提升了4%
	* Convolution with anchor boxes：YOLO最开始版本使用的是全连接层去直接预测Bounding Boxes的坐标值。YOLOv2则去掉最后连接层，通过32下采样，对于416的图片得到输出的13*13 的feature map。使用anchor box会使得精确度稍微下降，但用它可以让YOLO预测上千个框，同时recall达到 88%，mAP达到69.2%
	* Dimension clusters：这算是YOLOv2上一个比较好的设计内容，通过k-mean聚类真实框分布，从而找到一个较好的值进行迭代
	* Direct location prediction：对于使用Anchor Box会使得model变得不稳定，主要是参数x，y位置不稳定，为了保证其预测稳定，通过Logistic Activation使得预测结果在0-1之间。通过使用`Dimension clusters` 和 `Direct location prediction` 其效果比其他使用anchor box 提高了近 5%。
	* Fine-Grained Features：YOLOv2加上了一个Passthrough layer 去得26*26分辨率的特征，用来取得一些细粒度的特征信息
	* Multi-Scale Training:网络自动修改尺寸，继续训练。其尺度迭代每10个Batch更改一个（320,352，...608）
相比之前的版本，通过使用GoogLeNet架构，计算次数从306.9亿次降低到85.2亿次，虽然精度有所下降。  
{TODO}:理解Hierarchical classification  
![img][img-0608-01-02]

* YOLOv3   
> YOLOv3 的模型相比之前的复杂了不少，可以通过改变模型结构大小来权衡速度与精度。{TODO}:简而言之，YOLOv3 的先验检测（Prior detection）系统将分类器或定位器重新用于执行检测任务。他们将模型应用于图像的多个位置和尺度。而那些评分较高的区域就可以视为检测结果。此外，相对于其它目标检测方法，我们使用了完全不同的方法。我们将一个单神经网络应用于整张图像，该网络将图像划分为不同的区域，因而预测每一块区域的边界框和概率，这些边界框会通过预测的概率加权。我们的模型相比于基于分类器的系统有一些优势。它在测试时会查看整个图像，所以它的预测利用了图像中的全局信息。与需要数千张单一目标图像的 R-CNN 不同，它通过单一网络评估进行预测。这令 YOLOv3 非常快，一般它比 R-CNN 快 1000 倍、比 Fast R-CNN 快 100 倍。  


* 改进地方：
	* 多尺度（三个）预测三个box（类FPN）
	* 使用了类ResNet的分类网络
	* 分类器-类别预测，不适用Softmax对框进行预测
YOLOv3的结构：  
![img][img-0608-01-03]  
YOLOv3 在 mAP@0.5 及小目标 APs 上具有不错的结果，但随着 IOU的增大，性能下降，说明 YOLOv3 不能很好地与 ground truth 切合.  

YOLOv3 输入416*416 整个网络流程（使用keras生成）
![img](img/yolov3-416.png)




---
[Title-0000-01]:https://mp.weixin.qq.com/s/5WLw5Do909KaHsHsagD7WQ
[link-0000-0101]:https://zhuanlan.zhihu.com/p/32230004
[pdf-0000-0101]:http://www.huppelen.nl/publications/selectiveSearchDraft.pdf
[img-0000-0101]:img/20180000-01-01.jpg
[img-0000-0102]:img/20180000-01-02.jpg
[img-0000-0103]:img/20180000-01-03.jpg
[img-0000-0104]:img/20180000-01-04.jpg
[img-0000-0105]:img/20180000-01-05.jpg
[img-0000-0106]:img/20180000-01-06.jpg
[img-0000-0107]:img/20180000-01-07.jpg
[img-0000-0108]:img/20180000-01-08.jpg
[img-0000-0109]:img/20180000-01-09.jpg

[Title-0608-01]:https://mp.weixin.qq.com/s/f27A3hfXgGx0b24_5mKryw
[img-0608-01-01]:img/20180608-01-01.jpg
[img-0608-01-02]:img/20180608-01-02.jpg
[img-0608-01-03]:img/20180608-01-03.jpg
[PPT-0608-0101]:https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.p
