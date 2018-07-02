一、计算机视觉历史背景

    1、视觉的处理起于物体轮廓的简单描述；
 
    2、目标分割在机器视觉发展的早期较目标识别更为简单，而目标分割的任务在于把一张图片的像素点归类到有意义的区域中，也即图像分割；

    3、目标识别领域的发展在于有可供标注的数据集的出现（IMAGENET）

    4、2012年，IMAGENET大赛在目标识别领域出现突破性的算法（七层卷积神经网络—多伦多大学 Jeff Hinton小组（Alexnet））

二、目标分类

    1、图像分类，早期科学家试图通过编写具体特征规则以识别物体，后放弃而采用网上的大量该类别数据训练分类，此类算法包含两个函数（train function & predict function），前者用于接收图片和标签（CIFAR-10数据集），然后输出一个模型，后者使用生成的模型并对图片进行分类；

    2、比较图片的方法之一曼哈顿距离
     ![image](https://github.com/Alley-X/Git-Learning/blob/master/cv-learning/image_storage/L1_distance.png?raw=true)
     ！[image](https://github.com/Alley-X/Git-Learning/raw/master/cv-learning/image_storage/L1_distance.png)
     ![image](https://github.com/Alley-X/Git-Learning/raw/master/cv-learning/image_storage/L1_distance.png)
     ！[image](https://github.com/Alley-X/Git-Learning/blob/master/cv-learning/image_storage/L1_distance.png)

      最近邻分类算法在训练速度很快（实际上只是存储数据和标签），而在测试阶段很慢（进行大量的向量运算），而后发展的卷积神经网络等相反，训练过程很漫长，在测试阶段很迅速。

    3、KNN算法

    3.1、K值一般赋予较大的值，如此会使图像分类的决策边缘更平滑，同时也能减少背景噪声的影响。

    3.2、关于图片分类的算法除了曼哈顿距离（L1距离）还可使用欧式距离（L2距离）
    ![image](https://github.com/Alley-X/Git-Learning/blob/master/cv-learning/image_storage/L2_distance.png）

L1距离（出租车距离）中，方形上任意一点都与原点等距，|x1-x2|+|y1-y2|，在L2距离中，圆形上任意一点都与原点等距，在这两种度量中，有着更基本的拓扑假设，若改变L1距离的坐标轴方向，会导致距离的改变 ，而在L2中不存在此种现象，对应于实际，若你输入的特征向量有特殊意义，采用L1距离度量更好，若是一个公用空间的通用向量，则L2更合适。K值和距离度量一般称为超参数。

    3.3、在进行模型训练时，将一个数据集分为训练集和测试集，在训练集上选取不同的超参数进行模型训练，再在测试集上选取出性能最好的一组超参数，这种做法不建议，此种做法无可避免地只能在已有的测试集上表现优良，但在未知的数据中就未必有好效果。一般的做法如下：
    ![image](https://github.com/Alley-X/Git-Learning/blob/master/cv-learning/image_storage/dataset_classification.png)

将整个数据集分为三个部分，train中设置不同的超参数训练，再在验证集中进行评估，最后在test中拿出性能最好的模型。值得注意的是，test集一定要保持其“独立性”，不能与train中其他数据产生联系。有时也会将train分为几个部分，在前几个fold中训练，在最后一个fold中验证，再分别以之前的训练fold为验证集，其余部分为训练集再对超参数进行性能评估，此种方法称为交叉验证。
![image](https://github.com/Alley-X/Git-Learning/blob/master/cv-learning/image_storage/dataset_classification.png)


    3.4、在训练集中，算法“知道”图片数据的分类标签，而在验证集中“不知道”，算法根据训练集中得出的超参数对验证集中数据分类。更详细地说，对于KNN算法，训练集就是一堆有着标签的图片，验证集中的每一张图片与训练集的图片距离度量操作，计算出的距离最近以“投票”的形式（K值）决定该图片属于哪一标签，再用验证集的label检测算法分类的精确性。

    3.5、关于test数据集的建立，其不可避免地存在一定误差，即不能完全表示现实世界，其原因根植于统计学，所以在建立数据集时，必须采用同一种方法收集并随机打乱，分出train dataset、validation dataset、test dataset。

    4、线性分类器（linear classification）

    4.1、神经网络中最基础的模块，其可泛化到整个神经网络中，而在整个神经网络中，最核心的是构建F函数的形式，在linear classification中，所有训练的经验和知识都存储在参数矩阵W中，而linear classification的工作原理是将测试图片拉成一个列阵，再与W参数矩阵相乘加上参数矩阵每行对应类别的偏差项，最后得出评估分数，以分类类别。一般来说，评分最高的就代表其对应所属的类别，而在下图中因为模型过于简单，得分最高437.9却属于类别“狗”。
    ![image](http://git.tesool.com/tair/cv-learning/blob/master/image_storage/validation.png)


    4.2、损失函数（loss function），为使评分最高的类和测试图片相契合，需要不断调整参数矩阵W，而设计一种能达到这种功能的算法就是损失函数需做的。


