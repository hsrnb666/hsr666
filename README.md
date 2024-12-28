这是一个数字图像处理的大作业，里面有处理这些问题的代码，代码所需要的数据在pictures里面


（1）灰度级切片，位平面切片，直方图统计，直方图均衡化
（2）对原图进行三种不同的平滑处理，选择合适的均值滤波器、方框滤波器以及高斯滤波器，需要有前后处理的图片对比，以及说明哪种滤波器最好。
（3）对原图进行一阶锐化处理，从Roberts算子、Sobel算子、Prewitt算子以及Kirsch算子进行选择；对原图进行二阶锐化处理，即拉普拉斯算子；
（4）给出一张彩色图片的RGB以及HSI分量图
（5）分别在RGB和HSI空间上进行直方图均衡化
（6）RGB上进行均值滤波以及拉普拉斯变换，仅在HSI的强度分量上进行相同的操作，比较两者的结果。
（7）给出一张灰度图片的傅立叶变换频谱图，再进行逆变换
（8）对灰度图片进行理想、巴特沃思以及高斯低通滤波处理，并和之前的空间平滑滤波进行比较。
（9）空间域和频率域上的拉普拉斯算子比较
（10）在灰度图片上加上高斯噪声、均匀噪声以及椒盐噪声，分别给出原图加上噪声污染后的图片，并给出对应的四张直方图
（11）选择合适的滤波器对以上三张噪声污染图片进行噪声清除，并给出前后对比图
（12）灰度图片运动模糊并加上高斯噪声后，分别用维纳滤波以及约束最小二乘方滤波进行恢复
（13）对灰度图像进行Prewitt梯度算子边缘检测, 分析一下图片效果，考虑是否需要平滑后再次检测，或者采用对角线的Prewitt梯度算子进行处理，并给出原因？最后进行阈值化使边缘结果更加清晰
（14）对灰度图像进行kirsch算子边缘检测
（15）主成分提取
（16）HOG特征提取
（17）Harris 角点检测
（18）Hough Transform
（19）人脸检测