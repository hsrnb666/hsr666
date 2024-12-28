import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('../pictures/mountain.png')
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转成RGB 方便后面显示

# 灰度化处理图像
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯滤波
gaussianBlur = cv2.GaussianBlur(grayImage, (3, 3), 0)

# 阈值处理
ret, binary = cv2.threshold(gaussianBlur, 127, 255, cv2.THRESH_BINARY)



# Sobel算子
x = cv2.Sobel(binary, cv2.CV_16S, 1, 0)
y = cv2.Sobel(binary, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# Laplacian算子
dst = cv2.Laplacian(binary, cv2.CV_16S, ksize=3)
Laplacian = cv2.convertScaleAbs(dst)

# #效果图
# titles = ['Source Image', 'Binary Image', 'Roberts Image',
#           'Prewitt Image','Sobel Image', 'Laplacian Image']
# images = [lenna_img, binary, Roberts, Prewitt, Sobel, Laplacian]
# for i in np.arange(6):
#    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#    plt.title(titles[i])
#    plt.xticks([]),plt.yticks([])
# plt.show()

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# # 显示图形
plt.subplot(231), plt.imshow(img_RGB), plt.title('原始图像'), plt.axis('off')  # 坐标轴关闭
plt.subplot(232), plt.imshow(Sobel, cmap=plt.cm.gray), plt.title('Sobel算子'), plt.axis('off')
plt.subplot(233), plt.imshow(Laplacian, cmap=plt.cm.gray), plt.title('Laplacian算子'), plt.axis('off')

plt.show()