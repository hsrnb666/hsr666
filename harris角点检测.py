import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
image = cv2.imread('../pictures/weierlite.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 将灰度图像转换为浮动点型数据
gray = np.float32(gray)

# 使用OpenCV的cornerHarris函数计算Harris角点
# 这里的blockSize表示计算自相关矩阵时的局部邻域大小，ksize为Sobel算子的窗口大小，k为Harris角点检测的常数
corner_response = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# 进行膨胀，增强角点的显示
corner_response = cv2.dilate(corner_response, None)

# 设置阈值，标记角点
threshold = 0.01 * corner_response.max()
image[corner_response > threshold] = [0, 0, 255]  # 在原图上标记角点，红色标记

# 显示结果
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Harris Corner Detection')

# 如果需要查看角点响应图
plt.subplot(1, 2, 2)
plt.imshow(corner_response, cmap='gray')
plt.title('Corner Response Map')

plt.tight_layout()
plt.show()