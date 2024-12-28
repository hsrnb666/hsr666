import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# 读取图像并转换为灰度图像
image = cv2.imread('../pictures/mountain.png', cv2.IMREAD_GRAYSCALE)

# Canny 边缘检测
canny_edges = cv2.Canny(image, 100, 200)

# Prewitt 算子
# 定义 Prewitt 卷积核
prewitt_kernel_x = np.array([[1, 0, -1],
                             [1, 0, -1],
                             [1, 0, -1]])

prewitt_kernel_y = np.array([[1, 1, 1],
                             [0, 0, 0],
                             [-1, -1, -1]])

# 对图像进行卷积，分别计算水平和垂直方向的梯度
gradient_x = convolve(image, prewitt_kernel_x)
gradient_y = convolve(image, prewitt_kernel_y)

# 计算边缘强度
prewitt_edges = np.sqrt(gradient_x**2 + gradient_y**2)
prewitt_edges = np.uint8(prewitt_edges)

# 显示结果
plt.figure(figsize=(12, 6))

# 原图
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Canny 边缘检测结果
plt.subplot(1, 3, 2)
plt.imshow(canny_edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')

# Prewitt 边缘检测结果
plt.subplot(1, 3, 3)
plt.imshow(prewitt_edges, cmap='gray')
plt.title("Prewitt Edge Detection")
plt.axis('off')

plt.tight_layout()
plt.show()

