import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
image = cv2.imread('../pictures/weierlite.png', cv2.IMREAD_GRAYSCALE)

# 定义Kirsch算子的八个方向卷积核
kernels = [
    np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),   # 方向 0 (上)
    np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),   # 方向 45° (右上)
    np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),   # 方向 90° (右)
    np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),   # 方向 135° (右下)
    np.array([[-3, -3, -3], [-3, 0, 5], [-3, -3, 5]]),  # 方向 180° (下)
    np.array([[-3, -3, -3], [-3, 0, 5], [5, 5, -3]]),   # 方向 225° (左下)
    np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),   # 方向 270° (左)
    np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])    # 方向 315° (左上)
]

# 对图像应用八个方向的卷积核，计算梯度
gradient_images = []
for kernel in kernels:
    gradient_image = cv2.filter2D(image, -1, kernel)
    gradient_images.append(gradient_image)

# 计算每个像素点的最大梯度值
edge_magnitude = np.max(np.array(gradient_images), axis=0)

# 阈值化处理，提取边缘
_, thresholded_edges = cv2.threshold(edge_magnitude, 50, 255, cv2.THRESH_BINARY)

# 显示结果
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 3, 2), plt.imshow(edge_magnitude, cmap='gray'), plt.title('Kirsch Gradient Magnitude')
plt.subplot(1, 3, 3), plt.imshow(thresholded_edges, cmap='gray'), plt.title('Thresholded Edge Detection')
plt.tight_layout()
plt.show()