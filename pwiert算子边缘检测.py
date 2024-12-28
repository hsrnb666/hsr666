import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
image = cv2.imread('../pictures/weierlite.png', cv2.IMREAD_GRAYSCALE)

# 对图像进行高斯平滑处理，去噪
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Prewitt 水平和垂直算子
prewitt_horizontal = np.array([[-1, 0, 1],
                               [-1, 0, 1],
                               [-1, 0, 1]])

prewitt_vertical = np.array([[-1, -1, -1],
                             [ 0,  0,  0],
                             [ 1,  1,  1]])

# 计算水平和垂直方向的梯度
gradient_x = cv2.filter2D(blurred_image, -1, prewitt_horizontal)
gradient_y = cv2.filter2D(blurred_image, -1, prewitt_vertical)

# 计算梯度的幅度和方向
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_direction = np.arctan2(gradient_y, gradient_x)

# 归一化梯度幅度
gradient_magnitude = np.uint8(np.clip(gradient_magnitude, 0, 255))

# 对角线方向 Prewitt 算子（检测斜边缘）
prewitt_diag_45 = np.array([[-1, 0, 1],
                            [ 0, 0, 0],
                            [ 1, 0, -1]])

prewitt_diag_135 = np.array([[ 1, 0, -1],
                             [ 0, 0, 0],
                             [-1, 0,  1]])

# 计算对角线方向的梯度
gradient_diag_45 = cv2.filter2D(blurred_image, -1, prewitt_diag_45)
gradient_diag_135 = cv2.filter2D(blurred_image, -1, prewitt_diag_135)

# 计算对角线梯度的幅度
gradient_diag_45_magnitude = np.sqrt(gradient_diag_45**2)
gradient_diag_135_magnitude = np.sqrt(gradient_diag_135**2)

# 合并水平方向、垂直方向、对角线方向的梯度
gradient_combined = np.sqrt(gradient_magnitude**2 + gradient_diag_45_magnitude**2 + gradient_diag_135_magnitude**2)

# 归一化梯度合成图像
gradient_combined = np.uint8(np.clip(gradient_combined, 0, 255))

# 阈值化处理，去除弱边缘
_, thresholded_edge = cv2.threshold(gradient_combined, 100, 255, cv2.THRESH_BINARY)

# 显示结果
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(2, 3, 2), plt.imshow(blurred_image, cmap='gray'), plt.title('Blurred Image')
plt.subplot(2, 3, 3), plt.imshow(gradient_magnitude, cmap='gray'), plt.title('Prewitt Gradient Magnitude')
plt.subplot(2, 3, 4), plt.imshow(gradient_combined, cmap='gray'), plt.title('Combined Gradient')
plt.subplot(2, 3, 5), plt.imshow(thresholded_edge, cmap='gray'), plt.title('Thresholded Edge Detection')
plt.tight_layout()
plt.show()