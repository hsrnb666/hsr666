import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
img_path = '../pictures/mountain.png'  # 替换为你的图像路径
original_img = cv.imread(img_path, cv.IMREAD_COLOR)

# 创建窗口以显示图像
cv.namedWindow('Original Image', cv.WINDOW_AUTOSIZE)
cv.imshow('Original Image', original_img)

# 均值滤波
mean_filtered = cv.blur(original_img, (5, 5))

# 方框滤波（归一化）
box_filtered = cv.boxFilter(original_img, -1, (5, 5), normalize=True)

# 高斯滤波
gaussian_filtered = cv.GaussianBlur(original_img, (5, 5), 0)

# 显示滤波后的图像
cv.imshow('Mean Filtered Image', mean_filtered)
cv.imshow('Box Filtered Image', box_filtered)
cv.imshow('Gaussian Filtered Image', gaussian_filtered)

# 使用matplotlib显示图像对比
plt.figure(figsize=(15, 5))
# 显示原图
plt.subplot(141)
plt.imshow(cv.cvtColor(original_img, cv.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

# 显示均值滤波后的图像
plt.subplot(142)
plt.imshow(cv.cvtColor(mean_filtered, cv.COLOR_BGR2RGB))
plt.title('Mean Filtered')
plt.axis('off')

# 显示方框滤波后的图像
plt.subplot(143)
plt.imshow(cv.cvtColor(box_filtered, cv.COLOR_BGR2RGB))
plt.title('Box Filtered')
plt.axis('off')

# 显示高斯滤波后的图像
plt.subplot(144)
plt.imshow(cv.cvtColor(gaussian_filtered, cv.COLOR_BGR2RGB))
plt.title('Gaussian Filtered')
plt.axis('off')

plt.show()
