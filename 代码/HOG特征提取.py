import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog

# 读取图像
image = cv2.imread('../pictures/weierlite.png', cv2.IMREAD_GRAYSCALE)

# 设置HOG参数
orientations = 9  # 方向数量
pixels_per_cell = (8, 8)  # 每个细胞的大小（像素）
cells_per_block = (3, 3)  # 每个块的大小（细胞）
block_norm = 'L2-Hys'  # 块归一化方法

# 计算HOG特征
hog_features, hog_image = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                              cells_per_block=cells_per_block, block_norm=block_norm, visualize=True)

# 显示HOG图像
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Image')
plt.show()

# 绘制HOG特征的直方图
plt.figure(figsize=(10, 6))
plt.hist(hog_features.ravel(), bins=100, color='blue', alpha=0.7)
plt.title('HOG Features Histogram')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.show()