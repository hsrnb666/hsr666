import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 读取图像并转换为灰度图像
image = cv2.imread('../pictures/weierlite.png', cv2.IMREAD_GRAYSCALE)

# 将图像展平为二维数据
height, width = image.shape
image_flattened = image.flatten().reshape(1, -1)

# 进行PCA
pca = PCA()
pca.fit(image_flattened.T)  # PCA是基于列方向的协方差矩阵进行计算

# 选择合适的特征值个数（例如选择累计方差达到95%的特征值个数）
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(explained_variance_ratio >= 0.95) + 1

# 根据选择的主成分数重建图像
reduced_data = pca.transform(image_flattened.T)[:, :num_components]
reconstructed_image = pca.inverse_transform(reduced_data).reshape(height, width)

# 显示原图和恢复图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 2, 2), plt.imshow(reconstructed_image, cmap='gray'), plt.title('Reconstructed Image')
plt.tight_layout()
plt.show()