import cv2
import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 创建一个具有特定模式的灰度图像（中心为白色，周围为黑色的圆）
# 创建一个全零矩阵
image = cv2.imread('../pictures/mountain.png', 0)

# 计算离散傅里叶变换
fourier_transform = fft2(image)
fourier_shifted = fftshift(fourier_transform)

# 显示原始图像和频谱图
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(image, cmap='gray')
ax[0].set_title('原始灰度图像')
ax[0].axis('off')

ax[1].imshow(np.log(1 + np.abs(fourier_shifted)), cmap='gray')
ax[1].set_title('傅里叶变换频谱图')
ax[1].axis('off')

plt.show()

# 执行逆变换
inverse_transform = ifft2(fourier_transform)
restored_image = np.real(inverse_transform)

# 显示逆变换后的图像
plt.imshow(restored_image, cmap='gray')
plt.title('逆变换后的图像')
plt.axis('off')
plt.show()