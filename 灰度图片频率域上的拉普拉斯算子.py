from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, fftpack
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 加载示例灰度图片
image = Image.open('../pictures/mountain.png').convert('L')
image = np.array(image)

# 空间域拉普拉斯算子
laplacian_kernel = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]])
spatial_laplacian = ndimage.convolve(image, laplacian_kernel, mode='reflect')

# 频率域拉普拉斯算子
f = fftpack.fft2(image)
fshift = fftpack.fftshift(f)

rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
frequency_laplacian = np.zeros((rows, cols), dtype=np.float32)
for i in range(rows):
    for j in range(cols):
        frequency_laplacian[i, j] = -4 * np.pi**2 * ((i - crow)**2 + (j - ccol)**2)

frequency_laplacian[crow, ccol] = rows * cols  # 设置中心点为0

frequency_filtered = fshift * frequency_laplacian
frequency_ifft = fftpack.ifftshift(frequency_filtered)
frequency_laplacian_image = np.abs(fftpack.ifft2(frequency_ifft))

# 显示结果
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(image, cmap='gray')
axs[0].set_title('原始图像')
axs[0].axis('off')

axs[1].imshow(spatial_laplacian, cmap='gray')
axs[1].set_title('空间域拉普拉斯算子')
axs[1].axis('off')

axs[2].imshow(frequency_laplacian_image, cmap='gray')
axs[2].set_title('频率域拉普拉斯算子')
axs[2].axis('off')

plt.tight_layout()
plt.show()