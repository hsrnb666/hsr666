import numpy as np
import matplotlib.pyplot as plt
import cv2
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 读取上传的图片
image_path = '../pictures/mountain.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# 傅里叶变换
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

# 创建低通滤波器
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
radius = 30  # 截止频率半径

# 理想低通滤波器
ideal_lowpass = np.zeros((rows, cols), dtype=np.float32)
cv2.circle(ideal_lowpass, (ccol, crow), radius, 1, -1)

# 巴特沃思低通滤波器
butterworth_lowpass = np.zeros((rows, cols), dtype=np.float32)
for i in range(rows):
    for j in range(cols):
        d = np.sqrt((i - crow)**2 + (j - ccol)**2)
        butterworth_lowpass[i, j] = 1 / (1 + (d / radius)**4)

# 高斯低通滤波器
gaussian_lowpass = np.zeros((rows, cols), dtype=np.float32)
for i in range(rows):
    for j in range(cols):
        d = np.sqrt((i - crow)**2 + (j - ccol)**2)
        gaussian_lowpass[i, j] = np.exp(-(d**2 / (2.0 * (radius**2))))

# 应用滤波器
ideal_filtered = fshift * ideal_lowpass
butterworth_filtered = fshift * butterworth_lowpass
gaussian_filtered = fshift * gaussian_lowpass

# 逆傅里叶变换
ideal_ifft = np.fft.ifftshift(ideal_filtered)
butterworth_ifft = np.fft.ifftshift(butterworth_filtered)
gaussian_ifft = np.fft.ifftshift(gaussian_filtered)

ideal_image = np.abs(np.fft.ifft2(ideal_ifft))
butterworth_image = np.abs(np.fft.ifft2(butterworth_ifft))
gaussian_image = np.abs(np.fft.ifft2(gaussian_ifft))

# 显示结果
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('原始图像')
axs[0, 0].axis('off')

axs[0, 1].imshow(ideal_image, cmap='gray')
axs[0, 1].set_title('理想低通滤波')
axs[0, 1].axis('off')

axs[1, 0].imshow(butterworth_image, cmap='gray')
axs[1, 0].set_title('巴特沃思低通滤波')
axs[1, 0].axis('off')

axs[1, 1].imshow(gaussian_image, cmap='gray')
axs[1, 1].set_title('高斯低通滤波')
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()