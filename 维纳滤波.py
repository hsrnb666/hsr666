import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve

# 读取灰度图片
def read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

# 运动模糊
def apply_motion_blur(image, kernel_size=15):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return fftconvolve(image, kernel, mode='same')

# 添加高斯噪声
def add_gaussian_noise(image, mean=0, var=0.01):
    row, col = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

# 维纳滤波恢复
def wiener_filter(image, kernel, K=0.01):
    kernel /= np.sum(kernel)
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel, s=image.shape)
    kernel_conj_fft = np.conj(kernel_fft)
    psf = kernel_conj_fft / (np.abs(kernel_fft) ** 2 + K)
    restored_image_fft = image_fft * psf
    restored_image = np.fft.ifft2(restored_image_fft)
    restored_image = np.abs(restored_image)
    return np.clip(restored_image, 0, 255).astype(np.uint8)

# 约束最小二乘方滤波恢复
def constrained_least_squares_filter(image, kernel, lambda_param=0.1):
    kernel /= np.sum(kernel)
    image_size = image.shape
    kernel_fft = np.fft.fft2(kernel, s=image_size)
    kernel_conj_fft = np.conj(kernel_fft)
    regularization_term = lambda_param * np.eye(image_size[0])
    regularization_term_fft = np.fft.fft2(regularization_term, s=image_size)
    denominator = kernel_conj_fft * kernel_fft + regularization_term_fft
    numerator = np.fft.fft2(image) * kernel_conj_fft
    restored_image_fft = numerator / denominator
    restored_image = np.fft.ifft2(restored_image_fft)
    restored_image = np.abs(restored_image)
    return np.clip(restored_image, 0, 255).astype(np.uint8)

# 显示图片
def show_images(images, titles):
    plt.figure(figsize=(10, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 2, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()

# 主函数
def main():
    image_path = '../pictures/weierlite.png'

    original_image = read_image(image_path)

    # 应用运动模糊
    blurred_image = apply_motion_blur(original_image)
    # 添加高斯噪声
    noisy_image = add_gaussian_noise(blurred_image)

    # 维纳滤波恢复
    wiener_restored_image = wiener_filter(noisy_image, np.ones((15, 15)))
    # 约束最小二乘方滤波恢复
    cls_restored_image = constrained_least_squares_filter(noisy_image, np.ones((15, 15)))

    # 显示前后对比图
    show_images([original_image, blurred_image, noisy_image], ['Original Image', 'Blurred Image', 'Noisy Image'])
    show_images([wiener_restored_image, cls_restored_image],
                ['Wiener Filter Restored Image', 'Constrained Least Squares Filter Restored Image'])


if __name__ == '__main__':
    main()