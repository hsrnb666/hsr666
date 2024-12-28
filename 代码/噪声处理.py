import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图片
def read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

# 添加高斯噪声
def add_gaussian_noise(image, mean=0, var=0.01):
    row, col = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

# 添加均匀噪声
def add_uniform_noise(image, low=-10, high=10):
    row, col = image.shape
    uniform = np.random.uniform(low, high, (row, col))
    noisy = image + uniform
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

# 添加椒盐噪声
def add_salt_and_pepper_noise(image, prob=0.02):
    output = np.copy(image)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
    return output

# 均值滤波
def apply_mean_filter(image, kernel_size=5):
    return cv2.blur(image, (kernel_size, kernel_size))

# 中值滤波
def apply_median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

# 显示图片
def show_images(images, titles, cmap='gray'):
    plt.figure(figsize=(10, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.show()

# 主函数
def main():
    image_path = '../pictures/weierlite.png'  # 替换为你的图片路径
    original_image = read_image(image_path)

    # 添加噪声
    gaussian_noisy_image = add_gaussian_noise(original_image)
    uniform_noisy_image = add_uniform_noise(original_image)
    salt_and_pepper_noisy_image = add_salt_and_pepper_noise(original_image)

    # 应用滤波器
    gaussian_filtered_image = apply_mean_filter(gaussian_noisy_image)
    uniform_filtered_image = apply_mean_filter(uniform_noisy_image)
    salt_and_pepper_filtered_image = apply_median_filter(salt_and_pepper_noisy_image)

    # 显示前后对比图
    show_images([gaussian_noisy_image, gaussian_filtered_image], ['Gaussian Noisy', 'Gaussian Filtered'])
    show_images([uniform_noisy_image, uniform_filtered_image], ['Uniform Noisy', 'Uniform Filtered'])
    show_images([salt_and_pepper_noisy_image, salt_and_pepper_filtered_image], ['Salt and Pepper Noisy', 'Salt and Pepper Filtered'])

if __name__ == '__main__':
    main()