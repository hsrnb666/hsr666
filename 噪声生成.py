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

# 绘制直方图
def plot_histogram(image, title):
    plt.figure()
    plt.hist(image.ravel(), 256, [0,256])
    plt.title(title)
    plt.show()

# 主函数
def main():
    image_path = '../pictures/weierlite.png'  # 替换为你的图片路径
    original_image = read_image(image_path)

    # 添加高斯噪声
    gaussian_noisy_image = add_gaussian_noise(original_image)
    # 添加均匀噪声
    uniform_noisy_image = add_uniform_noise(original_image)
    # 添加椒盐噪声
    salt_and_pepper_noisy_image = add_salt_and_pepper_noise(original_image)

    # 显示图片
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.subplot(2, 2, 2)
    plt.imshow(gaussian_noisy_image, cmap='gray')
    plt.title('Gaussian Noisy Image')
    plt.subplot(2, 2, 3)
    plt.imshow(uniform_noisy_image, cmap='gray')
    plt.title('Uniform Noisy Image')
    plt.subplot(2, 2, 4)
    plt.imshow(salt_and_pepper_noisy_image, cmap='gray')
    plt.title('Salt and Pepper Noisy Image')
    plt.show()

    # 绘制直方图
    plot_histogram(original_image, 'Original Image Histogram')
    plot_histogram(gaussian_noisy_image, 'Gaussian Noisy Image Histogram')
    plot_histogram(uniform_noisy_image, 'Uniform Noisy Image Histogram')
    plot_histogram(salt_and_pepper_noisy_image, 'Salt and Pepper Noisy Image Histogram')

if __name__ == '__main__':
    main()