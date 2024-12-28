import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections
from numpy import cos, arccos, sqrt, power, pi


def histogram_equalization(histogram_e, lut_e, image_e):
    sum_temp = 0
    cf = []
    for i in histogram_e:
        sum_temp += i
        cf.append(sum_temp)
    for i, v in enumerate(lut_e):
        lut_e[i] = int(255.0 * (cf[i] / sum_temp) + 0.5)
    equalization_result = lut_e[image_e]
    return equalization_result


# 计算灰度图的直方图
def draw_histogram(grayscale):
    # 对图像进行通道拆分
    hsi_i = grayscale[:, :, 2]
    color_key = []
    color_count = []
    color_result = []
    histogram_color = list(hsi_i.ravel())  # 将多维数组转换成一维数组
    color = dict(collections.Counter(histogram_color))  # 统计图像中每个亮度级出现的次数
    color = sorted(color.items(), key=lambda item: item[0])  # 根据亮度级大小排序
    for element in color:
        key = list(element)[0]
        count = list(element)[1]
        color_key.append(key)
        color_count.append(count)
    for i in range(0, 256):
        if i in color_key:
            num = color_key.index(i)
            color_result.append(color_count[num])
        else:
            color_result.append(0)
    color_result = np.array(color_result)
    return color_result


#  HSI转RGB
def hsi_rgb(hsi):
    if hsi.dtype.type == np.uint8:
        hsi = (hsi).astype('float64') / 255.0
    for k in range(hsi.shape[0]):
        for j in range(hsi.shape[1]):
            h, s, i = hsi[k, j, 0], hsi[k, j, 1], hsi[k, j, 2]
            r, g, b = 0, 0, 0
            if 0 <= h < 2 / 3 * pi:
                b = i * (1 - s)
                r = i * (1 + s * cos(h) / cos(pi / 3 - h))
                g = 3 * i - (b + r)
            elif 2 / 3 * pi <= h < 4 / 3 * pi:
                r = i * (1 - s)
                g = i * (1 + s * cos(h - 2 / 3 * pi) / cos(pi - h))
                b = 3 * i - (r + g)
            elif 4 / 3 * pi <= h <= 5 / 3 * pi:
                g = i * (1 - s)
                b = i * (1 + s * cos(h - 4 / 3 * pi) / cos(5 / 3 * pi - h))
                r = 3 * i - (g + b)
            hsi[k, j, 0], hsi[k, j, 1], hsi[k, j, 2] = r, g, b
    return (hsi * 255).astype('uint8')


#  RGB转HSI
def rgb_hsi(rgb):
    # 如果没有归一化处理，则需要进行归一化处理（传入的是[0,255]范围值）
    if rgb.dtype.type == np.uint8:
        rgb = rgb.astype('float64') / 255.0
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            r, g, b = rgb[i, j, 0], rgb[i, j, 1], rgb[i, j, 2]
            # 计算h
            num = 0.5 * ((r - g) + (r - b))
            den = sqrt(power(r - g, 2) + (r - b) * (g - b))
            theta = arccos(num / den) if den != 0 else 0
            rgb[i, j, 0] = theta if b <= g else (2 * pi - theta)
            # 计算s
            rgb[i, j, 1] = (1 - 3 * min([r, g, b]) / (r + g + b)) if r + g + b != 0 else 0
            # 计算i
            rgb[i, j, 2] = 1 / 3 * (r + g + b)
    return (rgb * 255).astype('uint8')


x = []
for i in range(0, 256):  # 横坐标
    x.append(i)

# 原图及其直方图
origin = cv2.imread('../pictures/weierlite.png')
original = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)

histogram_original = draw_histogram(original)
plt.bar(x, histogram_original)  # 绘制原图直方图
plt.savefig('D:/深度学习/数字图像处理/pictures/before_histogram.png')
before_histogram = cv2.imread('../pictures/before_histogram.png')

# rgb转hsi
# hsi_original = cv2.cvtColor(original, cv2.COLOR_RGB2HSV_FULL)
hsi_original = rgb_hsi(original)

# hsi在亮度分量上均衡化
histogram_hsi_original = draw_histogram(hsi_original)
plt.bar(x, histogram_hsi_original)  # 绘制直方图
plt.savefig('D:/深度学习/数字图像处理/pictures/hsi_before_histogram.png')
hsi_before_histogram = cv2.imread('../pictures/hsi_before_histogram.png')

lut = np.zeros(256, dtype=hsi_original.dtype)  # 创建空的查找表
hsi_histogram = histogram_equalization(histogram_hsi_original, lut, hsi_original)  # 均衡化处理

# hsi转rgb
# result=cv2.cvtColor(hsi_histogram,cv2.COLOR_HSV2RGB_FULL)
result = hsi_rgb(hsi_histogram)

histogram_hsi_equalization = draw_histogram(result)
plt.bar(x, histogram_hsi_equalization)  # 绘制原图直方图
plt.savefig('D:/深度学习/数字图像处理/pictures/hsi_after_histogram.png')
plt.close()  # 关闭绘制直方图的窗口
hsi_after_histogram = cv2.imread('../pictures/hsi_after_histogram.png')

# 展示结果
plt.subplot(321), plt.imshow(original, cmap='gray')
plt.title('original')
plt.axis('off')

plt.subplot(322), plt.imshow(before_histogram, cmap='gray')
plt.title('before_histogram')
plt.axis('off')

plt.subplot(323), plt.imshow(hsi_original, cmap='gray')
plt.title('hsi_original')
plt.axis('off')

plt.subplot(324), plt.imshow(hsi_before_histogram, cmap='gray')
plt.title('hsi_before_histogram')
plt.axis('off')

plt.subplot(325), plt.imshow(result, cmap='gray')
plt.title('hsi_histogram')
plt.axis('off')

plt.subplot(326), plt.imshow(hsi_after_histogram, cmap='gray')
plt.title('hsi_after_histogram')
plt.axis('off')

plt.show()