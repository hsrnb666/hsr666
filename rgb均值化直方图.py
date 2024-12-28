# 导入库
import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections


# 直方图均衡化
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


x = []
for i in range(0, 256):  # 横坐标
    x.append(i)

# 原图及其直方图
origin = cv2.imread('../pictures/weierlite.png')
original = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)

histogram_original = draw_histogram(original)
plt.bar(x, histogram_original)  # 绘制原图直方图
plt.savefig('D:/深度学习/数字图像处理/pictures/before_histogram.png')  # 这里暂时不关闭绘制直方图的窗口，在处理图片后，再在该窗口绘制直方图作对比

before_histogram = cv2.imread('../pictures/before_histogram.png')

lut = np.zeros(256, dtype=original.dtype)  # 创建空的查找表
rgb_histogram = histogram_equalization(histogram_original, lut, original)  # 均衡化处理

histogram_rgb_equalization = draw_histogram(rgb_histogram)
plt.bar(x, histogram_rgb_equalization)  # 绘制原图直方图
plt.savefig('D:/深度学习/数字图像处理/pictures/rgb_after_histogram.png')
plt.close()  # 关闭绘制直方图的窗口
rgb_after_histogram = cv2.imread('../pictures/rgb_after_histogram.png')

# 展示结果
plt.subplot(221), plt.imshow(original, cmap='gray')
plt.title('original')
plt.axis('off')

plt.subplot(222), plt.imshow(before_histogram, cmap='gray')
plt.title('before_histogram')
plt.axis('off')

plt.subplot(223), plt.imshow(rgb_histogram, cmap='gray')
plt.title('rgb_histogram')
plt.axis('off')

plt.subplot(224), plt.imshow(rgb_after_histogram, cmap='gray')
plt.title('rgb_after_histogram')
plt.axis('off')

plt.show()
