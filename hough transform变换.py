import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
image = cv2.imread('../pictures/wmc2.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用Canny算法进行边缘检测
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 使用霍夫变换检测图像中的直线
# cv2.HoughLines()返回的是(r, θ)参数空间中的直线
# 第一个参数：输入的边缘图像
# 第二个参数：霍夫变换的分辨率（rho），以像素为单位
# 第三个参数：霍夫变换的角度分辨率（theta），单位为弧度
# 第四个参数：阈值，指明霍夫空间中投票数大于此值才认为是直线
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

# 在原图上绘制检测到的直线
for line in lines:
    r, theta = line[0]
    # 计算直线的两个点的坐标
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * r
    y0 = b * r
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    # 在原图上绘制直线
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 显示结果
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Lines')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection (Canny)')

plt.tight_layout()
plt.show()