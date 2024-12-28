import cv2
import numpy as np
from matplotlib import pyplot as plt
from pylab import mpl
# 设置中文显示字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

def cv_show(name, img):
    '''
     显示图像
    '''
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()
def show_two_pictures(img_one, img_two):
    '''
     对比显示两张图片
    '''
    cv_show("Two Pictures", np.hstack((img_one, img_two)))

def grayscale_slice(img, threshold):
    '''
     灰度级切片
    '''
    (h, w) = img.shape
    img_copy = img.copy()
    for i in range(h):
        for j in range(w):
            if img_copy[i, j] > threshold:
                img_copy[i, j] = 255
            else:
                img_copy[i, j] = 0
    return img_copy

img = cv2.resize(cv2.imread("../pictures/mountain.png", 0), (400, 250))
show_two_pictures(img, grayscale_slice(img, 100))

fig, ax = plt.subplots(figsize=(6, 4))
r = np.linspace(0, 255, 256)
T_r = np.where(r > 100, 255, 0)
ax.plot(r, T_r, color='black')
ax.set_title('转化函数 T(r)')
ax.set_xlabel('原始灰度值 r')
ax.set_ylabel('转换后的灰度值 T(r)')
ax.grid(True)
plt.tight_layout()

plt.show()
