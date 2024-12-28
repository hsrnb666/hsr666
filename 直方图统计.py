import cv2
from matplotlib import pyplot as plt

def Equalize_Hist():
    # 1、读取图片, 并转换成灰度图
    img = cv2.imread("../pictures/mountain.png", 0)  # 灰度图

    # 2、直方图均衡化
    img_equal = cv2.equalizeHist(img)  # 直方图均衡化

    # 3、显示直方图
    f, ax = plt.subplots(2, 2, figsize=(16, 16))
    # 显示图像
    ax[0, 0].set_title("origin")
    ax[0, 0].imshow(img, "gray")
    ax[0, 1].set_title("Equalized")
    ax[0, 1].imshow(img_equal, "gray")  # 注："gray"是有效名，不能乱写
    # 显示直方图
    ax[1, 0].hist(img.ravel(), 256)  # ravel()：多维数组转一维数组
    ax[1, 1].hist(img_equal.ravel(), 256)
    plt.show()


if __name__ == '__main__':
    plt.figure(figsize=(6, 4))  # 直方图大小
    Equalize_Hist()  # 直方图均衡化
