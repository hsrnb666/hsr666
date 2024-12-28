import cv2
import numpy as np


def cv_show(name, img):
    '''
     显示图像
    '''
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def bit_plain_slice(img):
    '''
     位平面切片
    '''
    (h, w) = img.shape
    bit_mask = np.zeros((h, w, 8), dtype=np.uint8)

    # 创建位平面掩码
    for i in range(8):
        bit_mask[:, :, i] = 2 ** i

    result_img = np.zeros((h, w, 8), dtype=np.uint8)

    cv_show("img_origin", img)

    bit_planes = []  # 用于保存各个切片的列表
    for i in range(7, -1, -1):  # 从高位到低位提取图像
        result_img[:, :, i] = cv2.bitwise_and(img, bit_mask[:, :, i])

        # 为了更加清楚，要将大于零的数处理成255
        mask = result_img[:, :, i] > 0
        result_img[mask] = 255

        bit_planes.append(result_img[:, :, i])  # 将每个切片图像加入列表

    # 合并所有位平面图像（水平拼接）
    combined_image = np.hstack(bit_planes)

    # 显示合并后的图像
    cv_show("Combined Bit Planes", combined_image)


# 读取并调整图像大小
img = cv2.resize(cv2.imread("../pictures/mountain.png", 0), (400, 250))
bit_plain_slice(img)