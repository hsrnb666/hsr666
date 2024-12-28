import numpy as np
import cv2

image = cv2.imread('../pictures/weierlite.png')
cv2.imshow("Original",image)
cv2.waitKey(0)

#R、G、B分量的提取
(B,G,R) = cv2.split(image)#提取R、G、B分量
cv2.imshow("Red",R)
cv2.imshow("Green",G)
cv2.imshow("Blue",B)
cv2.waitKey(0)

img_hsi = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
# 分离H、S、I通道
h_channel, s_channel, i_channel = cv2.split(img_hsi)

cv2.imshow("h",h_channel)
cv2.imshow("s",s_channel)
cv2.imshow("i",i_channel)
cv2.waitKey(0)


