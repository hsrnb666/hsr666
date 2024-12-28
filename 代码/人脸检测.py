import cv2
import matplotlib.pyplot as plt

# 加载OpenCV自带的Haar Cascade人脸检测模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 读取图像
image = cv2.imread('../pictures/renlian.jpeg')

# 转为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用Haar Cascade进行人脸检测
# 参数解释：scaleFactor和minNeighbors是检测过程中的两个重要参数。
# scaleFactor：在每一层图像金字塔上，图像的大小减小的比例。
# minNeighbors：每个目标矩形附近的邻居个数（默认值为3），这有助于减少误检。
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 绘制矩形框标记人脸
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 在图像上画矩形框

# 转换为RGB显示图像（OpenCV以BGR格式读取图像）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 显示图像
plt.imshow(image_rgb)
plt.title('Face Detection')
plt.axis('off')
plt.show()