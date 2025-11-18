import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import time
import os

# 摄像头
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# 小窗的偏移量
offset = 20

# 白窗的边长
imgSize = 300

# 存储
while True:
    label = input("请输入数字标签 (0-9): ")
    if label.isdigit():  # 检查是否是数字
        folder = "./dataset/" + label
        print(f"数据将保存到: {folder}")
        break
    else:
        print("输入错误，请输入数字！")

counter = 0
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    if not success:
        continue
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]

        # x: 边界框左上角的 x 坐标（水平位置）
        # y: 边界框左上角的 y 坐标（垂直位置）
        # w: 边界框的宽度（width）
        # h: 边界框的高度（height）
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + offset + h, x - offset:x + offset + w]

        # 检查裁剪后的图像是否为空
        if imgCrop.size == 0:
            continue

        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:  # 长<宽
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape

            # 让其在白窗居中
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("imgWhite", imgWhite)

    # 显示
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # 按's'键保存图像
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/{label}_Image_{time.time()}.png', imgWhite)
        print(counter)

    # 按'q'键退出循环
    if key == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

