import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import time
import os

# 摄像头
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# 骨架图像的尺寸
imgSize = 300

# 存储
while True:
    label = input("请输入数字标签 (0-9): ")
    if label.isdigit():  # 检查是否是数字
        folder = os.path.join("./SkeletonRecognition/dataset", label)
        print(f"数据将保存到: {folder}")
        break
    else:
        print("输入错误，请输入数字！")
counter = 0

# 手部骨架连接关系（MediaPipe标准）
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
    (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
]


def create_skeleton_image(landmarks, img_size=300):
    """创建纯骨架图像，保持手部原始比例"""
    # 创建空白黑色背景
    skeleton_img = np.zeros((img_size, img_size, 3), np.uint8)

    if len(landmarks) < 21:
        return skeleton_img

    # 获取所有点的坐标范围
    x_coords = [lm[0] for lm in landmarks]
    y_coords = [lm[1] for lm in landmarks]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # 计算手部原始宽高和比例
    hand_width = x_max - x_min
    hand_height = y_max - y_min
    hand_ratio = hand_width / hand_height if hand_height > 0 else 1

    # 根据手部比例确定缩放方式
    if hand_ratio > 1:  # 宽大于高（手掌张开）
        scale = (img_size - 20) / hand_width
        scaled_height = int(hand_height * scale)
        # 垂直居中
        y_offset = (img_size - scaled_height) // 2
        x_offset = 10
    else:  # 高大于宽（手掌合拢）
        scale = (img_size - 20) / hand_height
        scaled_width = int(hand_width * scale)
        # 水平居中
        x_offset = (img_size - scaled_width) // 2
        y_offset = 10

    # 归一化坐标，保持比例
    normalized_landmarks = []
    for lm in landmarks:
        nx = int((lm[0] - x_min) * scale) + x_offset
        ny = int((lm[1] - y_min) * scale) + y_offset
        # 确保坐标在图像范围内
        nx = max(0, min(nx, img_size - 1))
        ny = max(0, min(ny, img_size - 1))
        normalized_landmarks.append((nx, ny))

    # 绘制骨架连接线（白色）
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(normalized_landmarks) and end_idx < len(normalized_landmarks):
            start_point = normalized_landmarks[start_idx]
            end_point = normalized_landmarks[end_idx]
            cv2.line(skeleton_img, start_point, end_point, (255, 255, 255), 2)

    # 绘制关节点（白色圆点）
    for point in normalized_landmarks:
        cv2.circle(skeleton_img, point, 4, (255, 255, 255), -1)

    return skeleton_img


if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    if not success:
        continue
    img = cv2.flip(img, 1)

    # 检测手部并绘制在原图上
    hands, img_with_hands = detector.findHands(img, draw=True)  # 同时获取hands和绘制后的图像

    skeleton_img = np.zeros((imgSize, imgSize, 3), np.uint8)  # 默认黑色图像

    if hands:  # 如果检测到手部
        hand = hands[0]
        lmList = hand["lmList"]  # 获取21个关键点坐标

        # 创建骨架图像
        skeleton_img = create_skeleton_image(lmList, imgSize)

    # 显示原图和骨架图
    cv2.imshow("Original Image", img_with_hands)
    cv2.imshow("Skeleton Image", skeleton_img)

    key = cv2.waitKey(1)

    # 按's'键保存骨架图像（只在检测到手部时保存）
    if key == ord("s") and hands:
        counter += 1
        # 使用os.path.join确保路径分隔符在不同操作系统上正确工作
        img_path = os.path.join(folder, f'{label}_Skeleton_{time.time()}.png')
        cv2.imwrite(img_path, skeleton_img)
        print(f"保存第 {counter} 张骨架图像")

    # 按'q'键退出循环
    if key == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()