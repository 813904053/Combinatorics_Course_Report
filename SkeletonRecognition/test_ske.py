import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import time
import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets


# 神经网络定义（必须与训练时相同）
class SimpleCNN(nn.Module):
    def __init__(self, num_class):
        super(SimpleCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),

            nn.Linear(32 * 8 * 8, 64), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_class)
        )

    def forward(self, x):
        return self.layers(x)


# 摄像头
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# 置信度阈值
confidence_threshold = 0.7
imgSize = 300

# 数据预处理（必须与训练时相同）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据集（用于获取类别名称）
data_path = "./dataset"
full_dataset = datasets.ImageFolder(root=data_path, transform=transform)

# 获取类别名称
labels = full_dataset.classes
num_classes = len(full_dataset.classes)
print(f"类别: {labels}")
print(f"总样本数: {len(full_dataset)}")
print(f"分类个数：{num_classes}")

# 加载PyTorch模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleCNN(num_classes)
model.load_state_dict(torch.load('gesture_classifier.pth', map_location=device))
model.eval()

# 帧率
fps_start_time = time.time()
frame_count = 0

# 手部骨架连接关系
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


while True:
    success, img = cap.read()
    if not success:
        continue
    img = cv2.flip(img, 1)

    # 帧率计算
    frame_count += 1
    if frame_count % 100 == 0:
        fps = frame_count / (time.time() - fps_start_time)
        frame_count = 0
        fps_start_time = time.time()

    current_fps = frame_count / (time.time() - fps_start_time) if frame_count > 0 else 0
    cv2.putText(img, f"FPS: {current_fps:.1f}", (0, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        lmList = hand["lmList"]  # 获取关键点

        try:
            # 创建骨架图像（用于模型推理）
            skeleton_img = create_skeleton_image(lmList, imgSize)

            # 将骨架图像转换为模型需要的格式
            skeleton_gray = cv2.cvtColor(skeleton_img, cv2.COLOR_BGR2GRAY)
            imgTensor = transform(skeleton_gray).unsqueeze(0)  # 添加batch维度

            # 预测
            with torch.no_grad():
                predictions = model(imgTensor)
                _, predicted = torch.max(predictions, 1)
                index = predicted.item()
                confidence = torch.nn.functional.softmax(predictions, dim=1)[0][index].item()

            if confidence < confidence_threshold:
                display_text = "Unknown"
                display_color = (0, 0, 255)  # 红色
            else:
                display_text = labels[index]
                display_color = (255, 0, 255)  # 粉色

            # 显示结果
            cv2.rectangle(imgOutput, (x, y - 50),
                          (x + 180, y - 10), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, f'{display_text}', (x + 5, y - 25),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(imgOutput, f'Conf: {confidence:.2f}', (x + 5, y - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(imgOutput, (x, y),
                          (x + w, y + h), (255, 0, 255), 2)

            # 显示骨架图像
            cv2.imshow("Skeleton Image", skeleton_img)

        except Exception as e:
            print(f"处理错误: {e}")
            continue

    # 显示
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

    # 按'q'键退出循环
    if key == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()