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

# 小窗的偏移量
offset = 20
imgSize = 300


# 数据预处理（必须与训练时相同）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据集
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

while True:
    success, img = cap.read()
    if not success:
        continue
    img = cv2.flip(img, 1)

    # 每100帧计算一次帧率
    frame_count += 1
    if frame_count % 100 == 0:
        fps = frame_count / (time.time() - fps_start_time)
        frame_count = 0
        fps_start_time = time.time()
    # 在图像上显示FPS（每帧都显示）
    current_fps = frame_count / (time.time() - fps_start_time) if frame_count > 0 else 0
    cv2.putText(img, f"FPS: {current_fps:.1f}", (0, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + offset + h, x - offset:x + offset + w]

        # 检查裁剪后的图像是否为空
        if imgCrop.size == 0:
            continue

        aspectRatio = h / w

        try:
            if aspectRatio > 1:  # 高>宽
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:  # 宽>=高
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # 转换为模型需要的格式
            imgGray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
            imgTensor = transform(imgGray).unsqueeze(0)  # 添加batch维度

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
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 80, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, display_text, (x, y - 26),
                        cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.putText(imgOutput, f'{confidence:.2f}', (x, y + 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("imgWhite", imgWhite)

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

