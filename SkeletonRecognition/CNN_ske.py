import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os


# 训练过程
def trainer(train_loader, model, device, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    n_epochs = config['n_epochs']
    train_losses = []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)

        print(f'Epoch[{epoch + 1}/{n_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    # 画图
    plt.plot(range(1, n_epochs + 1), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()

    return train_losses


# 测试
def predict(test_loader, model, device, class_names):
    model.eval()
    correct, total = 0, 0
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            _, predicted = torch.max(pred.data, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

            # 计算每个类别的准确率
            for i in range(len(y)):
                label = y[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    accuracy = 100 * correct / total
    print(f'测试集准确度: {accuracy:.2f}%')

    # 打印每个类别的准确率
    print("\n=== 各类别准确率 ===")
    for i in range(len(class_names)):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f'{class_names[i]}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})')


# 神经网络（使用你原来的简单版本）
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


# 参数设置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'learning_rate': 0.001,
    'n_epochs': 15,
    'batch_size': 8
}

print(f"使用设备: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_path = "./dataset"

# 检查路径是否存在
if not os.path.exists(data_path):
    print(f"错误: 路径 {data_path} 不存在!")
else:
    # 加载数据集
    full_dataset = datasets.ImageFolder(root=data_path, transform=transform)

    # 获取类别名称
    class_names = full_dataset.classes
    num_classes = len(full_dataset.classes)
    print(f"类别: {class_names}")
    print(f"总样本数: {len(full_dataset)}")
    print(f"分类个数：{num_classes}")
    # 划分训练集和测试集
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_Data, test_Data = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    print(f"训练集样本数: {len(train_Data)}")
    print(f"测试集样本数: {len(test_Data)}")

    # 准备Dataloader
    train_loader = DataLoader(train_Data, shuffle=True, batch_size=config['batch_size'])
    test_loader = DataLoader(test_Data, shuffle=False, batch_size=config['batch_size'])

    # 模型
    model = SimpleCNN(num_classes).to(device)

    # 训练
    print("\n=== 开始训练 ===")
    trainer(train_loader, model, device, config)

    # 测试
    print("\n=== 测试结果 ===")
    predict(test_loader, model, device, class_names)

    # 保存模型
    torch.save(model.state_dict(), 'gesture_classifier.pth')
    print("\n模型已保存为 'gesture_classifier.pth'")

print('done!')