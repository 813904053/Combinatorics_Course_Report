import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
import time
from thop import profile  # 用于计算FLOPs和参数量


# 神经网络定义
# 深度可分离卷积模块（用于轻量级模型）
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                    stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        # 逐点卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)  # 使用bias
        # 只在最后使用一个ReLU
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.pointwise(x)  # 直接连接，减少操作
        x = self.relu(x)       # 只在最后激活一次
        return x

# 通道注意力模块（用于注意力机制）
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 添加max_pool，确保与forward方法中使用的一致
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

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

# 轻量级模型 - 使用深度可分离卷积减少参数量
class MobileNetLike(nn.Module):
    def __init__(self, num_class):
        super(MobileNetLike, self).__init__()
        self.layers = nn.Sequential(
            # 第一层：标准卷积
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 第二层：深度可分离卷积
            DepthwiseSeparableConv(16, 32, 1),
            nn.MaxPool2d(2),
            
            # 第三层：深度可分离卷积
            DepthwiseSeparableConv(32, 32, 1),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            
            nn.Linear(32 * 8 * 8, 64), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_class)
        )

    def forward(self, x):
        return self.layers(x)

# 带注意力机制的模型
class AttentionCNN(nn.Module):
    def __init__(self, num_class):
        super(AttentionCNN, self).__init__()
        # 特征提取部分
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        
        # 只在最后一层添加注意力
        self.attention = ChannelAttention(32, reduction_ratio=8)
        
        # 分类器（与SimpleCNN相同）
        self.avg_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_class)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # 应用注意力
        x = self.attention(x)
        
        x = self.avg_pool(x)
        x = self.classifier(x)
        
        return x

# 模型工厂类
class ModelFactory:
    @staticmethod
    def create_model(model_name, num_classes, device):
        models = {
            'SimpleCNN': SimpleCNN,
            'MobileNetLike': MobileNetLike,
            'AttentionCNN': AttentionCNN
        }
        
        if model_name not in models:
            raise ValueError(f"未知模型: {model_name}。可用模型: {list(models.keys())}")
        
        model = models[model_name](num_classes).to(device)
        return model
    
    @staticmethod
    def get_available_models():
        return ['SimpleCNN', 'MobileNetLike', 'AttentionCNN']


# 模型评估器
class ModelEvaluator:
    def __init__(self, device):
        self.device = device
    
    def compute_model_complexity(self, model, input_size=(1, 1, 300, 300)):
        """计算模型复杂度和参数量"""
        dummy_input = torch.randn(input_size).to(self.device)
        flops, params = profile(model, inputs=(dummy_input,))
        return flops, params
    
    def measure_inference_time(self, model, test_loader, num_batches=10):
        """测量推理时间"""
        model.eval()
        times = []
        
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                if i >= num_batches:
                    break
                x = x.to(self.device)
                
                start_time = time.time()
                _ = model(x)
                end_time = time.time()
                
                times.append(end_time - start_time)
        
        return np.mean(times), np.std(times)


# 训练过程
def trainer(train_loader, model, device, config, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)  # 添加权重衰减
    n_epochs = config['n_epochs']
    train_losses = []
    train_accuracies = []
    
    print(f"\n=== 训练 {model_name} ===")
    
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
        train_accuracies.append(epoch_acc)
        


        print(f'Epoch[{epoch + 1}/{n_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    return {
        'losses': train_losses,
        'accuracies': train_accuracies,
        'final_accuracy': train_accuracies[-1]
    }


# 测试
def predict(test_loader, model, device, class_names, model_name):
    model.eval()
    correct, total = 0, 0
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            _, predicted = torch.max(pred.data, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            # 计算每个类别的准确率
            for i in range(len(y)):
                label = y[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    accuracy = 100 * correct / total
    
    # 计算每个类别的准确率
    class_accuracies = {}
    for i in range(len(class_names)):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            class_accuracies[class_names[i]] = class_acc

    return {
        'overall_accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'predictions': all_preds,
        'labels': all_labels
    }


# 比较多个模型
def compare_models(model_names, train_loader, test_loader, num_classes, device, config, class_names):
    results = {}
    evaluator = ModelEvaluator(device)
    
    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"处理模型: {model_name}")
        print(f"{'='*50}")
        
        # 创建模型
        model = ModelFactory.create_model(model_name, num_classes, device)
        
        # 计算模型复杂度
        flops, params = evaluator.compute_model_complexity(model)
        
        # 训练模型
        train_results = trainer(train_loader, model, device, config, model_name)
        
        # 测试模型
        test_results = predict(test_loader, model, device, class_names, model_name)
        
        # 测量推理速度
        inference_time, inference_std = evaluator.measure_inference_time(model, test_loader)
        
        # 保存结果
        results[model_name] = {
            'model': model,
            'train_results': train_results,
            'test_results': test_results,
            'complexity': {
                'flops': flops,
                'params': params
            },
            'inference_time': inference_time,
            'inference_std': inference_std
        }
        
        # 打印当前模型结果
        print(f"\n{model_name} 结果:")
        print(f"  最终训练准确率: {train_results['final_accuracy']:.2f}%")
        print(f"  测试准确率: {test_results['overall_accuracy']:.2f}%")
        print(f"  参数量: {params/1e6:.2f}M")
        print(f"  FLOPs: {flops/1e9:.2f}G")
        print(f"  推理时间: {inference_time*1000:.2f}ms ± {inference_std*1000:.2f}ms")
    
    return results


# 可视化比较结果
def visualize_comparison(results, class_names):
    # 准备数据
    model_names = list(results.keys())
    test_accuracies = [results[name]['test_results']['overall_accuracy'] for name in model_names]
    train_accuracies = [results[name]['train_results']['final_accuracy'] for name in model_names]
    params = [results[name]['complexity']['params'] / 1e6 for name in model_names]  # 转换为百万
    inference_times = [results[name]['inference_time'] * 1000 for name in model_names]  # 转换为毫秒
    
    # 创建比较图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 准确率比较
    x = range(len(model_names))
    width = 0.35
    ax1.bar([i - width/2 for i in x], train_accuracies, width, label='Train Accuracy', alpha=0.7)
    ax1.bar([i + width/2 for i in x], test_accuracies, width, label='Test Accuracy', alpha=0.7)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for i, v in enumerate(train_accuracies):
        # 使用动态偏移量，让数值更接近柱子
        offset = max(0.5, v * 0.01)  # 动态调整偏移量，最小为0.5
        ax1.text(i - width/2, v + offset, f'{v:.1f}%', ha='center', va='bottom')
    for i, v in enumerate(test_accuracies):
        # 使用动态偏移量，让数值更接近柱子
        offset = max(0.5, v * 0.01)  # 动态调整偏移量，最小为0.5
        ax1.text(i + width/2, v + offset, f'{v:.1f}%', ha='center', va='bottom')
    
    # 参数量比较
    ax2.bar(model_names, params, color='orange', alpha=0.7)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Parameters (Million)')
    ax2.set_title('Model Parameters Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值
    for i, v in enumerate(params):
        # 使用一个较小的偏移量，让数值更接近柱子
        offset = 0.001  
        ax2.text(i, v + offset, f'{v:.2f}M', ha='center', va='bottom')
    
    # 推理时间比较
    ax3.bar(model_names, inference_times, color='green', alpha=0.7)
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Inference Time (ms)')
    ax3.set_title('Model Inference Time Comparison')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    for i, v in enumerate(inference_times):
        # 使用动态偏移量，让数值更接近柱子
        offset = 0.001 
        ax3.text(i, v + offset, f'{v:.2f}ms', ha='center', va='bottom')
    
    # 训练损失曲线
    for model_name in model_names:
        losses = results[model_name]['train_results']['losses']
        ax4.plot(range(1, len(losses) + 1), losses, label=model_name, marker='o')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Training Loss Curves')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细比较表格
    print(f"\n{'='*60}")
    print("Model Performance Comparison Summary")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'Test Acc':<12} {'Params':<12} {'Inference':<12}")
    print(f"{'-'*60}")
    for model_name in model_names:
        result = results[model_name]
        print(f"{model_name:<12} {result['test_results']['overall_accuracy']:<11.2f}% "
              f"{result['complexity']['params']/1e6:<11.2f}M "
              f"{result['inference_time']*1000:<11.2f}ms")


# 主程序
def main():
    # 参数设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'learning_rate': 0.001,
        'n_epochs': 20,  # 训练轮数
        'batch_size': 16  # 批量大小
    }

    print(f"使用设备: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_path = "./SkeletonRecognition/dataset"

    # 检查路径是否存在
    if not os.path.exists(data_path):
        print(f"错误: 路径 {data_path} 不存在!")
        return

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

    # 要比较的模型列表
    model_names = ModelFactory.get_available_models()
    print(f"\n要比较的模型: {model_names}")

    # 比较所有模型
    results = compare_models(
        model_names=model_names,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        device=device,
        config=config,
        class_names=class_names
    )

    # 可视化比较结果
    visualize_comparison(results, class_names)

    # 保存最佳模型
    best_model_name = max(results.keys(), 
                         key=lambda x: results[x]['test_results']['overall_accuracy'])
    best_model = results[best_model_name]['model']
    
    model_path = os.path.join("./SkeletonRecognition/", f'best_model_{best_model_name}.pth')
    torch.save(best_model.state_dict(), model_path)
    print(f"\n最佳模型已保存: {best_model_name}")
    print(f"保存路径: {os.path.abspath(model_path)}")


if __name__ == "__main__":
    main()