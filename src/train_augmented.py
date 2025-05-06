import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.datasets.folder import default_loader
from PIL import UnidentifiedImageError



DATA_DIR = 'data/PetImages'

BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 安全版图像加载函数
def safe_loader(path):
    try:
        return default_loader(path)
    except UnidentifiedImageError:
        print(f"[警告] 跳过坏图像文件：{path}")
        return None

# 自定义 ImageFolder 类：跳过 loader 出错的图像
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        # 无限尝试直到读到有效图片
        while True:
            try:
                sample, target = super().__getitem__(index)
                return sample, target
            except UnidentifiedImageError:
                print(f"[警告] 跳过坏图像文件: {self.samples[index][0]}")
                index = (index + 1) % len(self)


# 图像预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 加载整个图像数据集（自动识别 Cat 和 Dog 文件夹）
dataset = SafeImageFolder(DATA_DIR)

# 将数据集随机划分为训练集和验证集（80%:20%）
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# 构建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 验证划分数量
print("训练集中样本数量：", len(train_dataset))
print("验证集中样本数量：", len(val_dataset))

# 加载预训练的 ResNet18 模型
model = models.resnet18(pretrained=True)

# 获取原来的输出特征数
in_features = model.fc.in_features

# 替换最后一层为2类输出（猫 vs 狗）
model.fc = nn.Linear(in_features, 2)

model = model.to(device)

print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    model.train()  # 切换到训练模式
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()           # 清除旧梯度
        outputs = model(images)         # 正向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()                 # 反向传播
        optimizer.step()                # 参数更新

        running_loss += loss.item() * images.size(0)

    avg_train_loss = running_loss / len(train_loader.dataset)

    # 验证过程
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")

torch.save(model.state_dict(), "cat_dog_model.pth")
