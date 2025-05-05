# predict.py
import torch
from torchvision import models, transforms
from PIL import Image

# 1. 加载模型
model = models.resnet18()
model.fc = torch.nn.Linear(512, 2)  # 与训练时保持一致
model.load_state_dict(torch.load("cat_dog_model.pth"))
model.eval()

# 2. 预处理定义（必须与训练时一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 3. 加载图片
img = Image.open("test.jpg").convert("RGB")
img = transform(img).unsqueeze(0)  # [1, 3, 224, 224]

# 4. 推理
with torch.no_grad():
    outputs = model(img)
    _, pred = torch.max(outputs, 1)
    label = "狗" if pred.item() == 1 else "猫"
    print(f"预测结果：{label}")
