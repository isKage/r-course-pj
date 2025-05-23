import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import confusion_matrix

from models import ShallowCNNModel, CNNModel

dataset_dir = "data/CT"  # /cancer and /normal
param_path = "checkpoints/opt_model.pth"
log_path = "logs/logs.txt"
logs = ""  # 记录日志

# 训练集图像变化
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 验证集图像变化
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 1. 加载数据
full_dataset = datasets.ImageFolder(root=dataset_dir, transform=val_test_transform)

print("Classes:", full_dataset.classes)
print("Total images:", len(full_dataset))
logs += f"Classes:, {full_dataset.classes}\n"
logs += f"Total images: {len(full_dataset)}\n\n"

train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
train_dataset.dataset.transform = train_transform

batch_size = 16
num_classes = 2

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("DataLoaders created successfully!")
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
logs += f"DataLoaders created successfully!\n"
logs += f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}\n\n"
logs += f"Batch size: {batch_size}\n"
logs += f"Number of classes: {num_classes}\n"

# 2. 模型搭建
model = CNNModel(num_classes=num_classes)  # CNNModel 和 ShallowCNNModel 2 个模型
logs += f"Model created successfully!\n"

# CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logs += f"Device: {device}\n"

# 损失函数和调优器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
logs += f"Loss: CrossEntropyLoss\n"
logs += f"Optimizer: Adam\n"
logs += f"Learning Rate: {optimizer.param_groups[0]['lr']}\n"

# 超参数
num_epochs = 60
patience = 4  # Stop if no improvement after 4 epochs
best_val_acc = 0.0
epochs_no_improve = 0
logs += f"Epochs Num: {num_epochs}\n"
logs += f"Patience: {patience}\n"

# 3. 开始训练
logs += "\nTraining Started!\n"
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    logs += f"\n=== Epoch {epoch + 1}/{num_epochs} ===\n"

    # ===================== Training =====================
    model.train()
    total, correct, train_loss = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    print(f"Train Loss: {train_loss / total:.4f} | Train Acc: {train_acc:.2f}%")
    logs += f"Train Loss: {train_loss / total:.4f} | Train Acc: {train_acc:.2f}%\n"

    # ===================== Validation =====================
    model.eval()
    total, correct, val_loss = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f"Val Loss: {val_loss / total:.4f} | Val Acc: {val_acc:.2f}%")
    logs += f"Val Loss: {val_loss / total:.4f} | Val Acc: {val_acc:.2f}%\n"

    # ===================== Early Stopping =====================
    if val_acc > best_val_acc:  # 验证集表现更好, 保存模型
        best_val_acc = val_acc
        epochs_no_improve = 0
        print("Validation improved, saving model...")
        torch.save(model.state_dict(), param_path)
        logs += f"Validation improved, saving model at \"{param_path}\"\n"
    else:  # 未提升
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epochs.")
        logs += f"No improvement for {epochs_no_improve} epochs.\n"

    if epochs_no_improve >= patience:  # 未提升大于 4 轮, 提前结束
        print("Early stopping triggered!")
        logs += f"Early stopping triggered!\n"
        break

try:
    logs += f"\nFinal Train Loss: {train_loss / total:.4f} | Train Acc: {train_acc:.2f}%"
    logs += f"\nFinal Val Loss: {val_loss / total:.4f} | Val Acc: {val_acc:.2f}%"
except ZeroDivisionError:
    pass

# 4. 测试
model.load_state_dict(torch.load(param_path))
model.to(device)
model.eval()

total, correct, test_loss = 0, 0, 0
all_labels, all_preds = [], []

# 开始测试
logs += "\n\nTesting Started!\n\n"
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * images.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

test_loss /= total
test_acc = 100 * correct / total
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
logs += f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%\n"

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:\n", cm)
logs += f"\nConfusion Matrix:\n{cm}"

with open(log_path, "w") as f:
    f.write(logs)
