import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import albumentations as A  # make the data stronger
from albumentations.pytorch import ToTensorV2
from model import ImprovedIdentify

# ======================================================================================================================

class CustomImageDataset(Dataset):
    def __init__(self, main_folder, transform=None):
        self.main_folder = main_folder
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []

        for label, class_folder in enumerate(os.listdir(main_folder)):
            class_path = os.path.join(main_folder, class_folder)
            if not os.path.isdir(class_path):
                continue
            self.class_names.append(class_folder)

            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                self.images.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img = np.array(img)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        return img, label

# ======================================================================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30):
    device = model.device
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0
    best_model_weights = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim = 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        scheduler.step(epoch_val_loss)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_weights = model.state_dict()

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')
        print('-' * 50)

    model.load_state_dict(best_model_weights)

    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    return model

# ======================================================================================================================

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))

    # loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

# ======================================================================================================================

def evaluate_model(model, dataloader, class_names):
    model.eval()
    device = model.device
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names))
    return all_preds, all_labels

# ======================================================================================================================

def main():
    data_folder = r"C:\Users\72472\Desktop\Cambridge\Myself_data"

    train_transform = A.Compose([
        A.Resize(300, 300),
        A.RandomRotate90(),
        A.Flip(),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # 高斯模糊
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 使用ImageNet的均值和标准差
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(300, 300),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    full_dataset = CustomImageDataset(
        main_folder=data_folder,
        transform=None
    )

    # train-70% val-15% test-15%
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"总样本数: {len(full_dataset)}")
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
    print(f"类别: {full_dataset.class_names}")

    model = ImprovedIdentify(num_classes=len(full_dataset.class_names))
    print(f"使用设备: {model.device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )

    print("开始训练模型...")
    trained_model = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30
    )

    print("\n训练集评估:")
    evaluate_model(trained_model, train_loader, full_dataset.class_names)

    print("\n验证集评估:")
    evaluate_model(trained_model, val_loader, full_dataset.class_names)

    print("\n测试集评估:")
    evaluate_model(trained_model, test_loader, full_dataset.class_names)

    save_path1 = r"C:\Users\72472\Desktop\Cambridge\improved_model.pth"
    torch.save(trained_model.state_dict(), save_path1)
    print(f"\n模型已保存至: {save_path1}")


if __name__ == '__main__':
    main()
