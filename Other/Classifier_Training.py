import os
import zipfile
import random
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix


def main():
    archive_path = 'final_crop.zip'
    extracted_path = 'datasetV1'

    if not os.path.exists(extracted_path):
        print("Распаковываем архив...")
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_path)
    else:
        print("Архив уже распакован")

    # --- 2. Разбиение датасета на тестовую и валидационную выборку ---
    train_dir = 'datasetV1_train'
    val_dir = 'datasetV1_val'
    split_ratio = 0.8  # 80% на обучение, 20% на валидацию

    # Создаём директории для train и val
    for base_dir in [train_dir, val_dir]:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    # Проходим по каждой папке класса в распакованном датасете
    for class_name in os.listdir(extracted_path):
        class_path = os.path.join(extracted_path, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            random.shuffle(images)
            split_idx = int(len(images) * split_ratio)
            train_images = images[:split_idx]
            val_images = images[split_idx:]

            # Создаем папки для класса в train и val
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)

            # Копируем изображения в соответствующие папки
            for img in train_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(train_class_dir, img)
                shutil.copy(src, dst)
            for img in val_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(val_class_dir, img)
                shutil.copy(src, dst)

    print("Датасет разделён на train и val")

    # --- 3. Подготовка трансформаций и загрузка датасетов ---
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), # Преобразование изображение в оттенки серого. Так как изображения в датасете преимущественно черно-белые
        transforms.Resize((224, 224)), # Размер изображения
        transforms.RandomHorizontalFlip(), # Аугментация, переворот изображения
        transforms.RandomRotation(10), # Аугментация, поворачивает изображение на угол +- 10 градусов
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Аугментация, изменяет яркость и контракст изображения
        transforms.ToTensor(), # Преобразуется изображения в тензоры
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]) # Нормализация изображения

    val_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=val_transforms)

    # --- 4. Балансировка классов ---
    targets = [label for _, label in train_dataset.samples]
    unique_classes = np.unique(targets)
    class_counts = {cls: targets.count(cls) for cls in unique_classes}
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    samples_weights = [class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # --- 5. Инициализация модели, оптимизатора и обучение ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 15)  # 15 классов
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    def calculate_accuracy(outputs, labels):
        _, predicted = outputs.max(1)
        return (predicted == labels).sum().item()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", leave=False)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct_train += calculate_accuracy(outputs, labels)
            total_train += labels.size(0)
            train_bar.set_postfix(loss=loss.item())

        train_loss = running_loss / total_train
        train_acc = 100. * correct_train / total_train

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation", leave=False)

        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                correct_val += calculate_accuracy(outputs, labels)
                total_val += labels.size(0)
                val_bar.set_postfix(loss=loss.item())

        val_loss = running_val_loss / total_val
        val_acc = 100. * correct_val / total_val

        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # --- 6. Расчёт дополнительных метрик на валидационном наборе ---
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Calculating metrics"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # --- 7. Сохранение весов модели ---
    torch.save(model.state_dict(), 'modelV1.pth')
    print("Веса модели сохранены в файле modelV1.pth")


if __name__ == '__main__':
    main()
