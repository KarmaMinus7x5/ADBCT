import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

# Список имен классов в том же порядке, в котором они были обучены
class_names = [
    "Sus", "Putorius", "OTHER ANIMAL", "Nyctereutes", "Neovison",
    "Meles", "Martes", "Lynx", "Lutra", "Lepus",
    "Cnippon", "Capreolus", "Canis lupus", "Bison", "Alces"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Загрузка детекционной модели (Faster R-CNN) ---
detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
detection_model.to(device)
detection_model.eval()

# --- 2. Загрузка классификационной модели ---
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
classification_model = torchvision.models.efficientnet_b0(weights=weights)
num_features = classification_model.classifier[1].in_features
classification_model.classifier[1] = torch.nn.Linear(num_features, 15)  # 15 классов
# Загружаем дообученные веса
classification_model.load_state_dict(torch.load('efficientnet_b0_final.pth', map_location=device))
classification_model.to(device)
classification_model.eval()

# --- 3. Определение трансформаций для классификации ---
# Трансформации должны соответствовать тем, что использовались на этапе валидации
classification_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- 4. Параметры порогов ---
# Порог для обнаружения
detection_threshold = 0.7
# Порог для классификации
classification_threshold = 0.5

# --- 5. Открытие видео и подготовка для сохранения результата ---
input_video_path = "vid2.mp4"   # путь к вашему видео
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video2.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразуем BGR в RGB и создаем тензор для детекции
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = F.to_tensor(frame_rgb).to(device)

    with torch.no_grad():
        detections = detection_model([frame_tensor])[0]

    boxes = detections['boxes']
    scores = detections['scores']

    for i in range(len(scores)):
        if scores[i] < detection_threshold:
            continue
        # Получаем координаты бокса и приводим к целым числам
        box = boxes[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        # Извлекаем регион из оригинального кадра
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Преобразуем crop в формат PIL (BGR -> RGB) и применяем трансформации для классификации
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crop_tensor = classification_transform(crop_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = classification_model(crop_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)
            conf = conf.item()
            pred = pred.item()

        # Если уверенность классификации ниже порога, пропускаем этот бокс
        if conf < classification_threshold:
            continue

        label_text = f"{class_names[pred]}: {conf*100:.1f}%"
        # Отрисовка бокса и подписи на исходном кадре
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Записываем обработанный кадр в выходное видео
    out.write(frame)
    # Отображаем кадр в реальном времени
    cv2.imshow("Processed Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Обработка видео завершена. Результат сохранён в файле output_video.mp4")
