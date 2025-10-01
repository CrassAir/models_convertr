import os
import cv2
import glob
import random
from ultralytics import YOLO

def visualize_validation(
    model_weights: str,
    val_images_dir: str = "ttn_dataset/val/images",
    output_dir: str = "runs/ttn_detection/visuals",
    num_samples: int = 5,
    conf_threshold: float = 0.25
):
    """
    Визуальная валидация предсказаний YOLO на случайных валидационных изображениях.

    Args:
        model_weights: путь к файлу весов (best.pt).
        val_images_dir: папка с валидационными изображениями.
        output_dir: папка для сохранения визуализаций.
        num_samples: сколько случайных примеров показать.
        conf_threshold: порог уверенности для отрисовки боксов.
    """
    # Создать выходную папку
    os.makedirs(output_dir, exist_ok=True)

    # Загрузить модель
    model = YOLO(model_weights)

    # Получить список всех изображений
    images = glob.glob(f"{val_images_dir}/*.jpg") + glob.glob(f"{val_images_dir}/*.png")
    samples = random.sample(images, min(num_samples, len(images)))

    for img_path in samples:
        img = cv2.imread(img_path)

        # Предсказание
        results = model.predict(source=img_path, conf=conf_threshold, show=False)

        # Рисуем боксы на копии
        vis = img.copy()
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                if conf < 0.8 and cls_id == 8:
                    continue
                label = f"{str(cls_id)} {conf:.2f}"
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(vis, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Сохранить результат
        fname = os.path.basename(img_path)
        out_path = os.path.join(output_dir, f"vis_{fname}")
        cv2.imwrite(out_path, vis)
        print(f"Сохранено: {out_path}")

if __name__ == "__main__":
    # Пример вызова: визуализация на 5 случайных изображениях
    visualize_validation(
        model_weights="best.pt",
        val_images_dir="ttn_dataset/test",
        output_dir="runs/ttn_detection/visuals",
        num_samples=10,
        conf_threshold=0.3
    )
