# Код для дообучения YOLO модели на детекцию блоков в транспортных накладных (ТТН)

import os
import yaml
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class TTNBlockDetector:
    """
    Класс для дообучения YOLO модели на детекцию блоков в ТТН документах
    """
    
    def __init__(self, base_model="yolo11n.pt"):
        """
        Инициализация детектора
        
        Args:
            base_model (str): Базовая предобученная модель YOLO
        """
        self.base_model = base_model
        self.model = None
        self.dataset_path = None
        
    def prepare_dataset_config(self, dataset_path, class_names):
        """
        Создает конфигурационный файл для датасета
        
        Args:
            dataset_path (str): Путь к датасету
            class_names (list): Список имен классов для детекции
        """
        self.dataset_path = Path(dataset_path)
        
        # Создание структуры папок если не существует
        (self.dataset_path / "train" / "images").mkdir(parents=True, exist_ok=True)
        (self.dataset_path / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (self.dataset_path / "val" / "images").mkdir(parents=True, exist_ok=True)
        (self.dataset_path / "val" / "labels").mkdir(parents=True, exist_ok=True)
        
        # Создание YAML конфига
        config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(class_names),
            'names': {i: name for i, name in enumerate(class_names)}
        }
        
        config_path = self.dataset_path / "dataset.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
        print(f"Конфигурационный файл создан: {config_path}")
        return config_path
    
    def train_model(self, config_path, epochs=100, imgsz=640, batch_size=16, lr0=0.01):
        """
        Обучение YOLO модели на кастомном датасете
        
        Args:
            config_path (str): Путь к конфигурационному файлу датасета
            epochs (int): Количество эпох обучения
            imgsz (int): Размер входного изображения
            batch_size (int): Размер батча
            lr0 (float): Начальная скорость обучения
        """
        # Загрузка базовой модели
        self.model = YOLO(self.base_model)
        
        print(f"Начинаем обучение модели {self.base_model}")
        print(f"Датасет: {config_path}")
        print(f"Эпохи: {epochs}, Размер изображения: {imgsz}, Батч: {batch_size}")
        
        # Обучение модели
        results = self.model.train(
            data=str(config_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            lr0=lr0,
            save=True,
            project='runs/ttn_detection2',
            name='ttn_blocks_detection',
            patience=20,  # Early stopping
            augment=True,  # Аугментация данных
            hsv_h=0.015,   # Параметры аугментации цвета
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0,     # Поворот отключен для документов
            translate=0.1, # Небольшие сдвиги
            scale=0.1,     # Небольшое масштабирование
            shear=0,       # Сдвиг отключен
            perspective=0, # Перспектива отключена
            flipud=0,      # Вертикальный флип отключен
            fliplr=0.5,    # Горизонтальный флип для документов
            mosaic=0.5,    # Мозаичная аугментация
            mixup=0.1      # Смешивание изображений
        )
        
        print("Обучение завершено!")
        return results
    
    def validate_model(self, model_path=None):
        """
        Валидация обученной модели
        
        Args:
            model_path (str): Путь к обученной модели (если None, используется последняя)
        """
        if model_path:
            model = YOLO(model_path)
        else:
            model = self.model
            
        if model is None:
            print("Модель не загружена. Сначала обучите модель или укажите путь.")
            return
            
        print("Проведение валидации...")
        results = model.val()
        
        print(f"mAP50: {results.box.map50}")
        print(f"mAP50-95: {results.box.map}")
        
        return results
    
    def predict_ttn_blocks(self, image_path, model_path=None, conf_threshold=0.25):
        """
        Детекция блоков в ТТН документе
        
        Args:
            image_path (str): Путь к изображению ТТН
            model_path (str): Путь к обученной модели
            conf_threshold (float): Порог уверенности
        """
        if model_path:
            model = YOLO(model_path)
        else:
            model = self.model
            
        if model is None:
            print("Модель не загружена.")
            return None
            
        # Предсказание
        results = model.predict(
            source=image_path,
            conf=conf_threshold,
            save=True,
            project='runs/ttn_detection',
            name='predictions'
        )
        
        return results
    
    def extract_blocks_coordinates(self, results):
        """
        Извлечение координат обнаруженных блоков
        
        Args:
            results: Результаты детекции YOLO
            
        Returns:
            list: Список координат блоков [(x1, y1, x2, y2, class_id, confidence), ...]
        """
        blocks = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # Координаты x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()  # Уверенность
                class_ids = result.boxes.cls.cpu().numpy()  # ID классов
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    blocks.append({
                        'coordinates': box.tolist(),  # [x1, y1, x2, y2]
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'class_name': result.names[int(cls_id)]
                    })
        
        return blocks

def create_sample_training_script():
    """
    Пример использования TTNBlockDetector для обучения
    """
    # Инициализация детектора
    detector = TTNBlockDetector(base_model="yolo11n.pt")  # Используем Small модель
    
    # Определение классов для детекции в ТТН
    ttn_classes = [   
        "cargo_name",
        "cargo_weight",
        "count",
        "date",
        "document_number",
        "receiver",
        "sender",
        "stamp",
    ]
    
    # Подготовка конфигурации датасета
    dataset_path = "ttn_dataset"
    config_path = detector.prepare_dataset_config(dataset_path, ttn_classes)
    
    print("Структура датасета создана. Теперь поместите:")
    print("- Изображения ТТН в папки train/images и val/images")
    print("- Соответствующие аннотации в формате YOLO в папки train/labels и val/labels")
    print("\nФормат аннотаций YOLO (в .txt файлах):")
    print("class_id x_center y_center width height")
    print("где координаты нормализованы от 0 до 1")
    
    # Обучение модели (раскомментируйте после подготовки датасета)
    results = detector.train_model(
        config_path=config_path,
        epochs=200,           # Больше эпох для документов
        imgsz=1024,          # Больший размер для документов
        batch_size=8,        # Меньший батч из-за большего размера
        lr0=0.001           # Меньшая скорость обучения для fine-tuning
    )
    
    # Валидация модели
    detector.validate_model()
    
    return detector

def convert_labelstudio_to_yolo(labelstudio_json, output_dir, image_dir):
    """
    Конвертация аннотаций из Label Studio в формат YOLO
    
    Args:
        labelstudio_json (str): Путь к JSON файлу из Label Studio
        output_dir (str): Папка для сохранения YOLO аннотаций
        image_dir (str): Папка с изображениями
    """
    import json
    
    with open(labelstudio_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    class_mapping = {}
    class_id = 0
    
    for item in data:
        # Получение имени файла
        image_file = os.path.basename(item['data']['image'])
        image_name = os.path.splitext(image_file)[0]
        
        # Загрузка изображения для получения размеров
        img_path = os.path.join(image_dir, image_file)
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        
        # Создание файла аннотации
        txt_path = os.path.join(output_dir, f"{image_name}.txt")
        
        with open(txt_path, 'w') as txt_file:
            for annotation in item.get('annotations', []):
                for result in annotation.get('result', []):
                    if result['type'] == 'rectanglelabels':
                        # Извлечение координат
                        x = result['value']['x'] / 100 * img_width
                        y = result['value']['y'] / 100 * img_height
                        width = result['value']['width'] / 100 * img_width
                        height = result['value']['height'] / 100 * img_height
                        
                        # Конвертация в формат YOLO (центр + размеры, нормализованные)
                        x_center = (x + width/2) / img_width
                        y_center = (y + height/2) / img_height
                        norm_width = width / img_width
                        norm_height = height / img_height
                        
                        # Получение класса
                        label = result['value']['rectanglelabels'][0]
                        if label not in class_mapping:
                            class_mapping[label] = class_id
                            class_id += 1
                        
                        # Запись в файл
                        txt_file.write(f"{class_mapping[label]} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    print(f"Конвертация завершена. Классы: {class_mapping}")
    return class_mapping

def validate_annotations(labels_dir, num_classes):
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
            
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            for line_num, line in enumerate(f.readlines()):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Ошибка в {label_file}, строка {line_num+1}: неправильное количество значений")
                    continue
                    
                class_id = int(parts[0])
                if class_id >= num_classes or class_id < 0:
                    print(f"Ошибка в {label_file}, строка {line_num+1}: class_id={class_id} вне диапазона [0, {num_classes-1}]")
                
                # Проверка координат
                for i, coord in enumerate(parts[1:]):
                    val = float(coord)
                    if val < 0.0 or val > 1.0:
                        print(f"Ошибка в {label_file}, строка {line_num+1}: координата {val} вне диапазона [0.0, 1.0]")

if __name__ == "__main__":
    # Создание примера скрипта для обучения
    detector = create_sample_training_script()
    
    print("\n=== Инструкция по использованию ===")
    print("1. Подготовьте датасет изображений ТТН")
    print("2. Аннотируйте блоки в Label Studio или другом инструменте")
    print("3. Конвертируйте аннотации в формат YOLO используя функцию convert_labelstudio_to_yolo")
    print("4. Разместите файлы в структуре train/val")
    print("5. Раскомментируйте строки обучения и запустите скрипт")
    print("6. Используйте обученную модель для детекции блоков в новых ТТН")
    
    # Пример использования для предсказания
    # results = detector.predict_ttn_blocks("path/to/ttn_image.jpg", "runs/ttn_detection/ttn_blocks_detection/weights/best.pt")
    # blocks = detector.extract_blocks_coordinates(results)
    # print("Обнаруженные блоки:", blocks)