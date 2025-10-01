#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пайплайн для использования дообученных моделей keras_ocr для работы с ТТН
Автор: AI Assistant
Python версия: 3.10.10
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_ocr
import cv2
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTNProcessor:
    """Класс для обработки ТТН с использованием дообученных моделей"""

    def __init__(self, detector_path: str, recognizer_path: str):
        """
        Инициализация пайплайна с дообученными моделями

        Args:
            detector_path: Путь к весам дообученного детектора
            recognizer_path: Путь к весам дообученного распознавателя
        """
        self.detector_path = detector_path
        self.recognizer_path = recognizer_path

        # Инициализируем модели
        self.setup_models()

        # Словарь ключевых полей ТТН
        self.key_fields = self.get_ttn_field_mapping()

    def setup_models(self):
        """Настраивает и загружает дообученные модели"""

        try:
            # Создаем детектор и загружаем веса
            self.detector = keras_ocr.detection.Detector()
            if os.path.exists(self.detector_path):
                self.detector.model.load_weights(self.detector_path)
                logger.info(f"Загружена дообученная модель детектора: {self.detector_path}")
            else:
                logger.warning(f"Файл детектора не найден: {self.detector_path}. Используется базовая модель.")

            # Создаем распознаватель и загружаем веса
            self.recognizer = keras_ocr.recognition.Recognizer()
            if os.path.exists(self.recognizer_path):
                self.recognizer.model.load_weights(self.recognizer_path)
                logger.info(f"Загружена дообученная модель распознавателя: {self.recognizer_path}")
            else:
                logger.warning(f"Файл распознавателя не найден: {self.recognizer_path}. Используется базовая модель.")

            # Создаем пайплайн
            self.pipeline = keras_ocr.pipeline.Pipeline(
                detector=self.detector,
                recognizer=self.recognizer
            )

        except Exception as e:
            logger.error(f"Ошибка при инициализации моделей: {e}")
            raise

    def get_ttn_field_mapping(self) -> Dict[str, str]:
        """Возвращает маппинг ключевых полей ТТН"""
        return {
            'номер_накладной': 'Номер накладной',
            'дата_составления': 'Дата составления',
            'грузоотправитель': 'Грузоотправитель',
            'грузополучатель': 'Грузополучатель',
            'наименование_груза': 'Наименование груза',
            'количество_мест': 'Количество мест',
            'масса_груза': 'Масса груза (кг)',
            'стоимость_груза': 'Стоимость груза (руб.)',
            'водитель': 'ФИО водителя',
            'автомобиль_номер': 'Номер автомобиля',
            'путь_следования': 'Путь следования',
            'подпись_отправителя': 'Подпись отправителя',
            'подпись_получателя': 'Подпись получателя'
        }

    def process_ttn_image(self, image_path: str) -> Dict[str, any]:
        """
        Обрабатывает изображение ТТН и извлекает ключевые данные

        Args:
            image_path: Путь к изображению ТТН

        Returns:
            Словарь с извлеченными данными
        """

        try:
            # Загружаем изображение
            image = keras_ocr.tools.read(image_path)
            logger.info(f"Обрабатываем изображение: {image_path}")

            # Применяем пайплайн для распознавания
            predictions = self.pipeline.recognize([image])[0]

            # Классифицируем найденный текст по полям ТТН
            classified_data = self.classify_predictions(predictions, image)

            # Структурируем результат
            result = {
                'image_path': image_path,
                'status': 'success',
                'extracted_fields': classified_data,
                'total_fields_found': len(classified_data),
                'confidence_score': self.calculate_confidence(classified_data)
            }

            return result

        except Exception as e:
            logger.error(f"Ошибка при обработке изображения {image_path}: {e}")
            return {
                'image_path': image_path,
                'status': 'error',
                'error_message': str(e),
                'extracted_fields': {},
                'total_fields_found': 0,
                'confidence_score': 0.0
            }

    def classify_predictions(self, predictions: List[Tuple], image: np.ndarray) -> Dict[str, Dict]:
        """
        Классифицирует распознанный текст по полям ТТН

        Args:
            predictions: Список предсказаний от keras_ocr
            image: Исходное изображение

        Returns:
            Классифицированные данные по полям
        """

        classified_data = {}

        for text, box in predictions:
            # Определяем тип поля на основе содержимого и позиции
            field_type = self.identify_field_type(text, box, image.shape)

            if field_type:
                # Сохраняем информацию о поле
                classified_data[field_type] = {
                    'text': text,
                    'coordinates': box.tolist(),
                    'confidence': self.estimate_text_confidence(text),
                    'field_name': self.key_fields.get(field_type, field_type)
                }

        return classified_data

    def identify_field_type(self, text: str, box: np.ndarray, image_shape: Tuple) -> Optional[str]:
        """
        Определяет тип поля ТТН на основе содержимого текста и его позиции

        Args:
            text: Распознанный текст
            box: Координаты текста
            image_shape: Размеры изображения

        Returns:
            Тип поля или None
        """

        text_lower = text.lower().strip()

        # Определяем относительную позицию в изображении
        center_x = np.mean(box[:, 0]) / image_shape[1]  # Нормализованная X координата
        center_y = np.mean(box[:, 1]) / image_shape[0]  # Нормализованная Y координата

        # Правила классификации по содержимому
        if any(pattern in text_lower for pattern in ['№', 'номер', 'n°']):
            if any(word in text_lower for word in ['накладн', 'тт']):
                return 'номер_накладной'

        # Дата (различные форматы)
        if self.is_date_format(text):
            if center_y < 0.3:  # Дата обычно в верхней части
                return 'дата_составления'

        # ООО, ИП, организации
        if any(pattern in text_lower for pattern in ['ооо', 'ип', 'зао', 'оао']):
            if center_y < 0.5:  # Отправитель обычно в верхней части
                return 'грузоотправитель'
            else:
                return 'грузополучатель'

        # Наименование груза (обычно содержит описательные слова)
        if any(word in text_lower for word in ['товар', 'груз', 'продук', 'материал']):
            return 'наименование_груза'

        # Количественные данные
        if any(pattern in text_lower for pattern in ['кг', 'т', 'шт', 'л', 'м']):
            if any(word in text_lower for word in ['масс', 'вес']):
                return 'масса_груза'
            elif any(word in text_lower for word in ['кол', 'шт']):
                return 'количество_мест'

        # Стоимость
        if any(pattern in text_lower for pattern in ['руб', 'коп', 'рубл']):
            return 'стоимость_груза'

        # ФИО (паттерн из 2-3 слов с заглавными буквами)
        if self.is_likely_name(text):
            if center_y > 0.7:  # ФИО обычно в нижней части
                return 'водитель'

        # Номер автомобиля
        if self.is_car_number(text):
            return 'автомобиль_номер'

        return None

    def is_date_format(self, text: str) -> bool:
        """Проверяет, является ли текст датой"""
        import re

        date_patterns = [
            r'\d{1,2}\.\d{1,2}\.\d{4}',  # дд.мм.гггг
            r'\d{1,2}/\d{1,2}/\d{4}',   # дд/мм/гггг
            r'\d{1,2}-\d{1,2}-\d{4}',   # дд-мм-гггг
        ]

        for pattern in date_patterns:
            if re.search(pattern, text):
                return True
        return False

    def is_likely_name(self, text: str) -> bool:
        """Проверяет, похож ли текст на ФИО"""
        words = text.split()

        # ФИО обычно состоит из 2-3 слов
        if len(words) < 2 or len(words) > 4:
            return False

        # Каждое слово должно начинаться с заглавной буквы
        for word in words:
            if not word or not word[0].isupper():
                return False
            # И содержать только буквы
            if not word.replace('-', '').isalpha():
                return False

        return True

    def is_car_number(self, text: str) -> bool:
        """Проверяет, является ли текст номером автомобиля"""
        import re

        # Российские номера автомобилей
        car_patterns = [
            r'[А-Я]\d{3}[А-Я]{2}\d{2,3}',  # А123БВ77
            r'\d{4}[А-Я]{2}\d{2}',          # 1234АБ77
        ]

        text_upper = text.upper().replace(' ', '')

        for pattern in car_patterns:
            if re.search(pattern, text_upper):
                return True

        return False

    def estimate_text_confidence(self, text: str) -> float:
        """Оценивает уверенность в распознанном тексте"""

        # Простая эвристика для оценки качества распознавания
        confidence = 1.0

        # Снижаем уверенность за специальные символы
        special_chars = sum(1 for c in text if c in '!@#$%^&*')
        confidence -= special_chars * 0.1

        # Снижаем уверенность за очень короткий или длинный текст
        if len(text) < 2:
            confidence -= 0.3
        elif len(text) > 50:
            confidence -= 0.2

        return max(0.0, min(1.0, confidence))

    def calculate_confidence(self, classified_data: Dict) -> float:
        """Вычисляет общую уверенность для всех найденных полей"""

        if not classified_data:
            return 0.0

        total_confidence = sum(field['confidence'] for field in classified_data.values())
        return total_confidence / len(classified_data)

    def save_results_to_json(self, results: Dict, output_path: str):
        """Сохраняет результаты в JSON файл"""

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.info(f"Результаты сохранены в {output_path}")

        except Exception as e:
            logger.error(f"Ошибка при сохранении результатов: {e}")

    def create_results_dataframe(self, results: Dict) -> pd.DataFrame:
        """Создает DataFrame с результатами для анализа"""

        rows = []
        for field_type, field_data in results.get('extracted_fields', {}).items():
            rows.append({
                'Тип поля': field_data.get('field_name', field_type),
                'Текст': field_data.get('text', ''),
                'Уверенность': field_data.get('confidence', 0.0),
                'Координаты': str(field_data.get('coordinates', []))
            })

        return pd.DataFrame(rows)

    def visualize_results(self, image_path: str, results: Dict, save_path: Optional[str] = None):
        """Визуализирует результаты распознавания на изображении"""

        try:
            # Загружаем изображение
            image = keras_ocr.tools.read(image_path)

            # Создаем копию для рисования
            vis_image = image.copy()

            # Рисуем найденные поля
            for field_type, field_data in results.get('extracted_fields', {}).items():
                coordinates = np.array(field_data['coordinates'])
                text = field_data['text']
                confidence = field_data['confidence']

                # Рисуем прямоугольник
                cv2.polylines(vis_image, [coordinates.astype(int)], True, (0, 255, 0), 2)

                # Добавляем текст
                x, y = int(coordinates[0][0]), int(coordinates[0][1] - 5)
                label = f"{field_type}: {confidence:.2f}"
                cv2.putText(vis_image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Показываем результат
            plt.figure(figsize=(15, 10))
            plt.imshow(vis_image)
            plt.title(f'Результаты распознавания ТТН\nНайдено полей: {len(results.get("extracted_fields", {}))}')
            plt.axis('off')

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Визуализация сохранена в {save_path}")

            plt.show()

        except Exception as e:
            logger.error(f"Ошибка при визуализации: {e}")

def main():
    """Пример использования пайплайна для обработки ТТН"""

    # Пути к дообученным моделям
    detector_path = "ttn_detector_best.h5"
    recognizer_path = "ttn_recognizer_best.h5"

    # Создаем процессор
    processor = TTNProcessor(detector_path, recognizer_path)

    # Пример обработки изображения
    test_image_path = "path/to/test/ttn/image.jpg"

    if os.path.exists(test_image_path):
        # Обрабатываем изображение
        results = processor.process_ttn_image(test_image_path)

        # Выводим результаты
        print("=== РЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ ТТН ===")
        print(f"Статус: {results['status']}")
        print(f"Найдено полей: {results['total_fields_found']}")
        print(f"Общая уверенность: {results['confidence_score']:.3f}")

        if results['status'] == 'success':
            print("\n=== ИЗВЛЕЧЕННЫЕ ДАННЫЕ ===")
            df = processor.create_results_dataframe(results)
            print(df.to_string(index=False))

            # Сохраняем результаты
            processor.save_results_to_json(results, "ttn_results.json")

            # Визуализируем
            processor.visualize_results(test_image_path, results, "ttn_visualization.png")

        else:
            print(f"Ошибка: {results.get('error_message', 'Неизвестная ошибка')}")

    else:
        print(f"Тестовое изображение не найдено: {test_image_path}")
        print("Создайте тестовое изображение ТТН для проверки работы пайплайна")

if __name__ == "__main__":
    main()
