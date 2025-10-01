#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Дообучение модели keras_ocr для распознавания текста в ключевых данных ТТН
Автор: AI Assistant
Python версия: 3.10.10
"""

import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_ocr
import cv2
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa
from pathlib import Path
import pandas as pd
import string

from keras_ocr.ttn_detector_training import TTNDataGenerator

class TTNTextRecognizer:
    """Класс для дообучения распознавателя текста на данных ТТН"""

    def __init__(self, base_model_path=None):
        self.recognizer = keras_ocr.recognition.Recognizer()

        if base_model_path and os.path.exists(base_model_path):
            self.recognizer.model.load_weights(base_model_path)
            print(f"Загружена предобученная модель: {base_model_path}")

        # Расширенный алфавит для русского и английского текста
        self.setup_alphabet()

    def setup_alphabet(self):
        """Настраивает алфавит для распознавания русского и английского текста"""
        # Базовый алфавит keras_ocr (английский + цифры)
        base_alphabet = string.ascii_lowercase + string.digits

        # Добавляем русские символы
        russian_alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'

        # Добавляем специальные символы для ТТН
        special_chars = '.,/-()№:;'

        # Объединяем все символы
        full_alphabet = base_alphabet + russian_alphabet + special_chars

        print(f"Используемый алфавит ({len(full_alphabet)} символов): {full_alphabet}")

        # Обновляем алфавит распознавателя
        self.recognizer.alphabet = full_alphabet

        return full_alphabet

    def prepare_recognition_dataset(self, annotations_file, images_dir):
        """Подготавливает данные для обучения распознавателя"""

        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        dataset = []

        for annotation in annotations:
            image_path = Path(images_dir) / annotation['image_name']

            if not image_path.exists():
                continue

            # Обрабатываем каждое поле в изображении
            for field in annotation['fields']:
                if field['field_type'] in TTNDataGenerator.get_key_ttn_fields():
                    # Координаты области текста
                    box = np.array([
                        [field['x1'], field['y1']],
                        [field['x2'], field['y1']],
                        [field['x2'], field['y2']],
                        [field['x1'], field['y2']]
                    ]).astype(np.float32)

                    # Текст в нижнем регистре для обучения
                    text = field['text'].lower().strip()

                    # Фильтруем текст по алфавиту
                    filtered_text = self.filter_text_by_alphabet(text)

                    if filtered_text and len(filtered_text) > 0:
                        dataset.append((str(image_path), box, filtered_text))

        print(f"Подготовлено {len(dataset)} образцов для обучения распознавателя")
        return dataset

    def filter_text_by_alphabet(self, text):
        """Фильтрует текст по допустимым символам алфавита"""
        filtered_chars = []
        for char in text:
            if char in self.recognizer.alphabet or char == ' ':
                filtered_chars.append(char)

        return ''.join(filtered_chars).strip()

    def setup_text_augmentation(self):
        """Настраивает аугментацию для текстовых изображений"""
        return iaa.Sequential([
            # Геометрические трансформации
            iaa.Affine(
                scale=(0.95, 1.05),
                rotate=(-2, 2),
                shear=(-3, 3)
            ),
            # Изменение яркости и контрастности
            iaa.Multiply((0.9, 1.1)),
            iaa.Add((-10, 10)),
            # Размытие
            iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 0.5))),
            # Шум
            iaa.Sometimes(0.2, iaa.AdditiveGaussianNoise(scale=0.01*255))
        ])

    def create_recognition_generators(self, dataset, batch_size=8, validation_split=0.2):
        """Создает генераторы данных для обучения распознавателя"""

        # Разделяем данные
        train_data, val_data = train_test_split(
            dataset, 
            test_size=validation_split, 
            random_state=42
        )

        print(f"Размер обучающей выборки для распознавателя: {len(train_data)}")
        print(f"Размер валидационной выборки для распознавателя: {len(val_data)}")

        # Создаем генераторы изображений
        train_image_gen, train_steps = keras_ocr.datasets.get_recognizer_image_generator(
            labels=train_data,
            height=self.recognizer.model.input_shape[1],
            width=self.recognizer.model.input_shape[2],
            alphabet=self.recognizer.alphabet,
            augmenter=self.setup_text_augmentation()
        ), len(train_data) // batch_size

        val_image_gen, val_steps = keras_ocr.datasets.get_recognizer_image_generator(
            labels=val_data,
            height=self.recognizer.model.input_shape[1],
            width=self.recognizer.model.input_shape[2],
            alphabet=self.recognizer.alphabet,
            augmenter=None  # Без аугментации для валидации
        ), len(val_data) // batch_size

        # Создаем batch генераторы
        train_gen = self.recognizer.get_batch_generator(
            image_generator=train_image_gen,
            batch_size=batch_size
        )

        val_gen = self.recognizer.get_batch_generator(
            image_generator=val_image_gen,
            batch_size=batch_size
        )

        return train_gen, val_gen, train_steps, val_steps

    def train(self, dataset, epochs=100, batch_size=8, save_path='ttn_recognizer.h5'):
        """Обучает распознаватель текста"""

        # Компилируем модель
        self.recognizer.compile()

        # Создаем генераторы
        train_gen, val_gen, train_steps, val_steps = self.create_recognition_generators(
            dataset, batch_size
        )

        # Настраиваем callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                'ttn_recognizer_training.log'
            )
        ]

        print(f"Начинаем обучение распознавателя ТТН...")
        print(f"Шагов за эпоху: {train_steps}")
        print(f"Валидационных шагов: {val_steps}")

        # Запускаем обучение
        history = self.recognizer.training_model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def evaluate_model(self, test_dataset):
        """Оценивает качество модели на тестовых данных"""

        correct_predictions = 0
        total_predictions = 0

        for image_path, box, true_text in test_dataset[:100]:  # Тестируем на 100 образцах
            try:
                # Загружаем и обрезаем изображение
                image = keras_ocr.tools.read(image_path)
                cropped_image = keras_ocr.tools.warp_box(image, box)

                # Распознаем текст
                predicted_text = self.recognizer.recognize(cropped_image)

                # Сравниваем с истинным текстом
                if predicted_text.lower().strip() == true_text.lower().strip():
                    correct_predictions += 1

                total_predictions += 1

                # Выводим примеры
                if total_predictions <= 10:
                    print(f"Истинный текст: '{true_text}'")
                    print(f"Предсказанный текст: '{predicted_text}'")
                    print("---")

            except Exception as e:
                print(f"Ошибка при обработке {image_path}: {e}")
                continue

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Точность на тестовых данных: {accuracy:.3f} ({correct_predictions}/{total_predictions})")

        return accuracy

def create_sample_annotations():
    """Создает пример файла аннотаций для ТТН"""

    sample_annotations = [
        {
            "image_name": "ttn_001.jpg",
            "fields": [
                {
                    "field_type": "номер_накладной",
                    "text": "№ 012345",
                    "x1": 150, "y1": 50, "x2": 250, "y2": 80
                },
                {
                    "field_type": "дата_составления", 
                    "text": "15.10.2024",
                    "x1": 300, "y1": 50, "x2": 400, "y2": 80
                },
                {
                    "field_type": "грузоотправитель",
                    "text": "ООО Ромашка",
                    "x1": 100, "y1": 120, "x2": 300, "y2": 150
                }
            ]
        }
    ]

    with open('sample_ttn_annotations.json', 'w', encoding='utf-8') as f:
        json.dump(sample_annotations, f, ensure_ascii=False, indent=2)

    print("Создан пример файла аннотаций: sample_ttn_annotations.json")

# Основная функция для запуска обучения распознавателя
def main():
    """Основная функция для запуска обучения распознавателя"""

    # Пути к данным
    images_dir = "dataset/images"
    annotations_file = "data/converted_annotations.json"

    try:
        # Создаем пример аннотаций если файл не существует
        if not os.path.exists(annotations_file):
            create_sample_annotations()
            print("Используйте созданный пример аннотаций как шаблон")
            return

        # Создаем и обучаем распознаватель
        recognizer = TTNTextRecognizer()

        # Подготавливаем данные
        dataset = recognizer.prepare_recognition_dataset(annotations_file, images_dir)

        if not dataset:
            print("Ошибка: Датасет пуст! Проверьте пути к данным и аннотации.")
            return

        # Запускаем обучение
        history = recognizer.train(
            dataset=dataset,
            epochs=100,
            batch_size=8,
            save_path='ttn_recognizer_best.h5'
        )

        # Оцениваем модель
        test_data = dataset[-50:]  # Последние 50 образцов для тестирования
        recognizer.evaluate_model(test_data)

        print("Обучение распознавателя завершено!")

    except Exception as e:
        print(f"Ошибка во время обучения: {e}")

if __name__ == "__main__":
    main()
