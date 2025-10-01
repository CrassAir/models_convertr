#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Дообучение модели keras_ocr для определения ключевых данных на ТТН
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

# Проверяем доступность GPU
print("GPU доступно:", tf.config.list_physical_devices('GPU'))


class TTNDataGenerator:
    """Генератор данных для обучения детектора текста на ТТН"""

    def __init__(self, data_dir, annotations_file):
        self.data_dir = Path(data_dir)
        self.annotations = self.load_annotations(annotations_file)

    def load_annotations(self, annotations_file):
        """Загружает аннотации из JSON файла"""
        with open(annotations_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def prepare_dataset(self):
        """Подготавливает датасет в формате для keras_ocr"""
        dataset = []
        for annotation in self.annotations:
            image_path = self.data_dir / annotation['image_name']
            if not image_path.exists():
                continue
            lines = []
            for field in annotation['fields']:
                box = np.array([
                    [field['x1'], field['y1']],
                    [field['x2'], field['y1']],
                    [field['x2'], field['y2']],
                    [field['x1'], field['y2']],
                ], dtype=np.float32)
                text = field['text']
                line = [(box, char) for char in text]
                lines.append(line)
            if lines:
                dataset.append((str(image_path), lines, 1.0))
        return dataset

    @staticmethod
    def get_key_ttn_fields():
        """Возвращает список ключевых полей ТТН для обучения"""
        return [
            'номер_накладной',
            'дата_составления',
            'грузоотправитель',
            'грузополучатель',
            'наименование_груза',
            'количество_мест',
            'масса_груза',
            'стоимость_груза',
            'водитель',
            'автомобиль_номер',
            'путь_следования'
        ]

class TTNDetectorTrainer:
    """Класс для дообучения детектора текста на данных ТТН"""

    def __init__(self, base_model_path=None):
        self.detector = keras_ocr.detection.Detector()
        if base_model_path and os.path.exists(base_model_path):
            self.detector.model.load_weights(base_model_path)
            print(f"Загружена предобученная модель: {base_model_path}")

    def setup_data_augmentation(self):
        """Настраивает аугментацию данных"""
        return iaa.Sequential([
            iaa.Affine(
                scale=(0.9, 1.1),
                rotate=(-3, 3),
                shear=(-5, 5)
            ),
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.Multiply((0.8, 1.2)),
            iaa.Add((-20, 20)),
            iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.01, 0.05)))
        ])

    def create_data_generators(self, dataset, batch_size=2, validation_split=0.2):
        """Создает генераторы данных для обучения и валидации"""

        # Разделяем данные на обучающую и валидационную выборки
        train_data, val_data = train_test_split(
            dataset, 
            test_size=validation_split, 
            random_state=42
        )

        print(f"Размер обучающей выборки: {len(train_data)}")
        print(f"Размер валидационной выборки: {len(val_data)}")


        # Создаем генераторы
        train_generator = keras_ocr.datasets.get_detector_image_generator(
            labels=train_data,
            augmenter=self.setup_data_augmentation(),
            width=640,
            height=640
        )

        val_generator = keras_ocr.datasets.get_detector_image_generator(
            labels=val_data,
            augmenter=None,  # Без аугментации для валидации
            width=640,
            height=640
        )

        return train_generator, val_generator, len(train_data), len(val_data)

    def train(self, dataset, epochs=50, batch_size=2, save_path='ttn_detector.h5'):
        """Обучает детектор на данных ТТН"""

        # Создаем генераторы данных
        train_gen, val_gen, train_size, val_size = self.create_data_generators(
            dataset, batch_size
        )

        # Создаем batch генераторы
        train_batch_gen = self.detector.get_batch_generator(
            image_generator=train_gen,
            batch_size=batch_size
        )

        train_batch = next(iter(train_batch_gen))
        print("Train batch X shape:", train_batch[0].shape)
        print("Train batch y shape:", train_batch[1].shape)
        
        val_batch_gen = self.detector.get_batch_generator(
            image_generator=val_gen,
            batch_size=batch_size
        )
        
        # Настраиваем callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
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
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                'ttn_detector_training.log'
            )
        ]

        # Вычисляем количество шагов
        steps_per_epoch = max(1, train_size // batch_size)
        validation_steps = max(1, val_size // batch_size)

        print(f"Начинаем обучение детектора ТТН...")
        print(f"Шагов за эпоху: {steps_per_epoch}")
        print(f"Валидационных шагов: {validation_steps}")

        self.detector.model.compile(
            optimizer='adam',
            loss='mse',  # Простая среднеквадратичная ошибка
            run_eagerly=True  # Отключаем оптимизации графа
        )

        # Запускаем обучение
        history = self.detector.model.fit(
            train_batch_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_batch_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1,
            sample_weight=None,  
            class_weight=None
        )

        return history

    def visualize_training(self, history):
        """Визуализирует процесс обучения"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        if 'accuracy' in history.history:
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

        plt.tight_layout()
        plt.savefig('ttn_detector_training_plots.png')
        plt.show()

# Пример использования
def main():
    """Основная функция для запуска обучения"""

    # Пути к данным
    data_dir = "dataset"
    annotations_file = "data/converted_annotations.json"

    # Создаем генератор данных
    data_generator = TTNDataGenerator(data_dir, annotations_file)
    dataset = data_generator.prepare_dataset()

    if not dataset:
        print("Ошибка: Датасет пуст! Проверьте пути к данным и аннотации.")
        return

    # Создаем и обучаем детектор
    trainer = TTNDetectorTrainer()

    # Запускаем обучение
    history = trainer.train(
        dataset=dataset,
        epochs=50,
        batch_size=2,  # Маленький batch_size для экономии памяти
        save_path='ttn_detector_best.h5'
    )

    # Визуализируем результаты
    trainer.visualize_training(history)

    print("Обучение детектора завершено!")

   
if __name__ == "__main__":
    main()
