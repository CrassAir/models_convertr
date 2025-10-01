#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Вспомогательный скрипт для подготовки данных для обучения моделей ТТН
Автор: AI Assistant
Python версия: 3.10.10
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
import argparse

class TTNDataPreparator:
    """Класс для подготовки и валидации данных для обучения"""

    def __init__(self, images_dir: str, annotations_file: str):
        self.images_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)
        self.annotations = []

        if self.annotations_file.exists():
            self.load_annotations()

    def load_annotations(self):
        """Загружает аннотации из JSON файла"""
        try:
            with open(self.annotations_file, 'r', encoding='utf-8') as f:
                self.annotations = json.load(f)
            print(f"Загружено {len(self.annotations)} аннотаций")
        except Exception as e:
            print(f"Ошибка при загрузке аннотаций: {e}")
            self.annotations = []

    def create_sample_annotation(self, image_name: str) -> Dict:
        """Создает шаблон аннотации для изображения"""
        return {
            "image_name": image_name,
            "fields": [
                {
                    "field_type": "номер_накладной",
                    "text": "№ 000000",
                    "x1": 0, "y1": 0, "x2": 100, "y2": 30,
                    "confidence": 1.0
                },
                {
                    "field_type": "дата_составления",
                    "text": "01.01.2024",
                    "x1": 0, "y1": 0, "x2": 100, "y2": 30,
                    "confidence": 1.0
                }
            ]
        }

    def validate_annotations(self) -> List[str]:
        """Проверяет корректность аннотаций"""
        errors = []
        required_fields = ['image_name', 'fields']
        field_required = ['field_type', 'text', 'x1', 'y1', 'x2', 'y2']

        for i, annotation in enumerate(self.annotations):
            # Проверяем основные поля
            for field in required_fields:
                if field not in annotation:
                    errors.append(f"Аннотация {i}: отсутствует поле '{field}'")

            # Проверяем существование изображения
            image_path = self.images_dir / annotation.get('image_name', '')
            if not image_path.exists():
                errors.append(f"Аннотация {i}: изображение не найдено: {image_path}")

            # Проверяем поля в fields
            for j, field in enumerate(annotation.get('fields', [])):
                for req_field in field_required:
                    if req_field not in field:
                        errors.append(f"Аннотация {i}, поле {j}: отсутствует '{req_field}'")

                # Проверяем координаты
                try:
                    x1, y1, x2, y2 = field['x1'], field['y1'], field['x2'], field['y2']
                    if x1 >= x2 or y1 >= y2:
                        errors.append(f"Аннотация {i}, поле {j}: некорректные координаты")
                except (KeyError, TypeError):
                    errors.append(f"Аннотация {i}, поле {j}: некорректные координаты")

        return errors

    def visualize_annotation(self, annotation_index: int, save_path: str = None):
        """Визуализирует аннотацию на изображении"""
        if annotation_index >= len(self.annotations):
            print(f"Индекс {annotation_index} превышает количество аннотаций")
            return

        annotation = self.annotations[annotation_index]
        image_path = self.images_dir / annotation['image_name']

        if not image_path.exists():
            print(f"Изображение не найдено: {image_path}")
            return

        # Загружаем изображение
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Создаем фигуру
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(image)

        # Цвета for разных типов полей
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        field_types = list(set(field['field_type'] for field in annotation['fields']))
        color_map = {field_type: colors[i % len(colors)] for i, field_type in enumerate(field_types)}

        # Рисуем аннотации
        for field in annotation['fields']:
            x1, y1, x2, y2 = field['x1'], field['y1'], field['x2'], field['y2']
            field_type = field['field_type']
            text = field['text']

            # Создаем прямоугольник
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, 
                edgecolor=color_map[field_type],
                facecolor='none'
            )
            ax.add_patch(rect)

            # Добавляем подпись
            ax.text(x1, y1-5, f"{field_type}: {text}", 
                   fontsize=8, color=color_map[field_type],
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        ax.set_title(f"Аннотация для {annotation['image_name']}")
        ax.axis('off')

        # Легенда
        legend_elements = [patches.Patch(color=color_map[ft], label=ft) for ft in field_types]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Визуализация сохранена: {save_path}")

        plt.show()

    def generate_statistics(self) -> Dict:
        """Генерирует статистику по датасету"""
        stats = {
            'total_images': len(self.annotations),
            'total_fields': 0,
            'field_types': {},
            'avg_fields_per_image': 0,
            'image_sizes': [],
            'text_lengths': []
        }

        for annotation in self.annotations:
            fields = annotation.get('fields', [])
            stats['total_fields'] += len(fields)

            for field in fields:
                field_type = field['field_type']
                stats['field_types'][field_type] = stats['field_types'].get(field_type, 0) + 1
                stats['text_lengths'].append(len(field['text']))

            # Получаем размер изображения
            image_path = self.images_dir / annotation['image_name']
            if image_path.exists():
                image = cv2.imread(str(image_path))
                if image is not None:
                    h, w = image.shape[:2]
                    stats['image_sizes'].append((w, h))

        if stats['total_images'] > 0:
            stats['avg_fields_per_image'] = stats['total_fields'] / stats['total_images']

        return stats

    def print_statistics(self):
        """Выводит статистику датасета"""
        stats = self.generate_statistics()

        print("=== СТАТИСТИКА ДАТАСЕТА ===")
        print(f"Всего изображений: {stats['total_images']}")
        print(f"Всего полей: {stats['total_fields']}")
        print(f"Среднее количество полей на изображение: {stats['avg_fields_per_image']:.2f}")

        print("\n=== ТИПЫ ПОЛЕЙ ===")
        for field_type, count in sorted(stats['field_types'].items()):
            print(f"{field_type}: {count}")

        if stats['image_sizes']:
            widths, heights = zip(*stats['image_sizes'])
            print(f"\n=== РАЗМЕРЫ ИЗОБРАЖЕНИЙ ===")
            print(f"Ширина: мин={min(widths)}, макс={max(widths)}, среднее={np.mean(widths):.0f}")
            print(f"Высота: мин={min(heights)}, макс={max(heights)}, среднее={np.mean(heights):.0f}")

        if stats['text_lengths']:
            print(f"\n=== ДЛИНА ТЕКСТА ===")
            print(f"Символов: мин={min(stats['text_lengths'])}, макс={max(stats['text_lengths'])}, среднее={np.mean(stats['text_lengths']):.1f}")

    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """Разделяет датасет на train/val/test"""

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Сумма долей должна равняться 1.0")

        total = len(self.annotations)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        # Перемешиваем данные
        indices = np.random.permutation(total)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        splits = {
            'train': [self.annotations[i] for i in train_indices],
            'val': [self.annotations[i] for i in val_indices],
            'test': [self.annotations[i] for i in test_indices]
        }

        # Сохраняем разделения
        for split_name, split_data in splits.items():
            output_file = f"{split_name}_annotations.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            print(f"Сохранено {len(split_data)} образцов в {output_file}")

        return splits

    def create_empty_annotations_template(self):
        """Создает шаблон файла аннотаций"""
        template = []

        # Ищем все изображения в папке
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        for ext in image_extensions:
            for image_path in self.images_dir.glob(f"*{ext}"):
                template.append(self.create_sample_annotation(image_path.name))
            for image_path in self.images_dir.glob(f"*{ext.upper()}"):
                template.append(self.create_sample_annotation(image_path.name))

        # Сохраняем шаблон
        template_file = "annotations_template.json"
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)

        print(f"Создан шаблон аннотаций: {template_file}")
        print(f"Найдено {len(template)} изображений")
        print("Отредактируйте координаты и тексты полей в созданном файле")

def main():
    """Основная функция для работы с подготовкой данных"""

    parser = argparse.ArgumentParser(description='Подготовка данных для обучения OCR на ТТН')
    parser.add_argument('--images_dir', default='data/images', help='Папка с изображениями')
    parser.add_argument('--annotations', default='data/annotations.json', help='Файл аннотаций')
    parser.add_argument('--action', choices=['validate', 'stats', 'visualize', 'split', 'template'], 
                       default='stats', help='Действие для выполнения')
    parser.add_argument('--index', type=int, default=0, help='Индекс аннотации для визуализации')

    args = parser.parse_args()

    # Создаем препаратор данных
    preparator = TTNDataPreparator(args.images_dir, args.annotations)

    if args.action == 'validate':
        print("=== ВАЛИДАЦИЯ АННОТАЦИЙ ===")
        errors = preparator.validate_annotations()
        if errors:
            print("Найдены ошибки:")
            for error in errors:
                print(f"- {error}")
        else:
            print("Все аннотации корректны!")

    elif args.action == 'stats':
        preparator.print_statistics()

    elif args.action == 'visualize':
        preparator.visualize_annotation(args.index, f"annotation_{args.index}_visualization.png")

    elif args.action == 'split':
        preparator.split_dataset()

    elif args.action == 'template':
        preparator.create_empty_annotations_template()

if __name__ == "__main__":
    main()
