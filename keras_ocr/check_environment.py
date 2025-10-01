#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для проверки и исправления окружения для keras-ocr
Автор: AI Assistant
"""

import sys
import subprocess
import pkg_resources
from packaging import version

def check_python_version():
    """Проверяет версию Python"""
    print(f"Python версия: {sys.version}")

    if sys.version_info[:2] == (3, 10):
        print("✓ Python 3.10 - совместим")
        return True
    else:
        print("⚠ Рекомендуется Python 3.10 для максимальной совместимости")
        return False

def get_installed_version(package_name):
    """Получает версию установленного пакета"""
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def check_package_compatibility():
    """Проверяет совместимость ключевых пакетов"""

    packages_to_check = {
        'tensorflow': ('2.12.1', '2.12.1'),
        'tensorflow-intel': ('2.12.1', '2.12.1'),
        'keras': ('2.12.0', '2.13.0'),
        'protobuf': ('3.20.3', '5.0.0'),
        'numpy': ('1.21.0', '1.25.0'),
        'opencv-python': ('4.5.0', None),
        'matplotlib': ('3.5.0', None),
        'pandas': ('1.3.0', None),
        'scikit-learn': ('1.0.0', None),
    }

    print("\n=== ПРОВЕРКА СОВМЕСТИМОСТИ ПАКЕТОВ ===")

    issues = []

    for package, (min_ver, max_ver) in packages_to_check.items():
        installed_ver = get_installed_version(package)

        if installed_ver is None:
            print(f"❌ {package}: НЕ УСТАНОВЛЕН")
            issues.append(f"Установите {package}>={min_ver}")
            continue

        # Проверяем минимальную версию
        if version.parse(installed_ver) < version.parse(min_ver):
            print(f"❌ {package}: {installed_ver} (нужно >={min_ver})")
            issues.append(f"Обновите {package} до версии >={min_ver}")
            continue

        # Проверяем максимальную версию
        if max_ver and version.parse(installed_ver) >= version.parse(max_ver):
            print(f"⚠ {package}: {installed_ver} (рекомендуется <{max_ver})")
            issues.append(f"Понизьте {package} до версии <{max_ver}")
            continue

        print(f"✓ {package}: {installed_ver}")

    return issues

def check_keras_ocr():
    """Проверяет доступность keras-ocr"""
    print("\n=== ПРОВЕРКА KERAS-OCR ===")

    try:
        import keras_ocr
        print("✓ keras-ocr импортируется успешно")

        # Проверяем основные компоненты
        try:
            detector = keras_ocr.detection.Detector()
            print("✓ Детектор создается успешно")
        except Exception as e:
            print(f"❌ Ошибка создания детектора: {e}")

        try:
            recognizer = keras_ocr.recognition.Recognizer()
            print("✓ Распознаватель создается успешно")
        except Exception as e:
            print(f"❌ Ошибка создания распознавателя: {e}")

        return True

    except ImportError as e:
        print(f"❌ keras-ocr не импортируется: {e}")
        print("\nВозможные решения:")
        print("1. pip install keras-ocr")
        print("2. pip install image-ocr")
        print("3. pip install git+https://github.com/faustomorales/keras-ocr.git")
        return False

def suggest_fixes(issues):
    """Предлагает команды для исправления проблем"""
    if not issues:
        print("\n✓ Все пакеты совместимы!")
        return

    print("\n=== РЕКОМЕНДУЕМЫЕ ИСПРАВЛЕНИЯ ===")

    # Группируем команды
    uninstall_commands = []
    install_commands = []

    for issue in issues:
        print(f"• {issue}")

    # Предлагаем команды для исправления
    print("\n=== КОМАНДЫ ДЛЯ ИСПРАВЛЕНИЯ ===")

    print("# 1. Удалите конфликтующие пакеты:")
    print("pip uninstall tensorflow tensorflow-intel keras tensorboard tensorflow-estimator protobuf -y")

    print("\n# 2. Установите совместимые версии:")
    print("pip install tensorflow==2.12.1")
    print("pip install keras>=2.12.0,<2.13.0") 
    print("pip install tensorboard>=2.12.0,<2.13.0")
    print("pip install tensorflow-estimator>=2.12.0,<2.13.0")
    print('pip install "protobuf>=3.20.3,<5.0.0"')

    print("\n# 3. Установите остальные зависимости:")
    print("pip install -r requirements_fixed.txt")

def main():
    """Основная функция"""
    print("=== ПРОВЕРКА ОКРУЖЕНИЯ ДЛЯ KERAS-OCR ===\n")

    # Проверяем Python
    check_python_version()

    # Проверяем пакеты
    issues = check_package_compatibility()

    # Проверяем keras-ocr
    keras_ocr_ok = check_keras_ocr()

    # Предлагаем исправления
    if issues or not keras_ocr_ok:
        suggest_fixes(issues)
    else:
        print("\n🎉 Окружение настроено правильно!")
        print("Можете запускать обучение моделей ТТН")

if __name__ == "__main__":
    main()
