#!/usr/bin/env python3
"""
Конвертация PaddleOCR модели в TFLite (из локальной папки models)
"""

import os
import shutil
import subprocess
from pathlib import Path

def find_local_model():
    """Поиск модели в локальной папке models"""
    print("📁 Searching for model in local directory...")
    
    # Проверяем папку models в текущей директории
    models_dir = Path('models')
    
    if not models_dir.exists():
        raise FileNotFoundError(
            "Directory 'models' not found!\n"
            "Expected structure:\n"
            "  paddle_ocr/\n"
            "  ├── models/\n"
            "  │   ├── inference.pdmodel\n"
            "  │   ├── inference.pdiparams\n"
            "  │   └── inference.pdiparams.info\n"
            "  └── main.py"
        )
    
    # Проверяем файлы модели
    pdmodel = models_dir / 'inference.pdmodel'
    pdiparams = models_dir / 'inference.pdiparams'
    
    if not pdmodel.exists():
        raise FileNotFoundError(f"Model file not found: {pdmodel}")
    if not pdiparams.exists():
        raise FileNotFoundError(f"Params file not found: {pdiparams}")
    
    print(f"✅ Found model in: {models_dir.absolute()}")
    print(f"   Files:")
    for f in models_dir.glob('*'):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"   - {f.name} ({size_mb:.2f} MB)")
    
    return str(models_dir)

def paddle_to_onnx(model_dir, output_name='paddle_rec_ru'):
    """Конвертация Paddle → ONNX"""
    print(f"\n🔄 Converting Paddle → ONNX...")
    
    cmd = [
        'paddle2onnx',
        '--model_dir', model_dir,
        '--model_filename', 'inference.pdmodel',
        '--params_filename', 'inference.pdiparams',
        '--save_file', f'{output_name}.onnx',
        '--opset_version', '11',
        '--enable_onnx_checker', 'True'
    ]
    
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"   Output: {result.stdout}")
        
        onnx_size = os.path.getsize(f'{output_name}.onnx') / 1024 / 1024
        print(f"✅ ONNX model saved: {output_name}.onnx ({onnx_size:.2f} MB)")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed!")
        print(f"   STDOUT: {e.stdout}")
        print(f"   STDERR: {e.stderr}")
        raise
    except FileNotFoundError:
        print(f"❌ paddle2onnx not found!")
        print(f"   Install it: pip install paddle2onnx")
        raise

def onnx_to_tflite(onnx_file='paddle_rec_ru.onnx', output_name='paddle_rec_ru'):
    """Конвертация ONNX → TFLite используя onnx2tf"""
    print(f"\n🔄 Converting ONNX → TFLite...")
    
    if not os.path.exists(onnx_file):
        raise FileNotFoundError(f"ONNX file not found: {onnx_file}")
    
    import onnx
    import tensorflow as tf
    import shutil
    import subprocess
    import sys
    
    # Загрузка и упрощение ONNX (опционально)
    print("   Loading ONNX model...")
    onnx_model = onnx.load(onnx_file)
    print(f"   ✅ ONNX loaded")
    
    print("   Simplifying...")
    try:
        import onnxsim
        onnx_model, check = onnxsim.simplify(onnx_model)
        if check:
            simplified_file = f'{output_name}_simplified.onnx'
            onnx.save(onnx_model, simplified_file)
            onnx_file = simplified_file
            print("   ✅ Simplified")
    except Exception as e:
        print(f"   ⚠️  Skipping simplification: {e}")
    
    # ONNX → TensorFlow используя onnx2tf
    print("   Converting to TensorFlow using onnx2tf...")
    tf_dir = f'{output_name}_saved_model'
    
    try:
        # Вызов onnx2tf через командную строку
        cmd = [
            sys.executable, '-m', 'onnx2tf',
            '-i', onnx_file,
            '-o', tf_dir,
            '-osd'  # Output SavedModel
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"   ✅ TensorFlow SavedModel created")
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ onnx2tf conversion failed: {e.stderr}")
        raise
    except Exception as e:
        print(f"   ❌ Error: {e}")
        raise
    
    # TensorFlow SavedModel → TFLite
    print("   Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
    
    # Настройки оптимизации
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.allow_custom_ops = True
    
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"   ⚠️  Standard conversion failed, trying without optimizations...")
        # Попытка без оптимизаций
        converter.optimizations = []
        tflite_model = converter.convert()
    
    # Сохранение TFLite
    tflite_file = f'{output_name}.tflite'
    with open(tflite_file, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = len(tflite_model) / 1024 / 1024
    print(f"✅ TFLite saved: {tflite_file} ({size_mb:.2f} MB)")
    
    # Очистка временных файлов
    print("   Cleaning up...")
    if os.path.exists(onnx_file) and onnx_file.endswith('_simplified.onnx'):
        os.remove(onnx_file)
    if os.path.exists(tf_dir):
        shutil.rmtree(tf_dir)
    
    return True

        
def main():
    """Основная функция"""
    print("=" * 70)
    print("PaddleOCR → TFLite Converter")
    print("=" * 70)
    print()
    
    # Шаг 1: Найти локальную модель
    print("Step 1: Find local model")
    print("-" * 70)
    model_dir = find_local_model()
    
    # # Шаг 2: Paddle → ONNX
    # print("\n" + "=" * 70)
    # print("Step 2: Convert Paddle → ONNX")
    # print("=" * 70)
    # paddle_to_onnx(model_dir, 'paddle_rec_ru')
    
    # Шаг 3: ONNX → TFLite
    print("\n" + "=" * 70)
    print("Step 3: Convert ONNX → TFLite")
    print("=" * 70)
    onnx_to_tflite('paddle_rec_ru.onnx', 'paddle_rec_ru')
    
    # Успех!
    print("\n" + "=" * 70)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 70)
    
    tflite_file = 'paddle_rec_ru.tflite'
    if os.path.exists(tflite_file):
        size_mb = os.path.getsize(tflite_file) / 1024 / 1024
        print(f"\n📦 Generated file:")
        print(f"   📄 {tflite_file} ({size_mb:.2f} MB)")
        print(f"\n📋 Integration:")
        print(f"   1. Copy to Flutter:")
        print(f"      cp {tflite_file} your_app/assets/models/")
        print(f"   2. Update pubspec.yaml:")
        print(f"      flutter:")
        print(f"        assets:")
        print(f"          - assets/models/{tflite_file}")
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
