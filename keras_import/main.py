import tensorflow as tf
import keras_ocr
import os

def save_and_convert_detector(model, input_shape, saved_dir, tflite_file):
    if os.path.exists(saved_dir):
        tf.io.gfile.rmtree(saved_dir)
    
    inp = tf.keras.Input(shape=input_shape, batch_size=1, name='input_image')
    out = model(inp)
    func_model = tf.keras.Model(inputs=inp, outputs=out)
    func_model.save(saved_dir, include_optimizer=False, save_format='tf')
    
    converter = tf.lite.TFLiteConverter.from_saved_model(
        saved_dir, signature_keys=['serving_default']
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    
    with open(tflite_file, 'wb') as f:
        f.write(tflite_model)
    print(f'✅ Created {tflite_file}')

def save_and_convert_recognizer(model, input_shape, saved_dir, tflite_file):
    if os.path.exists(saved_dir):
        tf.io.gfile.rmtree(saved_dir)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(1,) + input_shape, dtype=tf.float32)])
    def inference_fn(x):
        return model(x, training=False)
    
    concrete_func = inference_fn.get_concrete_function()
    tf.saved_model.save(model, saved_dir, signatures={'serving_default': concrete_func})
    
    converter = tf.lite.TFLiteConverter.from_saved_model(
        saved_dir, signature_keys=['serving_default']
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # ТОЛЬКО TFLITE_BUILTINS (без SELECT_TF_OPS)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter._experimental_lower_tensor_list_ops = True  # ИЗМЕНЕНО на True
    
    tflite_model = converter.convert()
    
    with open(tflite_file, 'wb') as f:
        f.write(tflite_model)
    print(f'✅ Created {tflite_file}')

# ===============================
# 1. Детектор (RGB изображения)
# ===============================
# detector = keras_ocr.detection.Detector().model
# save_and_convert_detector(
#     model=detector,
#     input_shape=(300, 300, 3),  # RGB
#     saved_dir='saved_models/detector_tf',
#     tflite_file='detector.tflite'
# )

# ===============================
# 2. Распознаватель (Grayscale)
# ===============================
recognizer = keras_ocr.recognition.Recognizer().model
save_and_convert_recognizer(
    model=recognizer,
    input_shape=(31, 200, 1),  # Grayscale (1 канал)
    saved_dir='saved_models/recognizer_tf',
    tflite_file='recognizer.tflite'
)

# def create_simple_recognizer(input_shape=(31, 200, 1), num_classes=96):
#     """CNN-модель без LSTM, Relu6 и других Flex операций"""
#     inp = tf.keras.Input(shape=input_shape, name='input_image')
    
#     # CNN блоки с обычным ReLU (не ReLU6)
#     x = tf.keras.layers.Conv2D(32, 3, padding='same')(inp)
#     x = tf.keras.layers.Activation('relu')(x)  # Обычный ReLU
#     x = tf.keras.layers.MaxPooling2D()(x)
    
#     x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.MaxPooling2D()(x)
    
#     x = tf.keras.layers.Conv2D(128, 3, padding='same')(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
#     # Классификатор
#     x = tf.keras.layers.Dense(256)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Dropout(0.5)(x)
#     out = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
    
#     return tf.keras.Model(inputs=inp, outputs=out)

# def save_simple_recognizer(input_shape, saved_dir, tflite_file):
#     if os.path.exists(saved_dir):
#         tf.io.gfile.rmtree(saved_dir)
    
#     # Создаём модель
#     model = create_simple_recognizer(input_shape)
    
#     # Сохраняем SavedModel
#     model.save(saved_dir, include_optimizer=False, save_format='tf')
    
#     # Конвертируем в TFLite БЕЗ SELECT_TF_OPS
#     converter = tf.lite.TFLiteConverter.from_saved_model(
#         saved_dir, signature_keys=['serving_default']
#     )
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # ТОЛЬКО BUILTINS
    
#     tflite_model = converter.convert()
    
#     with open(tflite_file, 'wb') as f:
#         f.write(tflite_model)
#     print(f'✅ Created {tflite_file} (NO FLEX OPS)')

# # Создаём простой recognizer
# save_simple_recognizer(
#     input_shape=(31, 200, 1),
#     saved_dir='saved_models/simple_recognizer_tf',
#     tflite_file='recognizer.tflite'
# )