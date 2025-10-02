import tensorflow as tf


# Конвертация в TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('models')
tflite_model = converter.convert()

# Сохранение
with open('recognition_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Модель успешно экспортирована в recognition_model.tflite")