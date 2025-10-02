import tensorflow as tf
import numpy as np

# Путь к скачанной модели
model_path = "2.tflite"

# Загрузка TFLite модели
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Получение информации о входных и выходных тензорах
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Вывод информации о модели
print("Информация о входных тензорах:")
print(input_details)
print("\nИнформация о выходных тензорах:")
print(output_details)

# Пример использования модели (замените на свои данные)
# Определите размер входа из input_details
input_shape = input_details[0]['shape']
print(f"\nТребуемая форма входа: {input_shape}")

# Создайте тестовые данные нужного размера
input_data = np.random.random_sample(input_shape).astype(np.float32)

# Установите входной тензор
interpreter.set_tensor(input_details[0]['index'], input_data)

# Выполните инференс
interpreter.invoke()

# Получите результат
output_data = interpreter.get_tensor(output_details[0]['index'])
print(f"\nРезультат инференса: {output_data.shape}")
