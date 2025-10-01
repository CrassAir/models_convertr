from ultralytics import YOLO

# Загрузка обученной модели
model = YOLO('best.pt')

# Экспорт в TFLite формат
model.export(format='tflite')
# Создаст файл best_float32.tflite

