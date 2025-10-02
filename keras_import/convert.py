# build_and_export_logits.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # безопасная загрузка на CPU
import tensorflow as tf
from tensorflow import keras

def build_crnn_logits(alphabet, height, width, color, filters, rnn_units,
                      dropout, rnn_steps_to_discard, pool_size, stn=True):
    # Ваш код до формирования x без изменений ...
    inputs = keras.layers.Input((height, width, 3 if color else 1))
    x = keras.layers.Permute((2, 1, 3))(inputs)
    x = keras.layers.Lambda(lambda x: x[:, :, ::-1])(x)
    x = keras.layers.Conv2D(filters[0], 3, activation="relu", padding="same", name="conv_1")(x)
    x = keras.layers.Conv2D(filters[1], 3, activation="relu", padding="same", name="conv_2")(x)
    x = keras.layers.Conv2D(filters[2], 3, activation="relu", padding="same", name="conv_3")(x)
    x = keras.layers.BatchNormalization(name="bn_3")(x)
    x = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), name="maxpool_3")(x)
    x = keras.layers.Conv2D(filters[3], 3, activation="relu", padding="same", name="conv_4")(x)
    x = keras.layers.Conv2D(filters[4], 3, activation="relu", padding="same", name="conv_5")(x)
    x = keras.layers.BatchNormalization(name="bn_5")(x)
    x = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), name="maxpool_5")(x)
    x = keras.layers.Conv2D(filters[5], 3, activation="relu", padding="same", name="conv_6")(x)
    x = keras.layers.Conv2D(filters[6], 3, activation="relu", padding="same", name="conv_7")(x)
    x = keras.layers.BatchNormalization(name="bn_7")(x)

    # ВАЖНО: STN в вашей сборке реализован как Lambda(_transform, [x, localization_net(x)]).
    # Для конвертации без Flex оставьте как есть, но _transform должен быть чистым TF-ops без tf.numpy_function.
    # При наличии кастомной Python-функции TFLite не сконвертирует. Для простоты можно временно отключить stn=False.
    # x = <оставьте x либо примените STN, если он реализован тензорными операциями>

    x = keras.layers.Reshape(target_shape=(width // (pool_size**2),
                                           (height // (pool_size**2)) * filters[-1]),
                             name="reshape")(x)
    x = keras.layers.Dense(rnn_units[0], activation="relu", name="fc_9")(x)
    rnn_1_f = keras.layers.LSTM(rnn_units[0], kernel_initializer="he_normal",
                                return_sequences=True, name="lstm_10")(x)
    rnn_1_b = keras.layers.LSTM(rnn_units[0], kernel_initializer="he_normal",
                                go_backwards=True, return_sequences=True, name="lstm_10_back")(x)
    rnn_1 = keras.layers.Add()([rnn_1_f, rnn_1_b])
    rnn_2_f = keras.layers.LSTM(rnn_units[1], kernel_initializer="he_normal",
                                return_sequences=True, name="lstm_11")(rnn_1)
    rnn_2_b = keras.layers.LSTM(rnn_units[1], kernel_initializer="he_normal",
                                go_backwards=True, return_sequences=True, name="lstm_11_back")(rnn_1)
    x = keras.layers.Concatenate()([rnn_2_f, rnn_2_b])
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    logits = keras.layers.Dense(len(alphabet) + 1, kernel_initializer="he_normal",
                                activation="softmax", name="fc_12")(x)
    logits = keras.layers.Lambda(lambda t: t[:, rnn_steps_to_discard:], name="time_trim")(logits)

    # Модель для инференса на устройстве: ВОЗВРАЩАЕТ ТОЛЬКО LOGITS [B, T, C]
    infer_model = keras.models.Model(inputs=inputs, outputs=logits, name="crnn_logits")
    return infer_model

# Экспорт в TFLite
def export_tflite(infer_model: tf.keras.Model, out_dir="saved_crnn_logits", tflite_path="crnn_logits_fp16.tflite"):
    tf.saved_model.save(infer_model, out_dir)
    conv = tf.lite.TFLiteConverter.from_saved_model(out_dir)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tflite = conv.convert()
    open(tflite_path, "wb").write(tflite)
    print("Saved:", tflite_path)

if __name__ == "__main__":
    # Пример параметров
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    height, width = 31, 200
    color = False
    filters = [64, 64, 128, 128, 256, 256, 256]
    rnn_units = [256, 256]
    dropout = 0.5
    rnn_steps_to_discard = 0
    pool_size = 2
    stn = False  # для надёжной конвертации

    model = build_crnn_logits(alphabet, height, width, color, filters, rnn_units,
                              dropout, rnn_steps_to_discard, pool_size, stn=stn)
    export_tflite(model)
