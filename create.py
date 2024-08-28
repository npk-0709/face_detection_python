import tensorflow as tf
model = tf.keras.applications.MobileNetV2(weights="imagenet")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('mobilenet_v2.tflite', 'wb') as f:
    f.write(tflite_model)
