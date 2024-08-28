import numpy as np
import cv2
import tensorflow as tf
from mtcnn import MTCNN

# Hàm tiền xử lý ảnh cho mô hình TensorFlow
def preprocess_image(image, target_size):
    image = cv2.resize(image, target_size)
    image = image.astype("float32")
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Hàm phát hiện kính sử dụng TensorFlow Lite
def detectedGlass(face_roi):
    interpreter = tf.lite.Interpreter(model_path="model_v2.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Tiền xử lý khuôn mặt
    preprocessed_face = preprocess_image(face_roi, (224, 224))
    
    # Dự đoán
    interpreter.set_tensor(input_details[0]['index'], preprocessed_face)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Giải mã kết quả dự đoán
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(output_data, top=1)[0]
    if any(keyword in decoded_preds[0][1] for keyword in ["sunglass", "goggles", "spectacles", "eyeglasses", "sun glasses"]):
        return True
    return False

# Hàm phát hiện người và kính sử dụng MTCNN và TensorFlow Lite
def detect_person_and_glasses(image_path):
    # Đọc ảnh
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Khởi tạo MTCNN để phát hiện khuôn mặt
    detector = MTCNN()
    faces = detector.detect_faces(image_rgb)
    
    # Kiểm tra nếu chỉ có một khuôn mặt
    if len(faces) != 1:
        return "Ảnh phải chứa chính xác một người."
    face = faces[0]
    x, y, w, h = face['box']
    face_roi = image[y:y+h, x:x+w]
    if detectedGlass(face_roi):
        return "Người này có thể đang đeo kính."

    return "Ảnh hợp lệ: Có một người và không đeo kính."
image_path = 'image_2/100002832710918.png'
result = detect_person_and_glasses(image_path)
print(result)
