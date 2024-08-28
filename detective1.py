import cv2
from mtcnn import MTCNN

def detect_person(image_path):
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
    keypoints = face['keypoints']
    
    # Kiểm tra kính (phân tích vùng mắt)
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    
    # Phương pháp đơn giản: phân tích vùng xung quanh mắt (hoặc sử dụng mô hình huấn luyện)
    # Ví dụ đơn giản: nếu khoảng cách mắt quá gần, có thể là kính
    eye_distance = abs(left_eye[0] - right_eye[0])
    if eye_distance < 20:
        return "Người này có thể đang đeo kính."
    
    return "Ảnh hợp lệ: Có một người và không đeo kính."

# Sử dụng
image_path = 'image_2/100002832710918.png'
result = detect_person(image_path)
print(result)
