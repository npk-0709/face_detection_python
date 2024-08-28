import cv2
from mtcnn import MTCNN

# Tải ảnh
image = cv2.imread('image.jpeg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Khởi tạo MTCNN detector
detector = MTCNN()

# Phát hiện khuôn mặt và các đặc điểm
faces = detector.detect_faces(image_rgb)

# Kiểm tra các đặc điểm
for face in faces:
    box = face['box']
    keypoints = face['keypoints']
    
    # Lấy toạ độ của mắt, mũi, miệng, và tai
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    nose = keypoints['nose']
    mouth_left = keypoints['mouth_left']
    mouth_right = keypoints['mouth_right']
    
    # Vẽ các điểm trên ảnh để kiểm tra
    cv2.circle(image, left_eye, 5, (0, 255, 0), -1)
    cv2.circle(image, right_eye, 5, (0, 255, 0), -1)
    cv2.circle(image, nose, 5, (0, 255, 0), -1)
    cv2.circle(image, mouth_left, 5, (0, 255, 0), -1)
    cv2.circle(image, mouth_right, 5, (0, 255, 0), -1)

# Hiển thị kết quả
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
