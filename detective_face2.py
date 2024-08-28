import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
def is_human_face(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return False

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = [landmarks.part(i) for i in range(36, 42)]
        right_eye = [landmarks.part(i) for i in range(42, 48)]
        nose = [landmarks.part(i) for i in range(27, 36)]
        mouth = [landmarks.part(i) for i in range(48, 68)]
        if left_eye and right_eye and nose and mouth:
            return True
    return False

# Sử dụng hàm kiểm tra trên một hình ảnh
image_path = "image_2/504293099.png"  # Thay bằng đường dẫn tới ảnh của bạn
result = is_human_face(image_path)

if result:
    print("Hình ảnh là mặt người và có đủ đặc điểm.")
else:
    print("Hình ảnh không phải là mặt người hoặc thiếu đặc điểm.")
