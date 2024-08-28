import dlib
import cv2
import numpy as np
import os
import shutil
from PIL import Image


def flip_resize(image, savex):
    img = Image.open(image)
    width, height = img.size
    new_width = int(width * 1.5)
    new_height = int(height * 1.5)
    img_reversed_vertical = img.transpose(Image.FLIP_LEFT_RIGHT)
    e = img_reversed_vertical.resize((new_width, new_height), Image.LANCZOS)
    e.save(savex)


def landmarks_to_np(landmarks, dtype="int"):

    num = landmarks.num_parts
    coords = np.zeros((num, 2), dtype=dtype)
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    return coords


def get_centers(img, landmarks):
    EYE_LEFT_OUTTER = landmarks[2]
    EYE_LEFT_INNER = landmarks[3]
    EYE_RIGHT_OUTTER = landmarks[0]
    EYE_RIGHT_INNER = landmarks[1]

    x = ((landmarks[0:4]).T)[0]
    y = ((landmarks[0:4]).T)[1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]

    x_left = (EYE_LEFT_OUTTER[0]+EYE_LEFT_INNER[0])/2
    x_right = (EYE_RIGHT_OUTTER[0]+EYE_RIGHT_INNER[0])/2
    LEFT_EYE_CENTER = np.array([np.int32(x_left), np.int32(x_left*k+b)])
    RIGHT_EYE_CENTER = np.array([np.int32(x_right), np.int32(x_right*k+b)])

    pts = np.vstack((LEFT_EYE_CENTER, RIGHT_EYE_CENTER))
    cv2.polylines(img, [pts], False, (255, 0, 0), 1)
    cv2.circle(
        img, (
            LEFT_EYE_CENTER[0],
            LEFT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    cv2.circle(
        img, (
            RIGHT_EYE_CENTER[0],
            RIGHT_EYE_CENTER[1]), 3, (0, 0, 255), -1)

    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER


def get_aligned_face(img, left, right):
    desired_w = 256
    desired_h = 256
    desired_dist = desired_w * 0.5

    eyescenter = ((left[0]+right[0])*0.5, (left[1]+right[1])*0.5)
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    dist = np.sqrt(dx*dx + dy*dy)
    scale = desired_dist / dist
    angle = np.degrees(np.arctan2(dy, dx))
    M = cv2.getRotationMatrix2D(eyescenter, angle, scale)

    tX = desired_w * 0.5
    tY = desired_h * 0.5
    M[0, 2] += (tX - eyescenter[0])
    M[1, 2] += (tY - eyescenter[1])

    aligned_face = cv2.warpAffine(img, M, (desired_w, desired_h))

    return aligned_face


def judge_eyeglass(img):
    img = cv2.GaussianBlur(img, (11, 11), 0)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    edgeness = sobel_y

    retVal, thresh = cv2.threshold(
        edgeness, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    d = len(thresh) * 0.5
    x = np.int32(d * 6/7)
    y = np.int32(d * 3/4)
    w = np.int32(d * 2/7)
    h = np.int32(d * 2/4)

    x_2_1 = np.int32(d * 1/4)
    x_2_2 = np.int32(d * 5/4)
    w_2 = np.int32(d * 1/2)
    y_2 = np.int32(d * 8/7)
    h_2 = np.int32(d * 1/2)

    roi_1 = thresh[y:y+h, x:x+w]
    roi_2_1 = thresh[y_2:y_2+h_2, x_2_1:x_2_1+w_2]
    roi_2_2 = thresh[y_2:y_2+h_2, x_2_2:x_2_2+w_2]
    roi_2 = np.hstack([roi_2_1, roi_2_2])

    measure_1 = sum(sum(roi_1/255)) / (np.shape(roi_1)[0] * np.shape(roi_1)[1])
    measure_2 = sum(sum(roi_2/255)) / (np.shape(roi_2)[0] * np.shape(roi_2)[1])
    measure = measure_1*0.3 + measure_2*0.7
    if measure > 0.15:
        judge = True
    else:
        judge = False
    return judge


def detective_glass(image):
    predictor_path = "model/train.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    image_path = image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for i, rect in enumerate(rects):
        x_face = rect.left()
        y_face = rect.top()
        w_face = rect.right() - x_face
        h_face = rect.bottom() - y_face

        cv2.rectangle(
            img, (x_face, y_face), (x_face + w_face,
                                    y_face + h_face), (0, 255, 0), 2)
        cv2.putText(
            img, "Face #{}".format(i + 1), (x_face - 10, y_face - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        landmarks = predictor(gray, rect)
        landmarks = landmarks_to_np(landmarks)
        for (x, y) in landmarks:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

        LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(img, landmarks)
        aligned_face = get_aligned_face(
            gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)

        judge = judge_eyeglass(aligned_face)
        if judge:
            return True
        else:
            return False


def is_human_face(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "model/shape_predictor_68_face_landmarks.dat")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) != 1:
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


if __name__ == "__main__":
    dir_image = "images/"
    directories = ["success", "error"]

    [
        os.makedirs(dir_name) for dir_name in directories if not os.path.exists(dir_name)
    ]

    image_paths = os.listdir(dir_image)
    for image_path in image_paths:
        print("[*] PROCESSING: "+image_path)
        isp = is_human_face(dir_image+image_path)
        if isp:
            dev = detective_glass(dir_image+image_path)
            if dev:
                shutil.copy(f'{dir_image}{image_path}', f"error/{image_path}")
            else:
                flip_resize(f'{dir_image}{image_path}',f"success/{image_path}")

        else:
            shutil.copy(f'{dir_image}{image_path}', f"error/{image_path}")
