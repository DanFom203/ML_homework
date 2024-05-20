import cv2
import mediapipe as mp
import numpy as np

# Инициализация Face Mesh и Hands моделей
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.75)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)


# Загрузка эталонных изображений вашего лица и извлечение ключевых точек
def get_face_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    return None


# Список изображений для обучения
image_paths = ['C:/Users/Danii/PycharmProjects/pythonProject/homework5/imgs/photo_1.jpg',
               'C:/Users/Danii/PycharmProjects/pythonProject/homework5/imgs/photo_2.jpg',
               'C:/Users/Danii/PycharmProjects/pythonProject/homework5/imgs/photo_3.jpg',
               'C:/Users/Danii/PycharmProjects/pythonProject/homework5/imgs/photo_4.jpg',
               'C:/Users/Danii/PycharmProjects/pythonProject/homework5/imgs/photo_5.jpg',
               'C:/Users/Danii/PycharmProjects/pythonProject/homework5/imgs/photo_6.jpg',
               'C:/Users/Danii/PycharmProjects/pythonProject/homework5/imgs/photo_7.jpg',
               'C:/Users/Danii/PycharmProjects/pythonProject/homework5/imgs/photo_8.jpg',
               'C:/Users/Danii/PycharmProjects/pythonProject/homework5/imgs/photo_9.jpg',
               'C:/Users/Danii/PycharmProjects/pythonProject/homework5/imgs/photo_10.jpg',
               'C:/Users/Danii/PycharmProjects/pythonProject/homework5/imgs/photo_11.jpg',
               'C:/Users/Danii/PycharmProjects/pythonProject/homework5/imgs/photo_12.jpg',
               'C:/Users/Danii/PycharmProjects/pythonProject/homework5/imgs/photo_13.jpg']
reference_landmarks_list = []

for path in image_paths:
    landmarks = get_face_landmarks(path)
    if landmarks is not None:
        reference_landmarks_list.append(landmarks)

if not reference_landmarks_list:
    raise Exception("Could not find faces in the reference images.")


# Функция для вычисления евклидова расстояния между двумя наборами ключевых точек
def calculate_distance(landmarks1, landmarks2):
    landmarks1 = np.array([(lm.x, lm.y, lm.z) for lm in landmarks1])
    landmarks2 = np.array([(lm.x, lm.y, lm.z) for lm in landmarks2])
    return np.linalg.norm(landmarks1 - landmarks2)


# Функция для подсчета поднятых пальцев
def count_raised_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]  # Индексы кончиков пальцев в landmarks
    count = 0

    for tip in finger_tips:
        if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
            count += 1

    return count


# Захват видео с веб-камеры
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Преобразование изображения в RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Обработка изображения с помощью Face Mesh и Hands
    face_results = face_mesh.process(img_rgb)
    hand_results = hands.process(img_rgb)

    recognized = False

    # Если обнаружено лицо
    if face_results.multi_face_landmarks:
        current_landmarks = face_results.multi_face_landmarks[0].landmark
        distances = [calculate_distance(ref_landmarks, current_landmarks) for ref_landmarks in reference_landmarks_list]

        # Установка порогового значения для распознавания вашего лица
        threshold = 1
        min_distance = min(distances)
        if min_distance < threshold:
            recognized = True
            label = "Daniil Fomin"
        else:
            label = "Unknown"

        # Отображение метки на изображении
        cv2.putText(img, label, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # Отрисовка ключевых точек на лице
        for lm in current_landmarks:
            x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

    # Если лицо распознано и обнаружена рука
    if recognized and hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Подсчет поднятых пальцев
            num_fingers = count_raised_fingers(hand_landmarks.landmark)

            if num_fingers == 1:
                cv2.putText(img, "Daniil", (20, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            elif num_fingers == 2:
                cv2.putText(img, "Fomin", (20, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    # Отображение видео
    cv2.imshow('Face and Hand Recognition', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
