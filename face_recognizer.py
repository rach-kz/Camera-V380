import cv2
import os
import numpy as np
import pickle
from typing import List, Tuple, Optional
from datetime import datetime

class FaceRecognizer:
    def __init__(self,
                 detection_model: str = "face_detection_yunet_2023mar.onnx",
                 recognition_model: str = "face_recognition_sface_2021dec.onnx",
                 known_faces_dir: str = "known_faces",
                 cache_file: str = "face_cache.pkl"):

        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.known_faces_dir = known_faces_dir
        self.cache_file = cache_file

        # Проверка наличия моделей
        if not os.path.exists(detection_model):
            raise FileNotFoundError(f"Модель детекции не найдена: {detection_model}")
        if not os.path.exists(recognition_model):
            raise FileNotFoundError(f"Модель распознавания не найдена: {recognition_model}")

        # Инициализация детектора и распознавателя
        self.face_detector = cv2.FaceDetectorYN.create(
            detection_model,
            "",
            (320, 320),
            0.85,
            0.3,
            5000
        )
        self.face_recognizer = cv2.FaceRecognizerSF.create(
            recognition_model,
            ""
        )

        # Загрузка кэша или известных лиц
        if not self.load_cache():
            self.load_known_faces()

    def load_cache(self) -> bool:
        """Загрузка кэша признаков лиц из файла"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"[INFO] Загружен кэш с {len(self.known_face_names)} лицами")
                return True
            except Exception as e:
                print(f"[WARNING] Не удалось загрузить кэш: {e}")
        return False

    def save_cache(self) -> None:
        """Сохранение кэша признаков лиц в файл"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f)# type: ignore
            print(f"[INFO] Кэш сохранен ({len(self.known_face_names)} лиц)")
        except Exception as e:
            print(f"[WARNING] Не удалось сохранить кэш: {e}")

    def load_known_faces(self) -> None:
        """Загрузка известных лиц из директории"""
        os.makedirs(self.known_faces_dir, exist_ok=True)

        valid_images = [f for f in os.listdir(self.known_faces_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not valid_images:
            print(f"[INFO] В папке {self.known_faces_dir} нет изображений для распознавания")
            return

        loaded_faces = 0
        for filename in valid_images:
            img_path = os.path.join(self.known_faces_dir, filename)
            image = cv2.imread(img_path)

            if image is None:
                print(f"[WARNING] Не удалось загрузить: {filename}")
                continue

            faces = self.detect_faces(image)
            # ИСПРАВЛЕНО: правильная проверка
            if faces is None or len(faces) == 0:
                print(f"[WARNING] Лица не обнаружены на фото: {filename}")
                continue

            face_align = self.face_recognizer.alignCrop(image, faces[0])
            face_feature = self.face_recognizer.feature(face_align)

            name = os.path.splitext(filename)[0]

            if name in self.known_face_names:
                idx = self.known_face_names.index(name)
                self.known_face_encodings[idx] = (self.known_face_encodings[idx] + face_feature) / 2
            else:
                self.known_face_encodings.append(face_feature)
                self.known_face_names.append(name)
            loaded_faces += 1

        print(f"[INFO] Загружено {loaded_faces} известных лиц")
        if loaded_faces > 0:
            self.save_cache()

    def detect_faces(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Обнаружение лиц на изображении

        Returns:
            Массив с координатами лиц или None
        """
        h, w = image.shape[:2]
        self.face_detector.setInputSize((w, h))
        _, faces = self.face_detector.detect(image)
        # Возвращаем faces как есть (может быть None)
        return faces if faces is not None and len(faces) > 0 else None

    def recognize_face(self, image: np.ndarray, face: np.ndarray) -> Tuple[str, float]:
        """Распознавание лица"""
        try:
            face_align = self.face_recognizer.alignCrop(image, face)
            face_feature = self.face_recognizer.feature(face_align)
        except Exception as e:
            print(f"[ERROR] Ошибка при обработке лица: {e}")
            return "Error", 0.0

        if not self.known_face_encodings:
            return "NoDB", 0.0

        similarities = [self.face_recognizer.match(face_feature, enc, cv2.FaceRecognizerSF_FR_COSINE)
                        for enc in self.known_face_encodings]
        max_idx = int(np.argmax(similarities))
        return self.known_face_names[max_idx], float(similarities[max_idx])

    def process_frame(self, frame: np.ndarray, confidence_threshold: float = 0.65) -> np.ndarray:
        """Обработка кадра: обнаружение и распознавание лиц"""
        # Уменьшение размера для ускорения
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        faces = self.detect_faces(small_frame)

        if faces is not None:
            for face in faces:
                # Исправлено: преобразование координат в int с проверкой на None
                try:
                    # Получаем координаты и преобразуем в int
                    coords = face[:4]
                    if coords is None or len(coords) < 4:
                        continue

                    x = int(round(float(coords[0] * 2)))
                    y = int(round(float(coords[1] * 2)))
                    w = int(round(float(coords[2] * 2)))
                    h = int(round(float(coords[3] * 2)))

                    # Проверка валидности координат
                    if w <= 0 or h <= 0:
                        continue

                    # Распознавание лица
                    name, confidence = self.recognize_face(frame[y:y+h, x:x+w], face)

                    # Визуализация
                    if name == "NoDB":
                        color = (255, 255, 0)
                        label = "Unknown"
                    elif confidence >= confidence_threshold:
                        color = (0, 255, 0)
                        label = f"{name} ({confidence:.2f})"
                    else:
                        color = (0, 0, 255)
                        label = f"Unknown ({confidence:.2f})"

                    # Рисуем прямоугольник (гарантированно int координаты)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    # Фон для текста
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame, (x, y - label_size[1] - 10),
                                  (x + label_size[0] + 10, y), color, -1)
                    cv2.putText(frame, label, (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                except Exception as e:
                    print(f"[WARNING] Ошибка обработки лица: {e}")
                    continue

        return frame

    def add_new_face(self, image: np.ndarray, name: str) -> bool:
        """Добавление нового лица в базу с вырезкой и улучшением качества"""
        faces = self.detect_faces(image)

        if faces is None or len(faces) == 0:
            print("[ERROR] Лица не обнаружены на изображении")
            return False

        # Исправлено: безопасное преобразование координат
        try:
            coords = faces[0][:4]
            x = int(round(float(coords[0])))
            y = int(round(float(coords[1])))
            w = int(round(float(coords[2])))
            h = int(round(float(coords[3])))

            # Проверка валидности
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                print("[ERROR] Невалидные координаты лица")
                return False

            # Вырезаем область лица
            face_region = image[y:y+h, x:x+w]

            if face_region.size == 0:
                print("[ERROR] Пустая область лица")
                return False

            # Улучшение качества
            # 1. Увеличение резкости
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            face_region = cv2.filter2D(face_region, -1, kernel)

            # 2. Коррекция контраста
            face_region = cv2.convertScaleAbs(face_region, alpha=1.2, beta=10)

            # 3. Удаление шума
            face_region = cv2.medianBlur(face_region, 3)

            # 4. Приведение к стандартному размеру
            if face_region.shape[0] < 300 or face_region.shape[1] < 300:
                face_region = cv2.resize(face_region, (300, 300), interpolation=cv2.INTER_CUBIC)

        except Exception as e:
            print(f"[ERROR] Ошибка при обработке лица: {e}")
            return False

        # Получаем дескриптор лица
        face_align = self.face_recognizer.alignCrop(image, faces[0])
        face_feature = self.face_recognizer.feature(face_align)

        if name in self.known_face_names:
            print(f"[WARNING] Имя '{name}' уже существует в базе")
            return False

        self.known_face_encodings.append(face_feature)
        self.known_face_names.append(name)

        # Сохраняем лицо
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.jpg"
        filepath = os.path.join(self.known_faces_dir, filename)
        cv2.imwrite(filepath, face_region, [cv2.IMWRITE_JPEG_QUALITY, 95])

        self.save_cache()
        print(f"[INFO] Лицо '{name}' добавлено (размер: {face_region.shape[1]}x{face_region.shape[0]})")
        return True



    def remove_face(self, name: str) -> bool:
        """Удаление лица из базы"""
        if name not in self.known_face_names:
            print(f"[ERROR] Имя '{name}' не найдено в базе")
            return False

        idx = self.known_face_names.index(name)
        del self.known_face_names[idx]
        del self.known_face_encodings[idx]

        # Удаляем файлы изображений
        for filename in os.listdir(self.known_faces_dir):
            if filename.startswith(name):
                os.remove(os.path.join(self.known_faces_dir, filename))

        self.save_cache()
        print(f"[INFO] Лицо '{name}' удалено из базы")
        return True

    def get_known_faces(self) -> List[str]:
        """Получить список известных лиц"""
        return self.known_face_names.copy()