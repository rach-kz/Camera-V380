import os
import time
import numpy as np
import pickle
from typing import List, Tuple, Optional
import cv2

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
        self.status_message = "Готов к работе"
        self.status_color = (0, 200, 0)
        self.status_timestamp = time.time()
        self.status_duration = 3  # Длительность показа сообщения (сек)

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
                self.set_status(f"[INFO] Загружен кэш с {len(self.known_face_names)} лицами")
                return True
            except Exception as e:
                self.set_status(f"[WARNING] Не удалось загрузить кэш: {e}")
        return False

    def save_cache(self) -> None:
        """Сохранение кэша признаков лиц в файл"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f) #type: ignore
            self.set_status(f"[INFO] Кэш сохранен ({len(self.known_face_names)} лиц)")
        except Exception as e:
            self.set_status(f"[ERROR] Не удалось сохранить кэш: {e}")

    def load_known_faces(self) -> None:
        """Загрузка известных лиц из директории"""
        os.makedirs(self.known_faces_dir, exist_ok=True)

        valid_images = [f for f in os.listdir(self.known_faces_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not valid_images:
            self.set_status(f"[WARNING] В папке {self.known_faces_dir} нет изображений для распознавания")
            return

        loaded_faces = 0
        for filename in valid_images:
            img_path = os.path.join(self.known_faces_dir, filename)
            image = cv2.imread(img_path)

            if image is None:
                self.set_status(f"[ERROR] Не удалось загрузить: {filename}")
                continue

            faces = self.detect_faces(image)
            if faces is None or len(faces) == 0:
                self.set_status(f"[WARNING] Лица не обнаружены: {filename}")
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

        self.set_status(f"[INFO] Загружено {loaded_faces} известных лиц")
        if loaded_faces > 0:
            self.save_cache()

    def detect_faces(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Обнаружение лиц на изображении
        Returns: Массив с координатами лиц или None
        """
        h, w = image.shape[:2]
        self.face_detector.setInputSize((w, h))
        _, faces = self.face_detector.detect(image)
        return faces if faces is not None and len(faces) > 0 else None


    def recognize_face(self, image: np.ndarray, face: np.ndarray) -> Tuple[str, float]:
        """Распознавание лица"""

        if image is None or image.size == 0:
            return "Error", 0.0

        try:
            face_align = self.face_recognizer.alignCrop(image, face)
            face_feature = self.face_recognizer.feature(face_align)
        except Exception as e:
            self.set_status(f"[ERROR] Ошибка при обработке лица: {e}")
            return "Error", 0.0

        if not self.known_face_encodings:
            return "NoDB", 0.0

        similarities = [self.face_recognizer.match(face_feature, enc, cv2.FaceRecognizerSF_FR_COSINE)
                        for enc in self.known_face_encodings]
        max_idx = int(np.argmax(similarities))
        return self.known_face_names[max_idx], float(similarities[max_idx])


    def process_frame(self, frame: np.ndarray,
                      confidence_threshold: float = 0.65
                      ) -> np.ndarray:
        """Обработка кадра: обнаружение и распознавание лиц"""
        faces = self.detect_faces(frame)

        if faces is not None:
            for face in faces:
                try:
                    if len(face[:4]) >= 4:
                        x, y, w, h = map(int, face[:4])
                        frame_h, frame_w = frame.shape[:2]
                        if w < 20 or h < 20 or x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
                            continue

                        # Распознавание лица
                        name, confidence = self.recognize_face(frame[y:y+h, x:x+w], face)

                        # Визуализация =============================================
                        if name in ["NoDB", "Error"]:
                            color =  (200, 0, 0)
                            label =  "Unknown"
                        elif confidence >= confidence_threshold:
                            color =  (0, 200, 0)
                            label =  f"{name} ({confidence:.2f})"
                        else:
                            color = (200, 200, 0)
                            label = f"{name}? ({confidence:.2f})"

                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.7, 2)[0]
                        cv2.rectangle(frame, (x, y - label_size[1] - 10),
                                      (x + label_size[0] + 10, y), color, -1)
                        cv2.putText(frame, label, (x + 5, y - 5),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (200, 200, 200), 2)
                        #============================================================


                except Exception as e:
                    self.set_status(f"[WARNING] Ошибка обработки лица: {e}")
                    continue
        #Статус-бар
        frame = self.draw_status_bar(frame)
        return frame

    def get_known_faces(self) -> List[str]:
        """Получить список известных лиц"""
        return self.known_face_names.copy()


    def set_status(self, message: str, duration: float = 3):
        """Установка сообщения в статус-бар"""
        self.status_message = message
        self.status_timestamp = time.time()
        self.status_duration = duration
        if message[2] == 'E':
            self.status_color = (0, 0, 200)
        elif message[2] == 'W':
            self.status_color = (200, 200, 0)
        else:
            self.status_color = (0, 200, 0)

    def draw_status_bar(self, frame):
        """Рисование статус-бара внизу экрана"""
        height, width = frame.shape[:2]
        bar_height = 40
        bar_y = height - bar_height

        # Полупрозрачный фон для статус-бара
        overlay = frame[bar_y:height, 0:width].copy()
        cv2.rectangle(overlay, (0, 0), (width, bar_height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame[bar_y:height, 0:width], 0.3, 0,
                        frame[bar_y:height, 0:width])

        # Рисуем разделительную линию
        cv2.line(frame, (0, bar_y), (width, bar_y), (100, 100, 100), 2)

        # Статус сообщение
        # Проверяем, не истекло ли время показа
        if time.time() - self.status_timestamp > self.status_duration:
            self.status_message = "Готов к работе (h - справка)"

        # Рисуем текст статуса
        cv2.putText(frame, f"{self.status_message}", (20, bar_y + 28),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, self.status_color, 2)
        # Рисуем время справа
        cv2.putText(frame, f"Знакомых: {len(self.get_known_faces())}", (width - 200, bar_y + 28),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (200, 200, 200), 2)
        return frame
