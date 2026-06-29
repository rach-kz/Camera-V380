import os
import time
import numpy as np
import pickle
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import cv2
import logging

logger = logging.getLogger(__name__)


@dataclass
class FaceRecognitionResult:
    """Результат распознавания лица"""
    name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h

    @property
    def is_recognized(self) -> bool:
        return self.name not in ["NoDB", "Error", "Unknown"]

    @property
    def is_confident(self, threshold: float = 0.65) -> bool:
        return self.confidence >= threshold


class FaceRecognizer:
    # Константы класса
    MIN_FACE_SIZE = 20
    DEFAULT_CONFIDENCE_THRESHOLD = 0.65
    CACHE_VERSION = 1  # Для проверки совместимости кэша

    # Цвета в одном месте для легкой настройки
    COLORS = {
        'unknown': (200, 0, 0),
        'recognized': (0, 200, 0),
        'uncertain': (0, 204, 255),
        'error': (0, 0, 200),
        'warning': (0, 204, 255),
        'info': (0, 200, 0),
        'text': (180, 180, 180),
        'background': (30, 30, 30),
        'line': (100, 100, 100),
    }

    def __init__(self,
                 detection_model: str = "face_detection_yunet_2023mar.onnx",
                 recognition_model: str = "face_recognition_sface_2021dec.onnx",
                 known_faces_dir: str = "known_faces",
                 cache_file: str = "face_cache.pkl",
                 confidence_threshold: float = 0.65,
                 auto_save_cache: bool = True):

        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.known_faces_dir = Path(known_faces_dir)
        self.cache_file = Path(cache_file)
        self.confidence_threshold = confidence_threshold
        self.auto_save_cache = auto_save_cache

        self._status_message = "Готов к работе"
        self._status_color = self.COLORS['info']
        self._status_timestamp = time.time()
        self._status_duration = 3


        # Валидация и инициализация моделей
        self._init_models(detection_model, recognition_model)

        # Загрузка данных
        self._load_data()

    def _init_models(self, detection_model: str, recognition_model: str) -> None:
        """Инициализация моделей с обработкой ошибок"""
        for model_path in [detection_model, recognition_model]:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Модель не найдена: {model_path}")
            if os.path.getsize(model_path) == 0:
                raise ValueError(f"Файл модели пуст: {model_path}")
        try:
            self.face_detector = cv2.FaceDetectorYN.create(
                detection_model, "", (320, 320), 0.85, 0.3, 5000
            )
            self.face_recognizer = cv2.FaceRecognizerSF.create(
                recognition_model, ""
            )
        except cv2.error as e:
            raise RuntimeError(f"Ошибка инициализации моделей OpenCV: {e}")

    def _load_data(self) -> None:
        """Загрузка данных с приоритетом кэша"""
        if not self._load_cache():
            self.load_known_faces()
        elif not self.known_face_encodings:
            logger.warning("Кэш пуст, загружаем из директории")
            self.load_known_faces()

    def _load_cache(self) -> bool:
        """Загрузка кэша с проверкой версии"""
        if not self.cache_file.exists():
            return False

        try:
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)

            # Проверка версии кэша
            if data.get('version') != self.CACHE_VERSION:
                logger.info("Версия кэша устарела, будет создан новый")
                return False

            self.known_face_encodings = data['encodings']
            self.known_face_names = data['names']
            self.set_status(f"[INFO] Загружен кэш с {len(self.known_face_names)} лицами")
            return True

        except (pickle.UnpicklingError, KeyError, EOFError) as e:
            logger.warning(f"Поврежденный кэш: {e}")
            self.cache_file.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша: {e}")

        return False

    def save_cache(self) -> None:
        """Безопасное сохранение кэша с атомарной записью"""
        if not self.auto_save_cache:
            return

        temp_file = self.cache_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'wb') as f:
                pickle.dump({
                    'version': self.CACHE_VERSION,
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names,
                    'timestamp': time.time()
                }, f) #type: ignore

            # Атомарное переименование
            temp_file.replace(self.cache_file)
            logger.debug(f"Кэш сохранен: {len(self.known_face_names)} лиц")

        except Exception as e:
            logger.error(f"Ошибка сохранения кэша: {e}")
            temp_file.unlink(missing_ok=True)

    def load_known_faces(self) -> None:
        """Загрузка известных лиц с улучшенной обработкой"""
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)

        # Поддерживаемые форматы
        image_patterns = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        valid_images = []
        for pattern in image_patterns:
            valid_images.extend(self.known_faces_dir.glob(pattern))

        if not valid_images:
            self.set_status(f"[WARNING] Нет изображений в {self.known_faces_dir}")
            return

        loaded_faces = 0
        errors = 0

        for img_path in valid_images:
            try:
                if self._process_known_face(img_path):
                    loaded_faces += 1
                else:
                    errors += 1
            except Exception as e:
                logger.error(f"Ошибка обработки {img_path.name}: {e}")
                errors += 1

        status = f"[INFO] Загружено {loaded_faces} лиц"
        if errors > 0:
            status += f" (ошибок: {errors})"
        self.set_status(status)

        if loaded_faces > 0 and self.auto_save_cache:
            self.save_cache()

    def _process_known_face(self, img_path: Path) -> bool:
        """Обработка одного известного лица"""
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"Не удалось прочитать: {img_path.name}")
            return False

        faces = self.detect_faces(image)
        if faces is None:
            logger.warning(f"Лица не найдены: {img_path.name}")
            return False

        # Берем самое большое лицо если их несколько
        if len(faces) > 1:
            face = max(faces, key=lambda f: f[2] * f[3])  # По площади
            logger.debug(f"Найдено {len(faces)} лиц в {img_path.name}, выбрано наибольшее")
        else:
            face = faces[0]

        face_align = self.face_recognizer.alignCrop(image, face)
        face_feature = self.face_recognizer.feature(face_align)
        name = img_path.stem

        # Обновление или добавление с весовым усреднением
        if name in self.known_face_names:
            idx = self.known_face_names.index(name)
            # Взвешенное среднее с большим весом для существующих данных
            self.known_face_encodings[idx] = (
                    0.7 * self.known_face_encodings[idx] + 0.3 * face_feature
            )
        else:
            self.known_face_encodings.append(face_feature)
            self.known_face_names.append(name)

        return True

    def detect_faces(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Обнаружение лиц с валидацией размеров"""
        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]
        self.face_detector.setInputSize((w, h))

        try:
            _, faces = self.face_detector.detect(image)
        except cv2.error as e:
            logger.error(f"Ошибка детекции: {e}")
            return None

        if faces is None:
            return None

        # Фильтрация слишком маленьких лиц
        valid_faces = []
        for face in faces:
            if len(face) >= 4:
                _, _, fw, fh = face[:4]
                if fw >= self.MIN_FACE_SIZE and fh >= self.MIN_FACE_SIZE:
                    valid_faces.append(face)

        return np.array(valid_faces) if valid_faces else None

    def recognize_face(self, face_image: np.ndarray, face: np.ndarray) -> Tuple[str, float]:
        """Распознавание с улучшенной обработкой ошибок"""
        if face_image is None or face_image.size == 0:
            return "Error", 0.0

        try:
            face_align = self.face_recognizer.alignCrop(face_image, face)
            face_feature = self.face_recognizer.feature(face_align)

            # Проверка качества фичи
            if face_feature is None or np.all(face_feature == 0):
                return "Error", 0.0

        except cv2.error as e:
            logger.error(f"Ошибка распознавания: {e}")
            return "Error", 0.0

        if not self.known_face_encodings:
            return "NoDB", 0.0

        # Векторизованное вычисление сходства (быстрее)
        similarities = np.array([
            self.face_recognizer.match(
                face_feature, enc, cv2.FaceRecognizerSF_FR_COSINE
            )
            for enc in self.known_face_encodings
        ])

        max_idx = int(np.argmax(similarities))
        return self.known_face_names[max_idx], float(similarities[max_idx])

    def process_frame(self, frame: np.ndarray,
                      #confidence_threshold: Optional[float] = None
                      ) -> np.ndarray:
        """Обработка кадра с улучшенной отрисовкой"""
        if frame is None or frame.size == 0:
            return frame

        faces = self.detect_faces(frame)

        if faces is not None:
            frame_h, frame_w = frame.shape[:2]

            for face in faces:
                try:
                    result = self._process_single_face(frame, face, frame_w, frame_h)
                    if result:
                        self._draw_face_box(frame, result)
                except Exception as e:
                    logger.debug(f"Ошибка обработки лица: {e}")
                    continue

        return self._draw_status_bar(frame)

    def _process_single_face(self, frame: np.ndarray, face: np.ndarray,
                            frame_w: int, frame_h: int
                             ) -> Optional[FaceRecognitionResult]:
        """Обработка одного лица"""
        x, y, w, h = map(int, face[:4])

        # Валидация координат
        if not self._is_valid_bbox(x, y, w, h, frame_w, frame_h):
            return None

        # Извлечение ROI лица
        face_roi = frame[max(0, y):min(y+h, frame_h),
        max(0, x):min(x+w, frame_w)]

        if face_roi.size == 0:
            return None

        # Распознавание
        name, confidence = self.recognize_face(face_roi, face)

        return FaceRecognitionResult(
            name=name,
            confidence=confidence,
            bbox=(x, y, w, h)
        )

    @staticmethod
    def _is_valid_bbox(x: int, y: int, w: int, h: int,
                       frame_w: int, frame_h: int) -> bool:
        """Проверка валидности bounding box"""
        if w < FaceRecognizer.MIN_FACE_SIZE or h < FaceRecognizer.MIN_FACE_SIZE:
            return False
        if x < 0 or y < 0:
            return False
        if x + w > frame_w or y + h > frame_h:
            return False
        return True

    def _draw_face_box(self, frame: np.ndarray, result: FaceRecognitionResult) -> None:
        """Отрисовка рамки и метки лица"""
        x, y, w, h = result.bbox

        # Выбор цвета и метки
        if result.name in ["NoDB", "Error"]:
            color = self.COLORS['unknown']
            label = "Unknown"
        elif result.is_confident:
            color = self.COLORS['recognized']
            label = f"{result.name} ({result.confidence:.2f})"
        else:
            color = self.COLORS['uncertain']
            label = f"{result.name}? ({result.confidence:.2f})"

        # Отрисовка
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        self._draw_label(frame, label, x, y, color)

    def _draw_label(self, frame: np.ndarray, label: str,
                    x: int, y: int, color: Tuple[int, int, int]) -> None:
        """Отрисовка текстовой метки"""
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 0.8
        thickness = 2

        (label_w, label_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Фон для текста с отступом
        padding = 5
        cv2.rectangle(
            frame,
            (x, y - label_h - baseline - padding * 2),
            (x + label_w + padding * 2, y),
            color, -1
        )

        # Текст
        cv2.putText(
            frame, label,
            (x + padding, y - baseline - padding),
            font, font_scale, self.COLORS['text'], thickness
        )

    def _draw_status_bar(self, frame: np.ndarray) -> np.ndarray:
        """Отрисовка информационной панели"""
        height, width = frame.shape[:2]
        bar_height = 40
        bar_y = height - bar_height

        # Полупрозрачный фон
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (width, height), self.COLORS['background'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Разделительная линия
        cv2.line(frame, (0, bar_y), (width, bar_y), self.COLORS['line'], 2)

        # Обновление статуса по тайм-ауту
        if time.time() - self._status_timestamp > self._status_duration:
            self._status_message = "Готов к работе (h - справка)"
            self._status_color = self.COLORS['info']

        # Текст статуса (слева)
        cv2.putText(frame, self._status_message,
                    (20, bar_y + 28), cv2.FONT_HERSHEY_COMPLEX,
                    0.8, self._status_color, 2)

        # Количество известных лиц (справа)
        known_count_text = f"Знакомых: {len(self.known_face_names)}"
        text_size = cv2.getTextSize(known_count_text, cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)[0]
        cv2.putText(frame, known_count_text,
                    (width - text_size[0] - 20, bar_y + 28),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, self.COLORS['text'], 2)

        return frame

    def get_known_faces(self) -> List[str]:
        """Получить список известных лиц"""
        return self.known_face_names.copy()

    def set_status(self, message: str, duration: float = 3) -> None:
        """Установка статусного сообщения"""
        self._status_message = message
        self._status_timestamp = time.time()
        self._status_duration = duration

        # Определение цвета по префиксу
        if message.startswith('[ERROR]'):
            self._status_color = self.COLORS['error']
        elif message.startswith('[WARNING]'):
            self._status_color = self.COLORS['warning']
        else:
            self._status_color = self.COLORS['info']

    def remove_known_face(self, name: str) -> bool:
        """Удаление известного лица"""
        if name in self.known_face_names:
            idx = self.known_face_names.index(name)
            del self.known_face_names[idx]
            del self.known_face_encodings[idx]
            self.save_cache()
            return True
        return False

    def clear_database(self) -> None:
        """Очистка базы известных лиц"""
        self.known_face_encodings.clear()
        self.known_face_names.clear()
        self.cache_file.unlink(missing_ok=True)
        self.set_status("[INFO] База лиц очищена")

    # Свойства для инкапсуляции
    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def status_color(self) -> Tuple[int, int, int]:
        return self._status_color

    @property
    def known_faces_count(self) -> int:
        return len(self.known_face_names)