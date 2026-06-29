import os
import threading
import queue
import time
import numpy as np
from typing import Optional, Dict
os.environ["OPENCV_FFMPEG_DEBUG"] = "-1"
import cv2
from face_recognizer import FaceRecognizer
import logging

class RTSPCaptureThread:
    """Поток захвата RTSP с улучшенной обработкой ошибок"""

    def __init__(self, rtsp_url: str, frame_queue: queue.Queue):
        self.rtsp_url = rtsp_url
        self.frame_queue = frame_queue
        self.running = False
        self.thread = None

        # Настройки переподключения
        self.base_reconnect_delay = 0.5
        self.max_reconnect_delay = 30
        self.reconnect_delay = self.base_reconnect_delay

        # Статистика
        self.frame_count = 0
        self.reconnect_count = 0
        self.last_frame_time = 0

        # Настройки видеозахвата
        self.capture_params = {
            cv2.CAP_PROP_BUFFERSIZE: 2,
            cv2.CAP_PROP_FPS: 20,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter.fourcc(*'H264'),
        }

    def start(self):
        """Запуск потока захвата"""
        if self.running:
            logger.warning("Поток уже запущен")
            return

        self.running = True
        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="RTSPCapture"
        )
        self.thread.start()
        logger.info(f"Запущен захват с {self.rtsp_url}")

    def stop(self):
        """Остановка потока с тайм-аутом"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                logger.warning("Поток захвата не остановился за 2 секунды")

    def get_stats(self) -> Dict:
        """Получить статистику захвата"""
        return {
            'frames_captured': self.frame_count,
            'reconnects': self.reconnect_count,
            'queue_size': self.frame_queue.qsize(),
            'fps': self._calculate_fps()
        }

    def _calculate_fps(self) -> float:
        """Расчет текущего FPS"""
        current_time = time.time()
        if self.last_frame_time > 0:
            return 1.0 / (current_time - self.last_frame_time)
        return 0.0

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        """Создание и настройка видеозахвата"""
        try:
            # Пробуем разные методы захвата
            for api in [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]:
                cap = cv2.VideoCapture(self.rtsp_url, api)
                if cap.isOpened():
                    # Применяем настройки
                    for param, value in self.capture_params.items():
                        cap.set(param, value)
                    return cap
                cap.release()
        except Exception as e:
            logger.error(f"Ошибка создания захвата: {e}")
        return None

    def _capture_loop(self):
        """Основной цикл захвата"""
        while self.running:
            cap = None
            try:
                cap = self._create_capture()

                if cap is None:
                    logger.warning("Не удалось подключиться к потоку")
                    time.sleep(self.reconnect_delay)
                    self._increase_reconnect_delay()
                    continue

                self.reconnect_delay = self.base_reconnect_delay
                logger.info("Подключение установлено")

                while self.running:
                    ret, frame = cap.read()

                    if not ret or frame is None:
                        logger.warning("Потеря кадра или соединения")
                        break

                    self.last_frame_time = time.time()
                    self.frame_count += 1

                    # Неблокирующая вставка в очередь
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        # Удаляем старый кадр и добавляем новый
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame)
                        except (queue.Empty, queue.Full):
                            pass

            except Exception as e:
                logger.error(f"Ошибка захвата: {e}")
            finally:
                if cap:
                    cap.release()

            if self.running:
                self.reconnect_count += 1
                time.sleep(self.reconnect_delay)
                self._increase_reconnect_delay()

    def _increase_reconnect_delay(self):
        """Увеличение задержки переподключения"""
        self.reconnect_delay = min(
            self.reconnect_delay * 1.5,
            self.max_reconnect_delay
        )

class DisplayThread:
    """Поток отображения с управлением производительностью"""

    def __init__(self, recognizer: FaceRecognizer,
                 frame_queue: queue.Queue,
                 screenshots_dir: str = "screenshots/"
                 ):
        self.recognizer = recognizer
        self.frame_queue = frame_queue
        self.screenshots_dir = screenshots_dir
        self.window_name = "V380 Face Recognition"
        self.running = False
        self.thread = None


        # Буферизация
        self.current_frame = None
        self.last_displayed_frame = None

        # Статистика
        self.processed_frames = 0
        self.displayed_frames = 0
        self.skip_counter = 0

        # Настройки производительности
        self.process_every_n_frames = 2  # Обрабатываем каждый N-й кадр
        self.display_scale = 0.8  # Масштаб отображения для производительности

        # Обработчики клавиш
        self.key_handlers = {
            ord('q'): self._handle_quit,
            ord('n'): self._handle_show_names,
            ord('l'): self._handle_reload,
            ord('h'): self._handle_help,
            ord('s'): self._handle_screenshot,
            ord('f'): self._handle_fullscreen,
        }

    def start(self):
        """Запуск потока отображения"""
        if self.running:
            logger.warning("Поток отображения уже запущен")
            return

        self.running = True
        self.thread = threading.Thread(
            target=self._display_loop,
            daemon=True,
            name="Display"
        )
        self.thread.start()

    def stop(self):
        """Остановка потока отображения"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        cv2.destroyAllWindows()

    def _display_loop(self):
        """Основной цикл отображения"""
        self._setup_window()

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
                self._process_and_display(frame)
                self._handle_keys()

            except queue.Empty:
                self._display_last_frame()
            except Exception as e:
                logger.error(f"Ошибка отображения: {e}")
                self.recognizer.set_status(f"[ERROR] {str(e)[:50]}")

    def _setup_window(self):
        """Настройка окна отображения"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 480)

    def _process_and_display(self, frame: np.ndarray):
        """Обработка и отображение кадра"""
        self.skip_counter += 1

        # Пропускаем кадры для производительности
        if self.skip_counter % self.process_every_n_frames != 0:
            if self.current_frame is not None:
                self._show_frame(self.current_frame)
            return

        # Обработка кадра
        if frame is not None:
            processed = self.recognizer.process_frame(frame.copy())
            self.current_frame = processed
            self.last_displayed_frame = processed
            self.processed_frames += 1

            self._show_frame(processed)

    def _show_frame(self, frame: np.ndarray):
        """Отображение кадра с масштабированием"""
        if self.display_scale < 1.0:
            h, w = frame.shape[:2]
            new_w, new_h = int(w * self.display_scale), int(h * self.display_scale)
            display_frame = cv2.resize(frame, (new_w, new_h))
        else:
            display_frame = frame

        cv2.imshow(self.window_name, display_frame)
        self.displayed_frames += 1

    def _display_last_frame(self):
        """Отображение последнего кадра при отсутствии новых"""
        if self.last_displayed_frame is not None:
            self._show_frame(self.last_displayed_frame)
            self._handle_keys()

    def _handle_keys(self):
        """Обработка нажатий клавиш"""
        key = cv2.waitKey(1) & 0xFF
        if key in self.key_handlers:
            self.key_handlers[key]()

    def _handle_quit(self):
        """Обработчик выхода"""
        self.running = False

    def _handle_show_names(self):
        """Показать список известных лиц"""
        names = self.recognizer.get_known_faces()
        if names:
            names_str = "Знакомые: " + ", ".join(names[:10])
            if len(names) > 10:
                names_str += f" ... и еще {len(names) - 10}"
        else:
            names_str = "Нет знакомых лиц"
        self.recognizer.set_status(names_str)

    def _handle_reload(self):
        """Перезагрузить базу лиц"""
        self.recognizer.load_known_faces()

    def _handle_help(self):
        """Показать справку"""
        help_text = "q-выход n-список l-обновить s-скриншот f-полный экран"
        self.recognizer.set_status(help_text, duration=5)

    def _handle_screenshot(self):
        """Сохранить скриншот"""
        os.makedirs(self.screenshots_dir, exist_ok=True)
        if self.current_frame is not None:
            timestamp = time.strftime("%Y-%m-%d_%H-%M")
            filename = self.screenshots_dir + timestamp + ".jpg"
            cv2.imwrite(filename, self.current_frame)
            self.recognizer.set_status(f"[INFO] Скриншот сохранен: {filename}")

    def _handle_fullscreen(self):
        """Переключить полноэкранный режим"""
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN):
            cv2.setWindowProperty(
                self.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL
            )
        else:
            cv2.setWindowProperty(
                self.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )

class Controller:
    """Контроллер с управлением жизненным циклом"""

    def __init__(self):
        self.config = self._default_config()
        queue_size = self.config.get('queue_size', 10)
        self.frame_queue = queue.Queue(maxsize=queue_size)

        # Инициализация компонентов
        self.recognizer = FaceRecognizer()
        self.capture_thread = RTSPCaptureThread(
            self.config['rtsp_url'],
            self.frame_queue
        )
        self.display_thread = DisplayThread(
            self.recognizer,
            self.frame_queue
        )

        # Мониторинг
        self._stats_thread = None
        self._running = False

    @staticmethod
    def _default_config() -> Dict:
        """Конфигурация по умолчанию"""
        return {
            'rtsp_url': "rtsp://192.168.1.61:554",
            'queue_size': 10,
            'show_stats': False,
        }


    def start(self):
        """Запуск системы"""
        if self._running:
            logger.warning("Система уже запущена")
            return

        self._running = True

        # Запуск потоков
        self.capture_thread.start()
        time.sleep(0.5)  # Даем время на подключение
        self.display_thread.start()

        self.recognizer.set_status("[INFO] Система запущена")

        # Запуск мониторинга если нужно
        if self.config['show_stats']:
            self._start_stats_monitor()

    def stop(self):
        """Остановка системы"""
        logger.info("Остановка системы...")
        self._running = False

        # Остановка потоков в правильном порядке
        if self.config['show_stats']:
            self._stats_thread.join(timeout=2)
        self.display_thread.stop()
        self.capture_thread.stop()

        # Сохранение состояния
        self.recognizer.save_cache()

        logger.info("Система остановлена")

    def _start_stats_monitor(self):
        """Запуск мониторинга статистики"""
        def stats_loop():
            while self._running:
                time.sleep(5)
                stats = self.capture_thread.get_stats()
                logger.debug(
                    f"Статистика: FPS={stats['fps']:.1f}, "
                    f"Очередь={stats['queue_size']}, "
                    f"Переподключений={stats['reconnects']}"
                )

        self._stats_thread = threading.Thread(
            target=stats_loop,
            daemon=True,
            name="Stats"
        )
        self._stats_thread.start()

    def run(self):
        """Запуск и ожидание завершения"""
        self.start()

        try:
            while self._running and self.display_thread.running:
                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Программа остановлена пользователем")
        except Exception as e:
            logger.exception(f"Критическая ошибка: {e}")
        finally:
            self.stop()


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('face_recognition.log', encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    try:
        controller = Controller()
        controller.run()

    except Exception as ex:
        logger.critical(f"Критическая ошибка: {ex}")
        exit(1)
