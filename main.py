import os
import threading
import queue
import time
os.environ["OPENCV_FFMPEG_DEBUG"] = "-1"
import cv2
from face_recognizer import FaceRecognizer

class RTSPCaptureThread:
    """Захвата RTSP потока"""

    def __init__(self, rtsp_url: str, frame_queue: queue.Queue):
        self.rtsp_url = rtsp_url
        self.frame_queue = frame_queue
        self.running = False
        self.thread = None
        self.reconnect_delay = 1
        self.frame_count = 0

    def start(self):
        """Запуск потока захвата"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()


    def stop(self):
        """Остановка потока"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)


    def _capture_loop(self):
        """Основной цикл захвата с автоматическим переподключением"""
        while self.running:
            cap = None
            try:
                # Подключение к RTSP потоку
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

                if not cap.isOpened():
                    time.sleep(self.reconnect_delay)
                    continue

                # Настройки для RTSP
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Минимальный буфер
                cap.set(cv2.CAP_PROP_FPS, 20)

                self.reconnect_delay = 1  # Сброс задержки после успешного подключения

                # Цикл чтения кадров
                while self.running:
                    ret, frame = cap.read()

                    if not ret or frame is None:
                        break

                    self.frame_count = (self.frame_count + 1) % 1000000

                    # Управление очередью
                    if self.frame_queue.qsize() > 10:
                        # Очередь переполнена - удаляем старые кадры
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass

                    # Отправляем кадр в очередь
                    self.frame_queue.put(frame)

                    # Небольшая задержка для снижения нагрузки
                    time.sleep(0.001)

            except Exception:
                pass

            finally:
                if cap:
                    cap.release()

            if self.running:
                time.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, 30)  # Экспоненциальная задержка


class DisplayThread:
    """Поток для отображения и управления"""

    def __init__(self, recognizer: FaceRecognizer, frame_queue: queue.Queue):
        self.recognizer = recognizer
        self.frame_queue = frame_queue
        self.window_name = "V380 Face Recognition"
        self.running = False
        self.thread = None
        self.current_frame = None
        self.processed_frames = 0
        self.skip_frames = 0

    def start(self):
        """Запуск потока отображения"""
        self.running = True
        self.thread = threading.Thread(target=self._display_loop, daemon=True)
        self.thread.start()


    def stop(self):
        """Остановка потока"""
        self.running = False
        time.sleep(1)
        if self.thread:
            self.thread.join(timeout=1)
        cv2.destroyAllWindows()

    def get_current_frame(self):
        """Получить текущий кадр (для внешнего доступа)"""
        return self.current_frame

    def _display_loop(self):
        """Основной цикл отображения"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while self.running:
            try:
                # Берем кадр из очереди
                frame = self.frame_queue.get(timeout=0.5)
                self.current_frame = frame.copy()
                # Пропускаем каждый 2-й кадр для снижения нагрузки
                self.skip_frames += 1
                if self.skip_frames % 2 == 0:
                    if frame is not None:
                        cv2.imshow(self.window_name, frame)
                    continue

                # Обработка кадра
                processed_frame = self.recognizer.process_frame(frame)
                self.processed_frames = (self.processed_frames + 1) % 1000000

                if processed_frame is not None:
                    cv2.imshow(self.window_name, processed_frame)

                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    self.running = False
                    break

                elif key == ord('n'):
                    names = self.recognizer.get_known_faces()
                    str_names = "Знакомы: "
                    for i, name in enumerate(names, 1):
                        str_names +=f"{i}. {name} "
                    self.recognizer.set_status(str_names)

                elif key == ord('l'):
                    self.recognizer.load_known_faces()
                    self.recognizer.set_status("[INFO] Кэш знакомых лиц обновлен")

                elif key == ord('h'):
                    self.recognizer.set_status("Управление: q-выход, h-справка, n-список, l-обновить кэш")

            except queue.Empty:
                continue
            except Exception as e:
                self.recognizer.set_status(f"[ERROR] Ошибка отображения: {e}")


class MainController:
    """Контроллер для управления потоками"""

    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.frame_queue = queue.Queue(maxsize=10)

        # Инициализация распознавателя
        try:
            self.recognizer = FaceRecognizer()
        except Exception as e:
            self.recognizer.set_status(f"[ERROR] Ошибка инициализации распознавателя: {e}")
            raise

        # Создание потоков
        self.capture_thread = RTSPCaptureThread(rtsp_url, self.frame_queue)
        self.display_thread = DisplayThread(self.recognizer, self.frame_queue)

    def start(self):
        """Запуск всех потоков"""
        #print("[INFO] Запуск системы распознавания...")
        self.capture_thread.start()
        time.sleep(1)  # Даем время на подключение
        self.display_thread.start()
        self.recognizer.set_status("[INFO] Система запущена")

    def stop(self):
        """Остановка всех потоков"""
        self.display_thread.stop()
        self.capture_thread.stop()

    def run(self):
        """Запуск и ожидание завершения"""
        self.start()

        try:
            # Ждем завершения (поток отображения завершится при нажатии q)
            while self.display_thread.running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main():
    # Ваш URL
    url = "rtsp://192.168.1.61:554/stream1?tcp"
    try:
        controller = MainController(url)
        controller.run()
    except Exception as e:
        print(f"[FATAL] Критическая ошибка: {e}")
        return

if __name__ == "__main__":
    main()