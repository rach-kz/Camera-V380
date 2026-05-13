import cv2
import time
from face_recognizer import FaceRecognizer

def main():
    # URL RTSP потока
    rtsp_url = "rtsp://admin:123456@192.168.1.61:554/stream1?tcp"
    cap = None
    try:
        # Подключение к камере
        print("[INFO] Подключение к камере...")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            print("[ERROR] Не удалось подключиться к камере")
            return

        # Настройки потока
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cap.set(cv2.CAP_PROP_FPS, 20)

        # Инициализация распознавателя
        print("[INFO] Загрузка моделей распознавания...")
        try:
            face_recognizer = FaceRecognizer()
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            print("[INFO] Скачайте модели с https://github.com/opencv/opencv_zoo")
            return
        except Exception as e:
            print(f"[ERROR] Ошибка инициализации: {e}")
            return


        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("[WARNING] Ошибка получения кадра")
                time.sleep(0.1)
                continue

            # Обработка кадра
            small_frame = cv2.resize(frame, (640, 480))
            processed_frame = face_recognizer.process_frame(small_frame)

            #cv2.putText(processed_frame, info_txt , (10, 470),
            #            cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imshow("V380 Face Recognition ('q' - exit, 'a' - add face)", processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                name = input("Введите имя для нового лица: ").strip()
                if name:
                    face_recognizer.add_new_face(frame, name)
            elif key == ord('d'):
                name = input("Введите имя для удаления лица: ").strip()
                if name:
                    face_recognizer.remove_face(name)
            elif key == ord('n'):
                names_list=face_recognizer.get_known_faces()
                for i in range(len(names_list)):
                    print(i+1,'.', names_list[i])

    except KeyboardInterrupt:
        print("\n[INFO] Прерывание пользователем")
    except Exception as e:
        print(f"[ERROR] Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Завершение работы")

if __name__ == "__main__":
    main()