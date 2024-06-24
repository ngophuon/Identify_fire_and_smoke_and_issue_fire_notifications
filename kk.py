from ultralytics import YOLO
import cv2
import math
import pyaudio
import wave
import threading
import cvzone
import time

# Hàm để phát âm thanh cảnh báo
def play_notification_sound():
    global notification_playing
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    p = pyaudio.PyAudio()
    wf = wave.open("baodong.wav", 'rb')
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(CHUNK)
    while data:
        if not notification_playing:
            break  # Dừng nếu cảnh báo đã ngừng
        stream.write(data)
        data = wf.readframes(CHUNK)
    stream.stop_stream()
    stream.close()
    p.terminate()


# Hàm để dừng phát âm thanh cảnh báoq
def stop_notification_sound():
    global notification_playing
    notification_playing = False


# Hàm để nhận diện và báo động chạy song song
def detect_and_alert(cap, model):
    global notification_playing
    is_fire_smoke_detected = False  # Biến để theo dõi xem có lửa hoặc khói được phát hiện không
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Kết thúc nếu không còn khung hình nào được đọc được nữa

        frame = cv2.resize(frame, (640, 480))
        result = model(frame, stream=True)

        fire_smoke_detected_this_iteration = False  # Biến để theo dõi xem có lửa hoặc khói được phát hiện trong khung hình này không

        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 50:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                       scale=1.5, thickness=2)

                    fire_smoke_detected_this_iteration = True  # Đặt cờ báo hiệu rằng đã phát hiện được lửa hoặc khói

        # Kiểm tra xem có phát hiện lửa hoặc khói không
        if fire_smoke_detected_this_iteration:
            is_fire_smoke_detected = True
            # Chỉ phát cảnh báo nếu cảnh báo hiện tại đã kết thúc
            if not notification_playing:
                notification_playing = True
                alert_thread = threading.Thread(target=play_notification_sound)
                alert_thread.start()
        else:
            is_fire_smoke_detected = False
            stop_notification_sound()

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Nếu không còn lửa hoặc khói được phát hiện, tắt âm thanh cảnh báo
    if not is_fire_smoke_detected:
        stop_notification_sound()

# Khởi tạo webcam
cap = cv2.VideoCapture(0)  # Sử dụng webcam, 0 là số ID của webcam, nếu có nhiều webcam, bạn có thể thử các số khác nhau

# Khởi tạo mô hình YOLO
model = YOLO('best.pt')

# Đọc các lớp
classnames = ['FIRE', 'smoqqke']

# Biến để theo dõi trạng thái của âm thanh cảnh báo
notification_playing = False

# Khởi động một luồng riêng để nhận diện và báo động chạy song song
detection_thread = threading.Thread(target=detect_and_alert, args=(cap, model))
detection_thread.start()

# Chờ cho đến khi luồng nhận diện và báo động kết thúc trước khi giải phóng tài nguyên
detection_thread.join()

# Giải phóng tài nguyên và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()