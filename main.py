import pyaudio
import wave
from ultralytics import YOLO
import keyboard

# Hàm để phát âm thanh cảnh báo
def play_notification_sound():
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
        stream.write(data)
        data = wf.readframes(CHUNK)
    stream.stop_stream()
    stream.close()
    p.terminate()

# Hàm để kiểm tra xem phím tắt đã được nhấn hay chưa
def check_shortcut():
    return keyboard.is_pressed('q')

# Khởi tạo model YOLO
model = YOLO('best.pt')

try:
    while True:
        # Phát hiện đối tượng và hiển thị video thời gian thực
        result = model.predict(source=0, imgsz=640, conf=0.6, show=True)

        # Nếu có đối tượng được phát hiện, phát âm thanh cảnh báo
        if result == "detections":
            play_notification_sound()
        
        # Kiểm tra nếu phím tắt q đã được nhấn
        if check_shortcut():
            break  # Thoát vòng lặp nếu phím tắt đã được nhấn

except KeyboardInterrupt:
    pass
