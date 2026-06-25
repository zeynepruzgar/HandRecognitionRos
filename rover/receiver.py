import cv2
import socket
import os

# --- 1. AĞ AYARLARI (UDP KOMUT GÖNDERİCİ) ---
# Rover'ın (Raspberry Pi) IP Adresi. Sabit gömmek yerine env'den okunuyor:
#   PI_IP=10.42.0.1 python3 receiver.py
PI_IP = os.environ.get("PI_IP", "10.42.0.1")
UDP_PORT = 5001        # Rover'ın komut dinlediği port
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def komut_gonder(cmd):
    """Yakalanan tuşları anında ağ üzerinden Pi'ye fırlatır."""
    try:
        sock.sendto(cmd.encode('utf-8'), (PI_IP, UDP_PORT))
    except Exception as e:
        print(f"UDP Gönderme Hatası: {e}")

# --- 2. GSTREAMER YAYIN ALICISI ---
pipeline = (
    "udpsrc port=5000 ! "
    "application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96 ! "
    "rtph264depay ! avdec_h264 ! videoconvert ! "
    "video/x-raw,format=BGR ! " 
    "appsink drop=true max-buffers=1"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

print("Alıcı hazır, 5000 portundan rover bekleniyor...")
print("-" * 50)
print("KONTROLLER AKTİF:")
print("[W] İleri / Otonom Başlat")
print("[S] Geri  / Otonom Durdur")
print("[A] Sola Dön / Sol Şeride Geç")
print("[D] Sağa Dön / Sağ Şeride Geç")
print("[M] Sürüş Modunu Değiştir (On-Road <-> Off-Road)")
print("[ESC] veya [Q] Çıkış")
print("-" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        # Bağlantı gelene kadar veya koptuğunda hata fırlatmak yerine bekler
        cv2.waitKey(100) 
        continue
        
    cv2.imshow("Canli Yayin - Rover", frame)
    
    # --- 3. KLAVYE DİNLEYİCİSİ VE KONTROLCÜ ---
    # Odak "Canli Yayin - Rover" penceresindeyken çalışır
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27 or key == ord('q'): # ESC veya Q
        break
    elif key == ord('m'):
        komut_gonder("MODE_TOGGLE")
        print("Komut Gönderildi: MODE_TOGGLE")
    elif key == ord('w'):
        komut_gonder("CMD_W")
    elif key == ord('s'):
        komut_gonder("CMD_S")
    elif key == ord('a'):
        komut_gonder("CMD_A")
    elif key == ord('d'):
        komut_gonder("CMD_D")

cap.release()
cv2.destroyAllWindows()
