import cv2
import numpy as np
import serial
import time
import math
import socket
import os
import threading

# ---------------------------------------------------------
# 1. CAMERA MANAGER
# ---------------------------------------------------------
class CameraManager:
    def __init__(self, width=640, height=480):
        self.w = width
        self.h = height

        # Eski çalışan sender.py'daki temiz pipeline!
        gstreamer_pipeline = (
            f"libcamerasrc ! video/x-raw, width={self.w}, height={self.h}, framerate=30/1 ! "
            f"videoconvert ! appsink drop=true"
        )
        self.video_capture = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        
        if not self.video_capture.isOpened():
            print("KRİTİK HATA: Kamera açılamadı!")

        self.src = np.float32([
            [136, 236],   # 1. Sol Üst (Kamerada okuduğun gerçek değer)
            [522, 236],   # 2. Sağ Üst (Kamerada okuduğun gerçek değer)
            [1043, 480],  # 3. Sağ Alt (22.5 cm referansından hesaplandı)
            [-385, 480]   # 4. Sol Alt (22.5 cm referansından hesaplandı)
        ])

        self.dst = np.float32([
            [0, 0],             
            [self.w, 0],        
            [self.w, self.h],   
            [0, self.h]         
        ])

        self.warp_matrix = cv2.getPerspectiveTransform(self.src, self.dst)

    def process_frame(self):
        ret, frame = self.video_capture.read()
        if not ret: return None, None 

        debug_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        warped_binary = cv2.warpPerspective(binary, self.warp_matrix, (self.w, self.h))

        ignore_y = self.h // 4
        warped_binary[0:ignore_y, :] = 0

        return debug_frame, warped_binary


# ---------------------------------------------------------
# 2. LANE TRACKER 
# ---------------------------------------------------------
class LaneTracker:
    def __init__(self, px_to_cm_x=0.0656, px_to_cm_y=0.09375, mechanical_offset_y=32.8):
        self.left_fit = None
        self.right_fit = None
        self.px_to_cm_x = px_to_cm_x
        self.px_to_cm_y = px_to_cm_y
        self.mechanical_offset_y = mechanical_offset_y
        self.prev_left_base = None
        self.prev_right_base = None
        
        self.lookahead_y = 300
        self.lost_frames_count = 0
        self.max_lost_frames = 10
        self.search_window = 80
        self.proximity_limit = 150
        
    def get_lookahead_y(self):
        return self.lookahead_y

    def set_lookahead_y(self):
        # İki şeridi fuse eden (birleştiren) o mantığı buraya kurduk
        left_a = abs(self.left_fit[0]) if self.left_fit is not None else None
        right_a = abs(self.right_fit[0]) if self.right_fit is not None else None

        # Kavis hesaplama mantığı
        if left_a is not None and right_a is not None:
            curve = (left_a + right_a) / 2
        elif left_a is not None: curve = left_a
        elif right_a is not None: curve = right_a
        else: curve = 0.0 

        # Hedef belirleme ve Low-pass filtreleme
        target = np.interp(curve, [0.000, 0.002], [140, 400])
        self.lookahead_y = int(0.8 * self.lookahead_y + 0.2 * target)
        
        
    def detect_lanes(self, warped_binary):
        nonzero = warped_binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        histogram = np.sum(warped_binary[warped_binary.shape[0] // 2:, :], axis=0)
        active_x = np.where(histogram > 20)[0]

        found_lines_base_x = []
        if len(active_x) > 0:
            breaks = np.where(np.diff(active_x) > 50)[0]
            splits = np.split(active_x, breaks + 1)
            for split in splits:
                if len(split) > 0:
                    best_x = split[np.argmax(histogram[split])]
                    found_lines_base_x.append(best_x)

        current_left_base = None
        current_right_base = None
        midpoint = warped_binary.shape[1] // 2

        for line_x in found_lines_base_x:
            dist_to_left = abs(line_x - self.prev_left_base) if self.prev_left_base is not None else 9999
            dist_to_right = abs(line_x - self.prev_right_base) if self.prev_right_base is not None else 9999

            if dist_to_left < self.proximity_limit and dist_to_left < dist_to_right:
                current_left_base = line_x
            elif dist_to_right < self.proximity_limit and dist_to_right < dist_to_left:
                current_right_base = line_x
            else:
                if line_x < midpoint and current_left_base is None:
                    current_left_base = line_x
                elif line_x >= midpoint and current_right_base is None:
                    current_right_base = line_x

        left_lane_inds = []
        right_lane_inds = []

        if current_left_base is not None:
            left_lane_inds = ((nonzerox > current_left_base - self.search_window) &
                              (nonzerox < current_left_base + self.search_window)).nonzero()[0]
        if current_right_base is not None:
            right_lane_inds = ((nonzerox > current_right_base - self.search_window) &
                               (nonzerox < current_right_base + self.search_window)).nonzero()[0]

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        lines_found_this_frame = 0

        if len(leftx) > 25:
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.prev_left_base = current_left_base
            lines_found_this_frame += 1
        elif len(leftx) > 10:
            linear_fit = np.polyfit(lefty, leftx, 1)
            self.left_fit = [0.0, linear_fit[0], linear_fit[1]]
            self.prev_left_base = current_left_base
            lines_found_this_frame += 1
        else:
            self.left_fit = None

        if len(rightx) > 25:
            self.right_fit = np.polyfit(righty, rightx, 2)
            self.prev_right_base = current_right_base
            lines_found_this_frame += 1
        elif len(rightx) > 10:
            linear_fit = np.polyfit(righty, rightx, 1)
            self.right_fit = [0.0, linear_fit[0], linear_fit[1]]
            self.prev_right_base = current_right_base
            lines_found_this_frame += 1
        else:
            self.right_fit = None

        if lines_found_this_frame == 0:
            self.lost_frames_count += 1
            if self.lost_frames_count >= self.max_lost_frames:
                self.prev_left_base = None
                self.prev_right_base = None
        else:
            self.lost_frames_count = 0

    def get_target_carrot(self, frame_width=640, frame_height=480, action="KEEP_LANE"):
        current_left_x = None
        current_right_x = None

        if self.left_fit is not None:
            current_left_x = self.left_fit[0] * self.lookahead_y ** 2 + self.left_fit[1] * self.lookahead_y + self.left_fit[2]
        if self.right_fit is not None:
            current_right_x = self.right_fit[0] * self.lookahead_y ** 2 + self.right_fit[1] * self.lookahead_y + self.right_fit[2]

        left_x_cm = (current_left_x - (frame_width / 2.0)) * self.px_to_cm_x if current_left_x else None
        right_x_cm = (current_right_x - (frame_width / 2.0)) * self.px_to_cm_x if current_right_x else None

        lane_half_cm = 13.0
        target_x_cm = 0.0

        if action == "KEEP_LANE":
            if left_x_cm is not None and right_x_cm is not None:
                target_x_cm = (left_x_cm + right_x_cm) / 2.0
            elif left_x_cm is not None:
                target_x_cm = left_x_cm + lane_half_cm
            elif right_x_cm is not None:
                target_x_cm = right_x_cm - lane_half_cm
        elif action == "CHANGE_LEFT":
            if left_x_cm is not None:
                target_x_cm = left_x_cm - lane_half_cm
            elif right_x_cm is not None:
                target_x_cm = right_x_cm - 39.0
        elif action == "CHANGE_RIGHT":
            if right_x_cm is not None:
                target_x_cm = right_x_cm + lane_half_cm
            elif left_x_cm is not None:
                target_x_cm = left_x_cm + 39.0

        y_pixel_distance = frame_height - self.lookahead_y
        target_y_cm = (y_pixel_distance * self.px_to_cm_y) + self.mechanical_offset_y

        target_x_pixel = (target_x_cm / self.px_to_cm_x) + (frame_width / 2.0)
        return target_x_cm, target_y_cm, target_x_pixel


# ---------------------------------------------------------
# 3. PURE PURSUIT CONTROLLER 
# ---------------------------------------------------------
class PurePursuitController:
    def __init__(self, steering_gain=200.0, max_angular_vel=90.0):
        # base_speed_cmps yerine artık bir "kazanç" (gain) katsayımız var.
        # Bu değer arabanın virajlara ne kadar agresif gireceğini belirler.
        self.steering_gain = steering_gain
        
        # Güvenlik limiti: Arduino'ya gidecek maksimum derece/saniye sınırı
        self.max_angular_vel = max_angular_vel

    def compute_target_angular_velocity(self, target_x_cm, target_y_cm):
        l_squared = (target_x_cm ** 2) + (target_y_cm ** 2)
        # Hedef çok yakınsa veya sıfırsa sıfıra bölme hatasını engelle
        if l_squared < 0.1: 
            return 0.0
        # 1. Adım: Saf Takip Kavis (Curvature) Formülü
        # Kavis ne kadar büyükse, araç o kadar keskin dönmelidir.
        curvature = (2 * target_x_cm) / l_squared
        # 2. Adım: Hedef Açısal Hızı (rad/s) Hesapla
        # Kavis ile kazanç katsayımızı çarpıyoruz.
        omega_radps = self.steering_gain * curvature
        # 3. Adım: Radyanı Dereceye Çevir (Arduino'nun beklediği format)
        omega_degps = math.degrees(omega_radps)
        # 4. Adım: Çıkan değeri güvenli sınırlar (Limit) içine al
        # Çok keskin kamera hatalarında arabanın kendi etrafında fırıldak gibi dönmesini engeller
        omega_degps = max(min(omega_degps, self.max_angular_vel), -self.max_angular_vel)
        return omega_degps


# ---------------------------------------------------------
# 4. SERIAL COMMUNICATOR 
# ---------------------------------------------------------
class Communicator:
    def __init__(self, port_name="/dev/ttyUSB0", baud_rate=115200, udp_port=5001):
        # --- 1. ARDUINO (SERIAL) BAĞLANTISI ---
        self.port_name = port_name
        self.baud_rate = baud_rate
        self.is_connected = False
        # Seri porta hem ana döngü hem keep-alive thread'i yazıyor; yazımları
        # kilitleyerek paketlerin iç içe geçip bozulmasını engelliyoruz.
        self.write_lock = threading.Lock()
        try:
            self.serial_connection = serial.Serial(self.port_name, self.baud_rate, timeout=0.1)
            self.is_connected = True
            print(f"[BİLGİ] Arduino'ya {self.port_name} üzerinden bağlanıldı.")
            print("[BİLGİ] Arduino'nun uyanması için 2 saniye bekleniyor...")
            time.sleep(2.0)
        except Exception as e:
            print(f"[HATA] Serial Bağlantı Hatası: {e}")

        # --- 2. UBUNTU KONTROL (UDP) BAĞLANTISI ---
        self.udp_port = udp_port
        try:
            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_sock.bind(("0.0.0.0", self.udp_port))
            self.udp_sock.setblocking(False) # Bekleme yapmadan oku
            print(f"[BİLGİ] UDP Soketi {self.udp_port} portunda dinlemeye başladı.")
        except Exception as e:
            print(f"[HATA] UDP Soket Hatası: {e}")

    # --- ARDUINO KOMUTLARI ---
    def send_ang_vel(self, ang_vel):
        if self.is_connected:
            packet = f"L:{ang_vel:.1f}\n"
            with self.write_lock:
                self.serial_connection.write(packet.encode('utf-8'))

    def send_direct_command(self, cmd_str):
        """Arduino'ya doğrudan ILERI, GERI, DUR gibi metin komutları yollar."""
        if self.is_connected:
            packet = f"{cmd_str}\n"
            with self.write_lock:
                self.serial_connection.write(packet.encode('utf-8'))

    def read_telemetry(self):
        if self.is_connected and self.serial_connection.in_waiting > 0:
            return self.serial_connection.readline().decode('utf-8').strip()
        return None

    def send_emergency_stop(self):
        if self.is_connected:
            try:
                with self.write_lock:
                    self.serial_connection.write("DUR\n".encode('utf-8'))
                    self.serial_connection.flush() 
                time.sleep(0.2)
                self.serial_connection.close()
                self.is_connected = False
                print("[BİLGİ] Arduino'ya DUR komutu iletildi ve port kapatıldı.")
            except Exception as e:
                print(f"[HATA] Seri port kapatılırken sorun oluştu: {e}")

    # --- UBUNTU KOMUTLARI (YENİ) ---
    def read_udp_command(self):
        """Ağdaki tüm birikmiş eski paketleri silip sadece en son gelen komutu okur."""
        son_komut = None
        while True:
            try:
                data, _ = self.udp_sock.recvfrom(1024)
                son_komut = data.decode('utf-8')
            except BlockingIOError:
                break
        return son_komut


# ---------------------------------------------------------
# 5. MAIN ROVER SYSTEM 
# ---------------------------------------------------------
class Rover:
    def __init__(self):
        self.camera = CameraManager(width=640, height=480)
        self.tracker = LaneTracker(px_to_cm_x=0.0656, px_to_cm_y=0.09375, mechanical_offset_y=32.8)
        self.controller = PurePursuitController(steering_gain=200.0, max_angular_vel=90.0)
        self.communicator = Communicator(port_name="/dev/ttyUSB0", baud_rate=115200, udp_port=5001)

        self.system_state = "RUNNING"
        self.current_lane = "RIGHT"
        self.mode = "ON_ROAD"
        self.is_moving = False
        self.obstacle_count = 0
        # Engel algılama: Arduino'dan gelen ultrasonik mesafe (cm). ON_ROAD'da
        # obstacle_stop_cm içine girince araç durur (mod ON_ROAD'da kalır), engel
        # obstacle_clear_cm'in üstüne çıkınca otomatik devam eder. İki eşik
        # (histerezis) sınırda dur-kalk titremesini önler. 999 = okuma yok.
        self.last_distance = 999
        self.obstacle_stop_cm = 10
        self.obstacle_clear_cm = 15
        self.obstacle_blocked = False
        # Watchdog beslemesi: kamera bir kare düşürdüğünde son direksiyon
        # komutunu tekrar yollayıp Arduino'nun 200ms watchdog'unu canlı tutarız
        # (dur-kalk biter). Ama kamera max_blind_ms'den uzun süre kare vermezse
        # beslemeyi keseriz -> watchdog devreye girip güvenli durur.
        self.last_ang_vel = 0.0
        self.last_valid_frame_time = time.time()
        self.max_blind_ms = 400
        # --- GSTREAMER H264 YAYIN ---
        # Yayın kontrol döngüsünü yavaşlatıp Arduino'nun 200ms watchdog'unu
        # tetikleyebiliyor (git-dur git-dur). Bu yüzden:
        #   ENABLE_STREAM=0   -> yayını tamamen kapat (en akıcı sürüş)
        #   STREAM_EVERY=N    -> sadece her N. kareyi gönder (encode yükünü düşürür)
        #   VIDEO_TARGET_IP   -> yayının gideceği bilgisayarın IP'si
        target_ip = os.environ.get("VIDEO_TARGET_IP", "10.42.0.112")
        # Varsayılan KAPALI: video yayını döngüyü yavaşlatıp dur-kalk yapıyordu.
        # Sürüş için gerekmez. Video izlemek istersen ENABLE_STREAM=1 ile aç.
        self.enable_stream = os.environ.get("ENABLE_STREAM", "0") == "1"
        self.stream_every = max(1, int(os.environ.get("STREAM_EVERY", "3")))
        self._frame_idx = 0
        self.combined_w = self.camera.w * 2 
        self.combined_h = self.camera.h
        self.stream = None

        if self.enable_stream:
            pipe_out = (
                f"appsrc ! videoconvert ! video/x-raw,format=I420 ! "
                f"x264enc tune=zerolatency bitrate=1000 speed-preset=ultrafast ! "  
                f"rtph264pay config-interval=1 pt=96 ! "
                f"udpsink host={target_ip} port=5000 sync=false"
            )

            self.stream = cv2.VideoWriter(pipe_out, cv2.CAP_GSTREAMER, 0, 30.0, (self.combined_w, self.combined_h), True)

            if not self.stream.isOpened():
                print("KRİTİK HATA: VideoWriter (Yayıncı) başlatılamadı!")
            else:
                print(f"[BİLGİ] Video yayını açık -> {target_ip}:5000 (her {self.stream_every}. kare)")
        else:
            print("[BİLGİ] Video yayını KAPALI (ENABLE_STREAM=0). En akıcı sürüş modu.")

        # --- WATCHDOG KEEP-ALIVE THREAD ---
        # Kameranın read() çağrısı bazen kareyi geciktirip ana döngüyü 200ms'den
        # uzun bloklayabiliyor (otofokus/pipeline takılması). O sırada hiç komut
        # gidemediği için Arduino watchdog'u motoru kesip dur-kalk yapıyor. Bu
        # thread, kameradan BAĞIMSIZ olarak son direksiyon komutunu sürekli (~80ms)
        # gönderip watchdog'u canlı tutar. Kamera max_blind_ms'den uzun ölürse
        # gönderimi keser -> watchdog devreye girip güvenli durur.
        self._keepalive_stop = False
        self._keepalive_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._keepalive_thread.start()

    def _keepalive_loop(self):
        while not self._keepalive_stop:
            time.sleep(0.08)  # ~12 Hz; 200ms watchdog'un çok altında
            if self.mode == "ON_ROAD" and self.is_moving and not self.obstacle_blocked:
                blind_ms = (time.time() - self.last_valid_frame_time) * 1000
                if blind_ms < self.max_blind_ms:
                    self.communicator.send_ang_vel(self.last_ang_vel)

    # DÜZELTME: current_action parametresi eklendi
    def stream_to_ubuntu(self, debug_frame, warped_binary, target_x_pixel, ang_vel, current_action):
        pts = self.camera.src.astype(np.int32)
        cv2.polylines(debug_frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(debug_frame, f"AngVel: {ang_vel:.1f} Deg/s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if self.mode == "ON_ROAD":
            mode_color = (0, 255, 0)     # On-Road için Yeşil
            display_action = current_action
        else:
            mode_color = (0, 165, 255)   # Off-Road için Turuncu
            display_action = "MANUEL"

        cv2.putText(debug_frame, f"MOD: {self.mode}", (20, 80),cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
        cv2.putText(debug_frame, f"is_moving: {self.is_moving}", (20, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(debug_frame, f"Action: {display_action}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        warped_rgb = cv2.cvtColor(warped_binary, cv2.COLOR_GRAY2BGR)
        plot_y = np.linspace(0, self.camera.h - 1, self.camera.h)

        if self.tracker.left_fit is not None:
            left_fit_x = self.tracker.left_fit[0] * plot_y ** 2 + self.tracker.left_fit[1] * plot_y + self.tracker.left_fit[2]
            pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))], np.int32)
            cv2.polylines(warped_rgb, [pts_left], isClosed=False, color=(255, 0, 0), thickness=4)

        if self.tracker.right_fit is not None:
            right_fit_x = self.tracker.right_fit[0] * plot_y ** 2 + self.tracker.right_fit[1] * plot_y + self.tracker.right_fit[2]
            pts_right = np.array([np.transpose(np.vstack([right_fit_x, plot_y]))], np.int32)
            cv2.polylines(warped_rgb, [pts_right], isClosed=False, color=(0, 255, 0), thickness=4)

        cv2.line(warped_rgb, (0, self.tracker.get_lookahead_y()), (self.camera.w, self.tracker.get_lookahead_y()), (0, 255, 255), 1)
        if target_x_pixel is not None:
            cv2.circle(warped_rgb, (int(target_x_pixel), self.tracker.get_lookahead_y()), 10, (0, 0, 255), -1)

        combined_frame = np.hstack((debug_frame, warped_rgb))
        self.stream.write(combined_frame)

    def run_autonomous_loop(self):
        print(f"Sistem Başladı! Mod: {self.mode}. UDP komutları bekleniyor...")
        current_action = "KEEP_LANE"
        
        # OFF-ROAD için anlık durum takibi
        current_offroad_cmd = None

        try:
            while self.system_state == "RUNNING":
                telemetry = self.communicator.read_telemetry()
                # Arduino "M:<cm>" mesafe telemetrisi gönderiyor; engel kontrolü
                # için son geçerli mesafeyi sakla (999 = okuma yok).
                if telemetry and telemetry.startswith("M:"):
                    try:
                        self.last_distance = int(telemetry[2:])
                    except ValueError:
                        pass
                display_ang_vel = 0.0 

                cmd = self.communicator.read_udp_command()
                
                if cmd:
                    # Mutlak mod komutları (MODE_ONROAD / MODE_OFFROAD) tercih edilir;
                    # UDP paketi düşse bile client ile rover'ın modu senkron kalır.
                    # MODE_TOGGLE eski client'lar için geriye dönük uyumluluk.
                    if cmd in ("MODE_TOGGLE", "MODE_ONROAD", "MODE_OFFROAD"):
                        if cmd == "MODE_ONROAD":
                            self.mode = "ON_ROAD"
                        elif cmd == "MODE_OFFROAD":
                            self.mode = "OFF_ROAD"
                        else:  # MODE_TOGGLE
                            self.mode = "OFF_ROAD" if self.mode == "ON_ROAD" else "ON_ROAD"
                        self.is_moving = False  
                        current_offroad_cmd = None
                        print(f"\n[MOD DEĞİŞTİ] Yeni Mod: {self.mode}")

                    # --- ON_ROAD MODUNDA TETİKLEME ---
                    elif self.mode == "ON_ROAD":
                        if cmd == "CMD_W":
                            self.is_moving = True
                            self.communicator.send_direct_command("ILERI") 
                            print("[ON-ROAD] Şerit Takibi Başladı!")
                        elif cmd == "CMD_S":
                            self.is_moving = False
                            print("[ON-ROAD] Araç Duraklatıldı.")
                        elif cmd == "CMD_A" and self.is_moving:
                            current_action = "CHANGE_LEFT" 
                            print("[ON-ROAD] Sol Şeride Geçiş Tetiklendi!")
                        elif cmd == "CMD_D" and self.is_moving:
                            current_action = "CHANGE_RIGHT" 
                            print("[ON-ROAD] Sağ Şeride Geçiş Tetiklendi!")

                    # --- OFF_ROAD MODU DURUM GÜNCELLEMESİ ---
                    elif self.mode == "OFF_ROAD":
                        current_offroad_cmd = cmd # Son gelen komutu kaydet
                        # Timeout mekanizmasını Pi tarafına taşıyoruz
                        self.last_cmd_time = time.time()

                debug_frame, binary_map = self.camera.process_frame()
                
                if binary_map is None:
                    # Kamera bu turda kare vermedi. Kısa süreli düşüşse, hareket
                    # halindeyken son komutu tekrar yollayıp watchdog'u besle ki
                    # araç dur-kalk yapmasın. Uzun süre kare gelmezse (kamera
                    # ölmüş) gönderme -> Arduino 200ms watchdog'u ile güvenli dur.
                    blind_ms = (time.time() - self.last_valid_frame_time) * 1000
                    if self.mode == "ON_ROAD" and self.is_moving and not self.obstacle_blocked and blind_ms < self.max_blind_ms:
                        self.communicator.send_ang_vel(self.last_ang_vel)
                    time.sleep(0.01)
                    continue

                # Geçerli kare geldi: kör (blind) sayacını sıfırla.
                self.last_valid_frame_time = time.time()
                target_x_pixel = None

                # --- ON_ROAD SÜRÜŞ MANTIĞI ---
                if self.mode == "ON_ROAD" and self.is_moving:
                    # --- ENGEL DUR/DEVAM (histerezisli) ---
                    # Engel obstacle_stop_cm içine girince dur; obstacle_clear_cm'in
                    # ÜSTÜNE çıkınca devam et. İki farklı eşik (histerezis) sınırda
                    # titremeyi (dur-kalk) önler. Mod ON_ROAD'da kalır, kullanıcı
                    # tekrar komut vermez; engel kalkınca otomatik devam eder.
                    if 0 < self.last_distance <= self.obstacle_stop_cm:
                        if not self.obstacle_blocked:
                            print(f"[ENGEL] {self.last_distance}cm! Durdu, engel kalkinca devam edecek.")
                        self.obstacle_blocked = True
                    elif self.last_distance > self.obstacle_clear_cm:
                        if self.obstacle_blocked:
                            print("[ENGEL] Temizlendi, devam ediliyor.")
                            self.communicator.send_direct_command("ILERI")  # ileri latch'i tazele
                        self.obstacle_blocked = False

                    if self.obstacle_blocked:
                        # Dur ama ON_ROAD/is_moving korunur; engel kalkınca devam.
                        self.communicator.send_direct_command("DUR")
                        continue

                    if current_action == "CHANGE_LEFT":
                        if (self.tracker.prev_left_base is not None) and (self.tracker.prev_left_base > 320):
                            self.tracker.prev_right_base = self.tracker.prev_left_base
                            self.tracker.prev_left_base = None
                            self.current_lane = "LEFT"
                            current_action = "KEEP_LANE" 
                            print("[ON-ROAD] Sol Şeride Geçiş Tamamlandı!")

                    elif current_action == "CHANGE_RIGHT":
                        if (self.tracker.prev_right_base is not None) and (self.tracker.prev_right_base < 320):
                            self.tracker.prev_left_base = self.tracker.prev_right_base
                            self.tracker.prev_right_base = None
                            self.current_lane = "RIGHT"
                            current_action = "KEEP_LANE" 
                            print("[ON-ROAD] Sağ Şeride Geçiş Tamamlandı!")

                    self.tracker.detect_lanes(binary_map)
                    self.tracker.set_lookahead_y()
                    target_x_cm, target_y_cm, target_x_pixel = self.tracker.get_target_carrot(frame_width=self.camera.w, frame_height=self.camera.h, action=current_action)
                    
                    display_ang_vel = self.controller.compute_target_angular_velocity(target_x_cm, target_y_cm)
                    self.last_ang_vel = display_ang_vel
                    self.communicator.send_ang_vel(display_ang_vel)

                # --- OFF_ROAD SÜRÜŞ MANTIĞI ---
                elif self.mode == "OFF_ROAD":
                    # EĞER SON 0.3 SANİYEDİR TUŞA BASILMIYORSA ARACI DURDUR
                    if hasattr(self, 'last_cmd_time') and (time.time() - self.last_cmd_time > 0.3):
                        current_offroad_cmd = None
                        
                    if current_offroad_cmd == "CMD_W":
                        self.communicator.send_direct_command("ILERI")
                        self.communicator.send_ang_vel(0) 
                        display_ang_vel = 0.0
                    elif current_offroad_cmd == "CMD_S":
                        self.communicator.send_direct_command("GERI")
                        self.communicator.send_ang_vel(0) 
                        display_ang_vel = 0.0
                    elif current_offroad_cmd == "CMD_A":
                        self.communicator.send_ang_vel(-60) 
                        display_ang_vel = -60.0  
                    elif current_offroad_cmd == "CMD_D":
                        self.communicator.send_ang_vel(60) 
                        display_ang_vel = 60.0

                # Yayını sadece açıksa ve her N. karede gönder; encode yükü
                # kontrol döngüsünü yavaşlatıp watchdog'u tetiklemesin diye.
                self._frame_idx += 1
                if self.stream is not None and (self._frame_idx % self.stream_every == 0):
                    self.stream_to_ubuntu(debug_frame, binary_map, target_x_pixel, display_ang_vel, current_action)

        except KeyboardInterrupt:
            self.emergency_stop()
            
        except Exception as e:
            print(f"\n[KRİTİK HATA] Beklenmeyen hata oluştu: {e}")
            self.emergency_stop()

    def emergency_stop(self):
        print("\n[BİLGİ] Acil durdurma tetiklendi! Motorlar kilitleniyor...")
        self._keepalive_stop = True  # keep-alive thread'i durdur, yoksa DUR'u ezer
        time.sleep(0.12)             # son keep-alive turunun bitmesini bekle
        self.communicator.send_emergency_stop()
        self.system_state = "STOPPED"
        
        if hasattr(self, 'camera') and self.camera.video_capture:
            self.camera.video_capture.release()
        if hasattr(self, 'stream') and self.stream is not None:
            self.stream.release()
            
        print("[BİLGİ] Sistem Kapatılıyor... Tamamlandı.")

if __name__ == "__main__":
    my_rover = Rover()
    my_rover.run_autonomous_loop()
