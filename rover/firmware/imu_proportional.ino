#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

// --- PIN TANIMLAMALARI ---
const int ENA = 5, IN1 = 6, IN2 = 7;
const int ENB = 10, IN3 = 8, IN4 = 9;
const int trigPin = 2; // Ultrasonik Trig
const int echoPin = 3; // Ultrasonik Echo

// --- DEĞİŞKENLER ---
int tabanHiz = 100;
float hedefDonusHizi_DegS = 0.0;
const float Kp = 1.5; 
float gyroZ_ofset = 0;
unsigned long sonKomutZamani = 0;
unsigned long sonMesafeZamani = 0; // Mesafe gönderimi için zamanlayıcı

// YENİ: Vites yönünü hafızada tutacak değişken
bool ileriYonde = true; 

void setup() {
  Serial.begin(115200);

  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  // MPU gürültü kalkanı (Motorlar dönerken sensör kilitlenmesini önler)
  Wire.begin();
  Wire.setWireTimeout(25000, true); 

  if (!mpu.begin()) {
    while (1) { delay(10); }
  }
  
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  // Kalibrasyon
  float toplam = 0;
  for (int i = 0; i < 200; i++) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    toplam += g.gyro.z;
    delay(5);
  }
  gyroZ_ofset = toplam / 200.0;
  
  // Zıplama ve sonsuz döngü tuzakları buradan tamamen temizlendi!
  // Araç doğrudan loop'a inip güvenli bir şekilde uyku modunda komut bekleyecek.
}

void loop() {
  // 1. KOMUT OKUMA
  if (Serial.available() > 0) {
    String komut = Serial.readStringUntil('\n');
    komut.trim(); 
    
    if (komut == "DUR") {
      motorlariSur(0, 0);
    }
    // YENİ: İleri ve Geri vites komutları
    else if (komut == "ILERI") {
      ileriYonde = true;
      sonKomutZamani = millis();
    }
    else if (komut == "GERI") {
      ileriYonde = false;
      sonKomutZamani = millis();
    }
    else if (komut.startsWith("L:")) {
      hedefDonusHizi_DegS = komut.substring(2).toFloat();
      sonKomutZamani = millis();
    }
  }

  // 2. ULTRASONİK SENSÖR OKUMA VE Pİ'YE GÖNDERME
  if (millis() - sonMesafeZamani > 100) {
    long duration;
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    
    // Timeout'u 10ms (10000us) yaptık. En fazla ~1.7m uzağı bekler.
    duration = pulseIn(echoPin, HIGH, 10000); 
    
    int mesafe;
    if (duration == 0) {
      mesafe = 999; 
    } else {
      mesafe = (duration * 0.034) / 2;
    }
    
    Serial.print("M:");
    Serial.println(mesafe);
    
    sonMesafeZamani = millis();
  }

  // 3. GÜVENLİK VE PID SÜRÜŞÜ 
  // YENİ: Timeout süresi 500ms'den 200ms'ye düşürüldü
  if (millis() - sonKomutZamani > 200) {
    motorlariSur(0, 0);
  } 
  else {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    float gercekDonusHizi_DegS = (g.gyro.z - gyroZ_ofset) * 57.2958;

    int aciSiddeti = (int)abs(hedefDonusHizi_DegS); 
    if (aciSiddeti <= 10) tabanHiz = map(aciSiddeti, 0, 10, 100, 80); 
    else if (aciSiddeti <= 30) tabanHiz = map(aciSiddeti, 10, 30, 80, 60); 
    else tabanHiz = 60;

    float hata = hedefDonusHizi_DegS - gercekDonusHizi_DegS;
    float duzeltme = Kp * hata;
    duzeltme = constrain(duzeltme, -40, 40);
    // YENİ: Kinematik Tersinme (Geri viteste hata yönünü ters çevir)
    if (!ileriYonde) {
      duzeltme = -duzeltme;
      tabanHiz += 20;
    }

    int solMotorHiz = constrain(tabanHiz + duzeltme, 0, 255); 
    int sagMotorHiz = constrain(tabanHiz - duzeltme, 0, 255);

    motorlariSur(solMotorHiz, sagMotorHiz);
  }
}

// YENİ: Geri vites destekli motor sürücü
void motorlariSur(int solHiz, int sagHiz) {
  if (solHiz == 0 && sagHiz == 0) {
    digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
    digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
    analogWrite(ENA, 0); analogWrite(ENB, 0);
    return;
  }
  
  if (ileriYonde) {
    // İleri sürüş pin konfigürasyonu
    digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW); analogWrite(ENA, sagHiz);
    digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW); analogWrite(ENB, solHiz);
  } else {
    // Geri sürüş (Pinler terslendi)
    digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH); analogWrite(ENA, sagHiz);
    digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH); analogWrite(ENB, solHiz);
  }
}
