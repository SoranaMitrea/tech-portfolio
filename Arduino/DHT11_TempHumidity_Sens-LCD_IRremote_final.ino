#include <DHT.h>
#include <LiquidCrystal.h>
#include <IRremote.hpp>

// ---- DHT11 Setup ----
#define DHTPIN 7
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// ---- LCD Setup (RS, E, D4, D5, D6, D7) ----
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

// ---- IR Setup ----
const int IR_SEND_PIN = 9; // IR LED über Transistor an D9
IRsend irsend(IR_SEND_PIN);

// ---- Klima Schwellenwerte ----
const float T_ON = 22.0; // Einschalten >= 22°C
const float T_OFF = 21.0; // Ausschalten unter <= 21°C
bool klimaStatus = false; // false = aus, true = an

// Sende-Parameter ---
const uint8_t REPEATS = 12; // 8-10 ist gut
const uint16_t GAP_MS = 100; // 80-120 ms Pause zwischen Frames
const uint8_t KHZ     = 36; // erst 38Hz, bei Bedarf 36/ 40 testen

// ---- hier sind RAW-Codes - Fernbedienung ----
// AC ON
uint16_t rawOn[] = {
  3030,1620, 480,1070, 480,1120, 430,370, 480,320, 480,370, 430,1120, 430,370, 480,220, 580,1120, 430,1120, 480,370, 480,1070, 480,320, 480,370, 430,1120, 480,1070, 480,320, 480,1120, 480,1120, 480,320, 430,370, 480,1120, 430,370, 430,370, 480,320, 480,1120, 430,370, 430,370, 480,320, 480,370, 430,370, 430,370, 480,320, 480,320, 480,370, 480,320, 480,320, 480,320, 480,370, 430,370, 480,320, 480,320, 530,320, 480,320, 480,320, 480,370, 430,1120, 480,320, 480,320, 480,370, 480,320, 480,320, 480,370, 430,370, 480,1070, 480,1120, 480,320, 480,320, 480,320, 480,1120, 480,1070, 480,320, 480,370, 480,1070, 480,1120, 480,1070, 480,370, 430,370, 430,370, 430,370, 480,370, 430,320, 480,370, 430,420, 430,370, 430,370, 430,370, 480,370, 430,370, 430,370, 430,370, 430,420, 380,420, 430,370, 430,370, 480,370, 380,420, 430,370, 430,370, 430,420, 380,420, 430,370, 430,370, 480,370, 430,370, 430,370, 430,370, 430,420, 380,420, 430,370, 430,370, 480,320, 430,420, 430,370, 430,370, 430,370, 430,420, 430,370, 430,370, 430,370, 430,1170, 430,1120, 430
};

// AC OFF
uint16_t rawOff[] = {
  3030,1620, 480,1070, 480,1120, 430,370, 430,270, 530,370, 480,1120, 430,370, 480,320, 480,1120, 480,1070, 480,320, 480,1120, 430,370, 480,320, 480,1120, 430,1120, 480,320, 480,1120, 480,1070, 480,320, 530,320, 430,1120, 480,370, 430,370, 480,320, 480,1070, 480,320, 530,320, 480,320, 480,320, 480,370, 480,320, 480,320, 480,320, 480,370, 480,320, 480,320, 480,320, 480,320, 530,320, 480,320, 480,320, 480,320, 530,320, 480,370, 430,370, 430,1120, 480,370, 430,370, 430,370, 430,370, 480,370, 430,370, 430,370, 430,1170, 430,1120, 430,370, 430,370, 480,370, 430,1120, 430,1170, 380,420, 430,370, 430,1170, 380,1170, 430,1120, 480,370, 380,420, 430,370, 430,370, 480,370, 430,370, 430,370, 430,370, 430,420, 430,370, 430,370, 430,370, 480,370, 430,370, 430,370, 430,370, 480,370, 380,420, 430,370, 430,370, 480,370, 380,420, 430,370, 430,370, 480,370, 430,370, 430,370, 430,370, 430,420, 380,420, 430,370, 430,420, 430,370, 380,420, 430,370, 430,370, 480,370, 430,370, 430,370, 430,370, 430,420, 430,370, 430,370, 430,370, 480,1120, 430,1120, 430
};

// Helper: Frame mehrmals senden ---
void sendRawRepeat(uint16_t *data, size_t len, uint8_t khz, uint8_t reps) {
  for (uint8_t r = 0; r < reps; r++){
    IrSender.sendRaw(data, len, khz);
    delay(GAP_MS);
  }
}

void setup() {
  Serial.begin(115200);

  // LCD + DHT starten
  lcd.begin(16, 2);
  dht.begin();

  // IR starten
  IrSender.begin(IR_SEND_PIN, DISABLE_LED_FEEDBACK);

  lcd.print("Temp/Humidity");
  delay(2000);
  lcd.clear();
}

void loop() {
  float h = dht.readHumidity();
  float t = dht.readTemperature();

  if (isnan(h) || isnan(t)) {
    lcd.setCursor(0, 0);
    lcd.print("Sensor error ");
    delay(2000);
    return;
  }

  // Temperatur + Luftfeuchtigkeit anzeigen
  lcd.setCursor(0, 0);
  lcd.print("Temp: ");
  lcd.print(t);
  lcd.print((char)223); // Grad-Symbol
  lcd.print("C");

  lcd.setCursor(0, 1);
  lcd.print("Hum: ");
  lcd.print(h);
  lcd.print("% ");

  // ---- Klima-Logik ----
  if (!klimaStatus && t >= T_ON) {
    Serial.println("Klima EIN senden...");
    irsend.sendRaw(rawOn, sizeof(rawOn) / sizeof(rawOn[0]), 38);
    klimaStatus = true;
  }
  else if (klimaStatus && t <= T_OFF) {
    Serial.println("Klima AUS senden...");
    irsend.sendRaw(rawOff, sizeof(rawOff) / sizeof(rawOff[0]), 38);
    klimaStatus = false;
  }

  delay(3000);
}