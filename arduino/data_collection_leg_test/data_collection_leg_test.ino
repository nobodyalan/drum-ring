const int sensorPins[] = {A1, A0};
const int SENSOR_COUNT = 2;

const unsigned long sampleIntervalUs = 5000; // 200Hz

unsigned long lastSampleUs = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for native USB boards; harmless on Uno/Nano.
  }

  lastSampleUs = micros();
  Serial.println("time_ms,A1,A0");
}

void loop() {
  unsigned long nowUs = micros();
  if (nowUs - lastSampleUs < sampleIntervalUs) return;
  lastSampleUs += sampleIntervalUs;

  // Serial.print(millis());
  for (int i = 0; i < SENSOR_COUNT; ++i) {
    Serial.print(",");
    Serial.print(analogRead(sensorPins[i]));
  }
  Serial.println();
}
