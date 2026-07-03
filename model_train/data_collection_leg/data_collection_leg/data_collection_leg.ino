const int sensorPins[] = {A1};  // 只使用 A1
const int SENSOR_COUNT = 1;
const int ACTIVE_SENSOR_COUNT = 1; // 只激活 A1

const unsigned long sampleIntervalUs = 5000; // 200Hz

const int pressThresholds[SENSOR_COUNT] = {80};
const int lowReleaseFloors[SENSOR_COUNT] = {70};
const int releaseRiseDeltas[SENSOR_COUNT] = {25};

const int fallConfirmSamples = 2;
const int riseConfirmSamples = 2;
const unsigned long minPressMillis = 20;
const unsigned long minOppositeEdgeMillis = 80;

enum State {
  RELEASED,
  PRESSED
};

int prevVals[SENSOR_COUNT];
int minVals[SENSOR_COUNT];
int fallingConfirmCounts[SENSOR_COUNT];
int risingConfirmCounts[SENSOR_COUNT];
State states[SENSOR_COUNT];
unsigned long pressStartMs[SENSOR_COUNT];
unsigned long lastEdgeMs[SENSOR_COUNT];
unsigned long fallingCounters[SENSOR_COUNT];
unsigned long risingCounters[SENSOR_COUNT];

unsigned long lastSampleUs = 0;

void setup() {
  Serial.begin(115200);

  for (int i = 0; i < ACTIVE_SENSOR_COUNT; ++i) {
    int raw = analogRead(sensorPins[i]);
    prevVals[i] = raw;
    minVals[i] = raw;
    fallingConfirmCounts[i] = 0;
    risingConfirmCounts[i] = 0;
    pressStartMs[i] = 0;
    lastEdgeMs[i] = 0;
    fallingCounters[i] = 0;
    risingCounters[i] = 0;

    if (raw < pressThresholds[i]) {
      states[i] = PRESSED;
      pressStartMs[i] = millis();
    } else {
      states[i] = RELEASED;
    }
  }

  lastSampleUs = micros();
}

void loop() {

  unsigned long nowUs = micros();
  if (nowUs - lastSampleUs < sampleIntervalUs) return;
  lastSampleUs += sampleIntervalUs;

  unsigned long nowMs = millis();

  for (int i = 0; i < ACTIVE_SENSOR_COUNT; ++i) {
    int raw = analogRead(sensorPins[i]);
    int prev = prevVals[i];
    int delta = raw - prev; // 正数=回升，负数=下降
    prevVals[i] = raw;

    switch (states[i]) {
      case RELEASED:
        if (raw < pressThresholds[i]) {
          fallingConfirmCounts[i]++;
        } else {
          fallingConfirmCounts[i] = 0;
        }

        if (
          fallingConfirmCounts[i] >= fallConfirmSamples &&
          nowMs - lastEdgeMs[i] >= minOppositeEdgeMillis
        ) {
          fallingCounters[i]++;
          lastEdgeMs[i] = nowMs;
          pressStartMs[i] = nowMs;
          minVals[i] = raw;
          fallingConfirmCounts[i] = 0;
          risingConfirmCounts[i] = 0;
          states[i] = PRESSED;

          printEdge(i, "FALLING", fallingCounters[i], raw, delta, minVals[i]);
        }
        break;

      case PRESSED:
        if (raw < minVals[i]) {
          minVals[i] = raw;
        }

        if (
          nowMs - pressStartMs[i] >= minPressMillis &&
          raw > lowReleaseFloors[i] &&
          raw >= minVals[i] + releaseRiseDeltas[i] &&
          delta > 0
        ) {
          risingConfirmCounts[i]++;
        } else {
          risingConfirmCounts[i] = 0;
        }

        if (
          risingConfirmCounts[i] >= riseConfirmSamples &&
          nowMs - lastEdgeMs[i] >= minOppositeEdgeMillis
        ) {
          risingCounters[i]++;
          lastEdgeMs[i] = nowMs;
          fallingConfirmCounts[i] = 0;
          risingConfirmCounts[i] = 0;
          states[i] = RELEASED;

          printEdge(i, "RISING", risingCounters[i], raw, delta, minVals[i]);
        }
        break;
    }
  }
}

void printEdge(
  int sensorIndex,
  const char *edgeName,
  unsigned long count,
  int raw,
  int delta,
  int minRaw
) {
  Serial.print("Sensor ");
  Serial.print(sensorIndex);
  Serial.print(" ");
  Serial.print(edgeName);
  Serial.print(" count=");
  Serial.print(count);
  Serial.print(" raw=");
  Serial.print(raw);
  Serial.print(" delta=");
  Serial.print(delta);
  Serial.print(" min=");
  Serial.println(minRaw);
}
