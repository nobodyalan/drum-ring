"""精细测量: 密集打击期间混音缓冲峰值的分布。"""
import sys
import time
import numpy as np

sys.path.insert(0, ".")

import apps.audio.low_latency_player as llp

SOUNDS = {
    1: "gesture_1_closed_hat.wav",
    2: "gesture_2_tom2.wav",
    3: "gesture_3_snare.wav",
    4: "gesture_4_tom1.wav",
    5: "gesture_6_crash.wav",
    6: "gesture_5_kick.wav",
}

p = llp.LowLatencyDrumPlayer()
for cid, name in SOUNDS.items():
    p.load_sound(cid, f"assets/drum_gesture/sounds/{name}")

peaks = []
duration = 3.0
rate = 12.0
end = time.perf_counter() + duration
hits = 0
next_hit = time.perf_counter()
while time.perf_counter() < end:
    if time.perf_counter() >= next_hit:
        p.play((hits % 4) + 1, 1.0)
        hits += 1
        next_hit = time.perf_counter() + (1.0 / rate)
    with p._lock:
        peaks.append(float(np.abs(p.buf).max()))
    time.sleep(0.002)

peaks = np.array(peaks)
print(f"hits={hits} 采样点={len(peaks)}")
print(f"峰值 min={peaks.min():.3f} mean={peaks.mean():.3f} max={peaks.max():.3f}")
print(f"超过 0.90 的比例: {(peaks > 0.90).mean()*100:.1f}%")
print(f"超过 0.95 的比例: {(peaks > 0.95).mean()*100:.1f}%")
print(f"超过 0.99 的比例: {(peaks > 0.99).mean()*100:.1f}%")
p.close()
