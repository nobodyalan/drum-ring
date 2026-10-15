"""压力测试: 正确统计回调耗时与欠载 (子类覆盖 _callback)。"""
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


class P(llp.LowLatencyDrumPlayer):
    def __init__(self, *a, **k):
        self.stats = {"calls": 0, "max_cb_ms": 0.0, "under": 0, "over": 0, "long": 0}
        super().__init__(*a, **k)

    def _callback(self, outdata, frames, ti, status):
        self.stats["calls"] += 1
        if status:
            if getattr(status, "output_underflow", False):
                self.stats["under"] += 1
            if getattr(status, "output_overflow", False):
                self.stats["over"] += 1
        t = time.perf_counter()
        super()._callback(outdata, frames, ti, status)
        dt = (time.perf_counter() - t) * 1e3
        self.stats["max_cb_ms"] = max(self.stats["max_cb_ms"], dt)
        if dt > 5.3:  # 超过一个 block 时长
            self.stats["long"] += 1


p = P()
for cid, name in SOUNDS.items():
    p.load_sound(cid, f"assets/drum_gesture/sounds/{name}")

time.sleep(0.2)

duration = 5.0
rate = 12.0
end = time.perf_counter() + duration
hits = 0
next_hit = time.perf_counter()
play_cost = 0.0
n_play = 0
peak_during = 0.0
last_peak_sample = time.perf_counter()
while time.perf_counter() < end:
    if time.perf_counter() >= next_hit:
        t = time.perf_counter()
        p.play((hits % 4) + 1, 1.0)
        play_cost += time.perf_counter() - t
        n_play += 1
        hits += 1
        next_hit = time.perf_counter() + (1.0 / rate)
    # 播放期间持续采样缓冲峰值
    if time.perf_counter() - last_peak_sample > 0.05:
        with p._lock:
            peak_during = max(peak_during, float(np.abs(p.buf).max()))
        last_peak_sample = time.perf_counter()
    time.sleep(0.0003)

time.sleep(0.3)
with p._lock:
    peak = float(np.abs(p.buf).max())

print(f"hits={hits}")
print(f"回调次数={p.stats['calls']} (5s 应≈{int(48000/256*5)})")
print(f"回调最长耗时={p.stats['max_cb_ms']:.3f} ms")
print(f"回调超时(>5.3ms)次数={p.stats['long']}")
print(f"输出欠载(underflow)次数={p.stats['under']}")
print(f"输出溢出(overflow)次数={p.stats['over']}")
print(f"play() 平均耗时={play_cost / max(1, n_play) * 1e6:.1f} us")
print(f"播放期间混音缓冲峰值={peak_during:.3f} (应≤0.9)")
print(f"结束静止后缓冲峰值={peak:.3f}")
p.close()
