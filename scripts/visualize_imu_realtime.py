"""
Real-time IMU visualization — ring raw acc / linear acc / gyro
===============================================================
Connects to the stream hub (TCP JSON-line), subscribes to IMU data,
computes online linear acceleration, and renders 3 clear subplots
with matplotlib FuncAnimation (same approach as latency_compare.py).

Usage:
  python scripts/visualize_imu_realtime.py
  python scripts/visualize_imu_realtime.py --host 127.0.0.1 --port 17888
  python scripts/visualize_imu_realtime.py --history 6.0   # show last 6 seconds
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.drum_gesture_recognizer import OnlineLinearAccelerationEstimator

# ── defaults ──
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 17888
HISTORY_SECONDS = 6.0
PLOT_FIGSIZE = (16, 10)
PLOT_INTERVAL_MS = 40  # ~25 fps

# ── axis colors (matching project palette) ──
COLOR_X = "#E53E3E"  # red
COLOR_Y = "#F59E0B"  # amber
COLOR_Z = "#10B981"  # emerald

# ═══════════════════════════════════════════════════════════════════
# Ring buffer
# ═══════════════════════════════════════════════════════════════════


class IMURingBuffer:
    """Thread-safe ring buffer for IMU time series."""

    def __init__(self, capacity: int = 12000):
        self.capacity = capacity
        self.timestamps = np.zeros(capacity, dtype=np.float64)
        self.raw_acc = np.zeros((capacity, 3), dtype=np.float32)
        self.raw_gyro = np.zeros((capacity, 3), dtype=np.float32)
        self.lin_acc = np.zeros((capacity, 3), dtype=np.float32)
        self.write_idx = 0
        self.count = 0
        self.lock = threading.Lock()

    def append(self, t: float, acc: np.ndarray, gyr: np.ndarray, lin: np.ndarray) -> None:
        with self.lock:
            idx = self.write_idx
            self.timestamps[idx] = t
            self.raw_acc[idx] = acc
            self.raw_gyro[idx] = gyr
            self.lin_acc[idx] = lin
            self.write_idx = (idx + 1) % self.capacity
            self.count = min(self.count + 1, self.capacity)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with self.lock:
            if self.count == 0:
                return (np.empty(0),) * 4
            if self.count < self.capacity:
                ts = self.timestamps[:self.count].copy()
                ra = self.raw_acc[:self.count].copy()
                rg = self.raw_gyro[:self.count].copy()
                la = self.lin_acc[:self.count].copy()
            else:
                w = self.write_idx
                ts = np.concatenate([self.timestamps[w:], self.timestamps[:w]])
                ra = np.concatenate([self.raw_acc[w:], self.raw_acc[:w]])
                rg = np.concatenate([self.raw_gyro[w:], self.raw_gyro[:w]])
                la = np.concatenate([self.lin_acc[w:], self.lin_acc[:w]])
            return ts, ra, rg, la


# ═══════════════════════════════════════════════════════════════════
# Stream hub client (plain TCP, no Qt dependency)
# ═══════════════════════════════════════════════════════════════════


class StreamHubClient:
    """Connect to JSON-line TCP stream hub, receive IMU, compute linear acc."""

    def __init__(self, host: str, port: int, buffer: IMURingBuffer):
        self.host = host
        self.port = port
        self.buffer = buffer
        self._sock: socket.socket | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_ts: float | None = None
        self._ts_is_ms: bool | None = None  # auto-detect ms vs s

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass

    def _run(self) -> None:
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(2.0)
            self._sock.connect((self.host, self.port))
            self._sock.settimeout(0.1)
        except OSError as exc:
            print(f"[StreamClient] connect failed: {exc}")
            return

        # Subscribe to IMU
        sub_msg = json.dumps({"cmd": "subscribe", "rings": [], "streams": ["imu"]}) + "\n"
        try:
            self._sock.sendall(sub_msg.encode())
        except OSError:
            return

        print(f"[StreamClient] connected to {self.host}:{self.port}, subscribed to IMU")

        buf = bytearray()
        estimator_per_address: dict[str, OnlineLinearAccelerationEstimator] = {}

        while self._running:
            try:
                chunk = self._sock.recv(65536)
                if not chunk:
                    break
            except socket.timeout:
                continue
            except OSError:
                break

            buf.extend(chunk)
            while b"\n" in buf:
                line_bytes, buf = buf.split(b"\n", 1)
                try:
                    msg = json.loads(line_bytes.decode("utf-8"))
                except Exception:
                    continue

                if msg.get("type") != "imu":
                    continue

                acc_list = msg.get("acc")
                gyr_list = msg.get("gyr")
                if not isinstance(acc_list, list) or not isinstance(gyr_list, list):
                    continue
                if len(acc_list) != 3 or len(gyr_list) != 3:
                    continue

                address = str(msg.get("address", ""))
                est = estimator_per_address.get(address)
                if est is None:
                    est = OnlineLinearAccelerationEstimator()
                    estimator_per_address[address] = est

                acc = np.asarray(acc_list, dtype=np.float32)
                gyr = np.asarray(gyr_list, dtype=np.float32)

                # Timestamp — auto-detect ms vs s (same logic as drum_gesture_app.py)
                ts_raw = None
                for key in ("local_timestamp", "timestamp", "host_timestamp"):
                    try:
                        ts_raw = float(msg.get(key, 0.0))
                    except (TypeError, ValueError):
                        continue
                    if ts_raw and ts_raw > 0:
                        break

                if ts_raw is None:
                    ts_raw = time.time()

                # Detect if timestamp is in milliseconds
                if self._ts_is_ms is None and self._last_ts is not None:
                    dt = ts_raw - self._last_ts
                    if dt > 1.0:
                        self._ts_is_ms = True
                        print(f"[StreamClient] timestamps appear to be in ms (dt={dt:.0f}), auto-converting to seconds")
                    elif dt > 0:
                        self._ts_is_ms = False

                if self._ts_is_ms:
                    ts_val = ts_raw / 1000.0
                else:
                    ts_val = ts_raw

                self._last_ts = ts_raw

                # Linear acceleration
                lin, _gravity = est.update(acc, gyr, 1.0 / 200.0)

                self.buffer.append(float(ts_val), acc, gyr, lin)

        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        print("[StreamClient] disconnected")


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════


def _set_axis_style(ax, title: str, ylabel: str, ylim: tuple[float, float]) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("time [s] (0 = now)", fontsize=9)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    ax.axhline(y=0, color="gray", ls="--", lw=0.6, alpha=0.4)


def run_plot(host: str, port: int, history_seconds: float) -> None:
    buffer = IMURingBuffer(capacity=int(history_seconds * 200 * 2))
    client = StreamHubClient(host, port, buffer)
    client.start()

    fig, (ax_raw, ax_lin, ax_gyr) = plt.subplots(3, 1, figsize=PLOT_FIGSIZE)
    fig.suptitle(
        f"Ring IMU — Raw Acc | Linear Acc | Gyro   ({host}:{port})",
        fontsize=13,
        fontweight="bold",
    )

    _set_axis_style(ax_raw, "Raw Acceleration [g]", "g", (-8, 8))
    _set_axis_style(ax_lin, "Linear Acceleration [g] (gravity removed)", "g", (-8, 8))
    _set_axis_style(ax_gyr, "Gyroscope [rad/s]", "rad/s", (-20, 20))

    (rx,) = ax_raw.plot([], [], color=COLOR_X, lw=0.7, label="X")
    (ry,) = ax_raw.plot([], [], color=COLOR_Y, lw=0.7, label="Y")
    (rz,) = ax_raw.plot([], [], color=COLOR_Z, lw=0.7, label="Z")
    (lx,) = ax_lin.plot([], [], color=COLOR_X, lw=0.7, label="X")
    (ly,) = ax_lin.plot([], [], color=COLOR_Y, lw=0.7, label="Y")
    (lz,) = ax_lin.plot([], [], color=COLOR_Z, lw=0.7, label="Z")
    (gx,) = ax_gyr.plot([], [], color=COLOR_X, lw=0.7, label="X")
    (gy,) = ax_gyr.plot([], [], color=COLOR_Y, lw=0.7, label="Y")
    (gz,) = ax_gyr.plot([], [], color=COLOR_Z, lw=0.7, label="Z")

    for ax in (ax_raw, ax_lin, ax_gyr):
        ax.legend(loc="upper right", fontsize=7, ncol=3)

    fig.tight_layout(pad=3.0)

    lines = [rx, ry, rz, lx, ly, lz, gx, gy, gz]
    data_started = [False]

    def animate(_frame: int) -> list:
        ts, raw_acc, raw_gyr, lin_acc = buffer.snapshot()
        if ts.size < 2:
            return lines

        now = ts[-1]
        rel_t = ts - now
        cutoff = -history_seconds
        mask = rel_t >= cutoff
        if not mask.any():
            return lines

        rel_t = rel_t[mask]

        # Build channel data arrays
        raw_data = [raw_acc[mask, i] for i in range(3)]
        lin_data = [lin_acc[mask, i] for i in range(3)]
        gyr_data = [raw_gyr[mask, i] for i in range(3)]

        # Update raw acc
        rx.set_data(rel_t, raw_data[0])
        ry.set_data(rel_t, raw_data[1])
        rz.set_data(rel_t, raw_data[2])

        # Update linear acc
        lx.set_data(rel_t, lin_data[0])
        ly.set_data(rel_t, lin_data[1])
        lz.set_data(rel_t, lin_data[2])

        # Update gyro
        gx.set_data(rel_t, gyr_data[0])
        gy.set_data(rel_t, gyr_data[1])
        gz.set_data(rel_t, gyr_data[2])

        # Keep x-axis anchored: [-history_seconds, 0]
        for ax in (ax_raw, ax_lin, ax_gyr):
            ax.set_xlim(-history_seconds, 0)

        # Auto-scale Y dynamically
        for ax, data_arrays in [(ax_raw, raw_data), (ax_lin, lin_data), (ax_gyr, gyr_data)]:
            y_max = 1.0
            for arr in data_arrays:
                if arr.size:
                    m = float(np.max(np.abs(arr)))
                    if np.isfinite(m):
                        y_max = max(y_max, m)
            y_max = max(y_max * 1.15, 1.0)  # 15% headroom, min ±1
            current_lo, current_hi = ax.get_ylim()
            if abs(y_max - current_hi) / max(current_hi, 0.1) > 0.15:
                ax.set_ylim(-y_max, y_max)

        if not data_started[0]:
            data_started[0] = True
            print(f"[Plot] rendering IMU data ({ts.size} points)")

        return lines

    def on_close(_event) -> None:
        client.stop()

    _anim = FuncAnimation(
        fig, animate, interval=PLOT_INTERVAL_MS,
        blit=False, cache_frame_data=False,
    )
    fig.canvas.mpl_connect("close_event", on_close)

    plt.tight_layout(pad=2.5)
    plt.show()

    client.stop()


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time ring IMU visualization (matplotlib)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--history", type=float, default=HISTORY_SECONDS,
        help=f"Seconds of history to display (default: {HISTORY_SECONDS})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Connecting to stream hub at {args.host}:{args.port}...")
    print(f"Showing last {args.history}s of IMU data.")
    print("Close the plot window to exit.\n")
    run_plot(args.host, args.port, args.history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
