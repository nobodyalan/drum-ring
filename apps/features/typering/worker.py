from __future__ import annotations

import json
import queue
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import typering_core.math as cm
from typering_core.model import TouchDetectionModel8

from .config import TypeRingConfig, TypeRingPaths


MSG_READY = "READY"
MSG_ERROR = "ERROR"
MSG_EVENT = "EVENT"
MSG_STATUS = "STATUS"
MSG_QUIT = "QUIT"
MSG_IMU_FRAME = "IMU_FRAME"
MSG_TOUCH_EVENT = "TOUCH_EVENT"


@dataclass(slots=True)
class WorkerRuntime:
    role: str
    config: TypeRingConfig
    paths: TypeRingPaths


def _load_model(runtime: WorkerRuntime):
    model_dir = runtime.paths.touch_detection_model_dir(runtime.config.model_name)
    args = json.load(open(model_dir / "args.json", "r"))
    model = TouchDetectionModel8(
        args["n_in"],
        args["n_out"],
        args["dropout"],
        args["alpha"],
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    return model.eval()


def typering_worker_main(
    role: str,
    root_dir: str,
    config_dict: dict,
    input_queue,
    output_queue,
) -> None:
    runtime = WorkerRuntime(
        role=role,
        config=TypeRingConfig(**config_dict),
        paths=TypeRingPaths(Path(root_dir)),
    )
    label_names = ["none", "thumb", "index", "middle", "ring/pinky"]
    acc_mean, acc_std = -2.6163, 5.6607
    gyro_mean, gyro_std = 0.0055, 1.1548

    try:
        model = _load_model(runtime)
    except Exception as exc:
        output_queue.put({"type": MSG_ERROR, "role": role, "message": str(exc)})
        return

    all_imu = np.zeros((0, 6), dtype=np.float32)
    all_ts = np.zeros((0,), dtype=np.float64)
    pred_idx = 0
    pred_label = 0
    pred_cnt = 0
    last_pred_t = -1e10
    last_tap_timestamp = None
    mirror = role == "left" and runtime.config.left_window_mirror
    output_queue.put({"type": MSG_READY, "role": role})

    while True:
        try:
            msg = input_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        msg_type = msg.get("type")
        if msg_type == MSG_QUIT:
            break
        if msg_type == MSG_TOUCH_EVENT:
            action = str(msg.get("action", ""))
            ts = float(msg.get("timestamp", time.perf_counter()))
            if action == "press":
                last_tap_timestamp = ts
            elif action == "release":
                if last_tap_timestamp is not None:
                    output_queue.put({"type": MSG_EVENT, "role": role, "event": "tap", "timestamp": last_tap_timestamp})
                last_tap_timestamp = None
            continue
        if msg_type != MSG_IMU_FRAME:
            continue

        acc = np.asarray(msg.get("acc", []), dtype=np.float32)
        gyr = np.asarray(msg.get("gyr", []), dtype=np.float32)
        if acc.shape != (3,) or gyr.shape != (3,):
            continue
        ts = float(msg.get("timestamp", 0.0))
        sample = np.concatenate([acc, gyr], axis=0)[None, :]
        if mirror:
            sample[:, 0] = -sample[:, 0]
            sample[:, 4] = -sample[:, 4]
            sample[:, 5] = -sample[:, 5]
        all_imu = np.concatenate([all_imu, sample], axis=0)
        all_ts = np.concatenate([all_ts, np.asarray([ts], dtype=np.float64)], axis=0)

        now = time.perf_counter()
        if last_tap_timestamp is not None and now >= last_tap_timestamp + runtime.config.long_tap_threshold:
            output_queue.put({"type": MSG_EVENT, "role": role, "event": "long tap", "timestamp": now})
            last_tap_timestamp = None

        while len(all_imu) - pred_idx >= runtime.config.window_size:
            window = all_imu[pred_idx:pred_idx + runtime.config.window_size, :].copy()
            window[:, :3] = (window[:, :3] - acc_mean) / acc_std
            window[:, 3:] = (window[:, 3:] - gyro_mean) / gyro_std
            tensor = torch.FloatTensor(window[None])
            with torch.no_grad():
                out = model(tensor)[0]
            label = int(torch.argmax(out))
            if label == pred_label:
                pred_cnt += 1
            else:
                pred_label, pred_cnt = label, 1

            if pred_label != 0 and pred_cnt == runtime.config.stable_count:
                rcs = cm.rcs(all_imu[pred_idx:pred_idx + runtime.config.window_size, :3], d=2.0)
                pred_t = float(all_ts[pred_idx + int(np.argmax(rcs))])
                if pred_t - last_pred_t >= runtime.config.min_prediction_interval:
                    last_pred_t = pred_t
                    output_queue.put(
                        {
                            "type": MSG_EVENT,
                            "role": role,
                            "event": label_names[label],
                            "timestamp": pred_t,
                        }
                    )
            pred_idx += runtime.config.stride
