from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppLaunchSpec:
    id: str
    title: str
    script_path: str
    module_path: str
    button_class: str = "btn-primary"


APP_LAUNCH_SPECS: tuple[AppLaunchSpec, ...] = (
    AppLaunchSpec("demo", "示范应用", "apps/demo_processor_app.py", "apps.demo_processor_app"),
    AppLaunchSpec("imu_visualizer", "IMU 可视化", "apps/imu_visualizer_app.py", "apps.imu_visualizer_app"),
    AppLaunchSpec("imu_realtime", "IMU 实时 (matplotlib)", "scripts/visualize_imu_realtime.py", "scripts.visualize_imu_realtime"),
    AppLaunchSpec(
        "pressure_imu_visualizer",
        "压力+IMU",
        "apps/pressure_imu_visualizer_app.py",
        "apps.pressure_imu_visualizer_app",
    ),
    AppLaunchSpec("audio_monitor", "音频监控", "apps/audio_monitor_app.py", "apps.audio_monitor_app"),
    AppLaunchSpec("recorder", "录制中心", "apps/recorder_app.py", "apps.recorder_app"),
    AppLaunchSpec("gesture_recognition", "手势识别", "apps/gesture_recognition_app.py", "apps.gesture_recognition_app"),
    AppLaunchSpec("drum_gesture", "Drum Gesture", "apps/drum_gesture_app.py", "apps.drum_gesture_app"),
    AppLaunchSpec("touchpad", "触控板", "apps/touchpad_app.py", "apps.touchpad_app"),
    AppLaunchSpec("unified_gesture", "综合识别", "apps/unified_gesture_app.py", "apps.unified_gesture_app"),
    AppLaunchSpec("handwriting_demo", "Handwriting Demo", "apps/handwriting_demo_app.py", "apps.handwriting_demo_app"),
    AppLaunchSpec("whisper_writing", "手写识别", "apps/whisper_number_touch_app.py", "apps.whisper_number_touch_app"),
    AppLaunchSpec("handwriting_experiment", "手写识别实验", "apps/handwriting_experiment_app.py", "apps.handwriting_experiment_app"),
    AppLaunchSpec("swipe", "Swipe 检测", "apps/swipe_realtime_app.py", "apps.swipe_realtime_app"),
    AppLaunchSpec("pinch", "Pinch 检测", "apps/pinch_realtime_app.py", "apps.pinch_realtime_app"),
    AppLaunchSpec("snake", "Snake", "apps/snake_app.py", "apps.snake_app"),
    AppLaunchSpec("typering", "戒指打字", "apps/typering_app.py", "apps.typering_app"),
)
