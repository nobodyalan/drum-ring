import Foundation
import CoreMotion

/// 封装 CMMotionManager，以 60Hz 提供去重力后的 IMU 数据
final class SensorManager: ObservableObject {
    private let motion = CMMotionManager()

    /// 每收到一帧 IMU 数据时回调（在 main 队列）
    var onFrame: ((IMUFrame) -> Void)?

    /// 6 轴 IMU 帧（与 ZigSim 通道顺序一致）
    struct IMUFrame {
        /// 陀螺仪 (rad/s) — 通道 1-3
        let gx, gy, gz: Float
        /// 用户加速度 (g, 已去重力) — 通道 4-6
        let ax, ay, az: Float
        /// 到达时间戳 (Unix 秒)
        let timestamp: TimeInterval
    }

    func start() {
        guard motion.isDeviceMotionAvailable else {
            print("[SensorManager] DeviceMotion 不可用")
            return
        }

        motion.deviceMotionUpdateInterval = 1.0 / 60.0
        motion.startDeviceMotionUpdates(to: .main) { [weak self] data, error in
            if let error = error {
                print("[SensorManager] 错误: \(error.localizedDescription)")
                return
            }
            guard let self, let d = data else { return }

            self.onFrame?(IMUFrame(
                gx: Float(d.rotationRate.x),
                gy: Float(d.rotationRate.y),
                gz: Float(d.rotationRate.z),
                ax: Float(d.userAcceleration.x),
                ay: Float(d.userAcceleration.y),
                az: Float(d.userAcceleration.z),
                timestamp: Date().timeIntervalSince1970
            ))
        }
    }

    func stop() {
        motion.stopDeviceMotionUpdates()
    }
}
