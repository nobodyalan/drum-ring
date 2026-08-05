import Foundation
import CoreMotion

/// CoreMotion 管理器 — 200Hz IMU 数据采集
///
/// 关键设计：
///   - 使用专用后台 OperationQueue 接收回调，不阻塞主线程
///   - 回调中直接提取所有需要的字段（避免后续访问 CMDeviceMotion 的线程问题）
///   - deviceMotionUpdateInterval = 1/200 = 5ms
///
/// ⚠️ 实际采样率：
///   - iPhone 12+ 通常能达到 ~190-200Hz
///   - 旧设备可能被系统降频到 ~100Hz
///   - 低电量模式下会被强制降低
///   - 请在实际设备上测试确认
final class SensorManager {

    /// 一帧完整的 IMU 数据（从 CMDeviceMotion 提取）
    struct IMUFrame {
        let gyroX: Float
        let gyroY: Float
        let gyroZ: Float
        let accelX: Float      // userAcceleration (已去重力)
        let accelY: Float
        let accelZ: Float
        let quatX: Float       // attitude quaternion
        let quatY: Float
        let quatZ: Float
        let quatW: Float
        let gravityX: Float    // gravity vector
        let gravityY: Float
        let gravityZ: Float
        let timestamp: TimeInterval
    }

    private let motion = CMMotionManager()
    private let queue: OperationQueue = {
        let q = OperationQueue()
        q.name = "com.imustreamer.sensor"
        q.maxConcurrentOperationCount = 1  // 串行队列，保证帧顺序
        q.qualityOfService = .userInteractive
        return q
    }()

    /// 每收到一帧数据时回调（在后台队列上调用）
    var onFrame: ((IMUFrame) -> Void)?

    // MARK: - Public

    /// 启动传感器，以指定频率采集
    /// - Parameter hz: 目标频率，默认 200Hz
    /// - Returns: true 如果成功启动
    @discardableResult
    func start(hz: Double = 200) -> Bool {
        guard motion.isDeviceMotionAvailable else {
            print("[SensorManager] ❌ DeviceMotion 不可用 (模拟器?)")
            return false
        }

        motion.deviceMotionUpdateInterval = 1.0 / hz

        motion.startDeviceMotionUpdates(to: queue) { [weak self] data, error in
            guard let self else { return }

            if let error = error {
                print("[SensorManager] 错误: \(error.localizedDescription)")
                return
            }
            guard let d = data else { return }

            let frame = IMUFrame(
                gyroX: Float(d.rotationRate.x),
                gyroY: Float(d.rotationRate.y),
                gyroZ: Float(d.rotationRate.z),
                accelX: Float(d.userAcceleration.x),
                accelY: Float(d.userAcceleration.y),
                accelZ: Float(d.userAcceleration.z),
                quatX: Float(d.attitude.quaternion.x),
                quatY: Float(d.attitude.quaternion.y),
                quatZ: Float(d.attitude.quaternion.z),
                quatW: Float(d.attitude.quaternion.w),
                gravityX: Float(d.gravity.x),
                gravityY: Float(d.gravity.y),
                gravityZ: Float(d.gravity.z),
                timestamp: d.timestamp  // CMDeviceMotion 自带的时间戳 (秒, 自 boot)
            )

            self.onFrame?(frame)
        }

        print("[SensorManager] ✅ 已启动, 目标 \(Int(hz))Hz, "
              + "间隔 \(String(format: "%.1f", 1000.0/hz))ms")
        return true
    }

    func stop() {
        motion.stopDeviceMotionUpdates()
        print("[SensorManager] ⏹ 已停止")
    }

    deinit {
        stop()
    }
}
