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

    // MARK: - 频率对比测试（临时）

    /// 同时启动原始加速度计，测量实际可达到的采样率
    func startRawAccelRateTest() {
        guard motion.isAccelerometerAvailable else {
            print("[RateTest] ❌ 加速度计不可用")
            return
        }
        motion.accelerometerUpdateInterval = 1.0 / 200.0  // 请求 200Hz
        var count = 0
        var t0 = CACurrentMediaTime()

        motion.startAccelerometerUpdates(to: queue) { data, error in
            count += 1
            let now = CACurrentMediaTime()
            if now - t0 >= 1.0 {
                let rate = Double(count) / (now - t0)
                print("[RateTest] 原始加速度计: \(count) 帧 = \(Int(rate)) Hz")
                count = 0
                t0 = now
            }
        }
        print("[RateTest] ✅ 原始加速度计已启动, 目标 200Hz, 每秒打印实际频率")
    }

    func stopRawAccelRateTest() {
        motion.stopAccelerometerUpdates()
        print("[RateTest] ⏹ 已停止")
    }

    // MARK: - 原始传感器模式（替代 CMDeviceMotion）

    /// 使用原始加速度计 + 陀螺仪，频率不受 CMDeviceMotion 姿态融合限制
    /// gravity 从 CMDeviceMotion 以较低频率获取，quaternion 填占位值
    @discardableResult
    func startRaw(hz: Double = 200) -> Bool {
        guard motion.isAccelerometerAvailable, motion.isGyroAvailable else {
            print("[SensorManager] ❌ 原始传感器不可用")
            return false
        }

        motion.accelerometerUpdateInterval = 1.0 / hz
        motion.gyroUpdateInterval = 1.0 / hz

        // 缓存最新帧的各分量（两个传感器不同步，各取最新值合并）
        var latestAccelX: Float = 0, latestAccelY: Float = 0, latestAccelZ: Float = 0
        var latestGyroX: Float = 0, latestGyroY: Float = 0, latestGyroZ: Float = 0
        let dataLock = os_unfair_lock_s()

        motion.startAccelerometerUpdates(to: queue) { data, _ in
            guard let d = data else { return }
            os_unfair_lock_lock(&dataLock)
            latestAccelX = Float(d.acceleration.x)
            latestAccelY = Float(d.acceleration.y)
            latestAccelZ = Float(d.acceleration.z)
            os_unfair_lock_unlock(&dataLock)
        }

        // 陀螺仪回调触发帧输出（两者取最新合并）
        motion.startGyroUpdates(to: queue) { [weak self] data, _ in
            guard let self, let d = data else { return }
            os_unfair_lock_lock(&dataLock)
            latestGyroX = Float(d.rotationRate.x)
            latestGyroY = Float(d.rotationRate.y)
            latestGyroZ = Float(d.rotationRate.z)
            let ax = latestAccelX, ay = latestAccelY, az = latestAccelZ
            let gx = latestGyroX, gy = latestGyroY, gz = latestGyroZ
            os_unfair_lock_unlock(&dataLock)

            let frame = IMUFrame(
                gyroX: gx, gyroY: gy, gyroZ: gz,
                accelX: ax, accelY: ay, accelZ: az,  // 原始加速度（含重力）
                quatX: 0, quatY: 0, quatZ: 0, quatW: 1,  // 占位
                gravityX: 0, gravityY: 0, gravityZ: 0,    // 占位
                timestamp: d.timestamp
            )
            self.onFrame?(frame)
        }

        print("[SensorManager] ✅ 原始传感器已启动, 目标 \(Int(hz))Hz")
        return true
    }

    func stopRaw() {
        motion.stopAccelerometerUpdates()
        motion.stopGyroUpdates()
        print("[SensorManager] ⏹ 原始传感器已停止")
    }
