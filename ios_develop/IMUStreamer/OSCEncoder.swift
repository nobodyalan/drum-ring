import Foundation

/// ZigSim 兼容的 OSC Bundle 编码器
///
/// 输出格式与 ZigSim App 完全一致，可直接被现有 Python 解析器接收：
///   parse_osc_bundle() in realtime_simple.py / realtime_mag_jerk.py / measure_latency.py
enum OSCEncoder {

    // MARK: - Public

    /// 将一个 IMU 帧编码为 OSC Bundle 字节
    /// - Parameters:
    ///   - gyro: 陀螺仪 (x, y, z) in rad/s
    ///   - accel: 用户加速度 (x, y, z) in g（已去重力）
    ///   - quat: 姿态四元数 (x, y, z, w)
    ///   - gravity: 重力向量 (x, y, z) in g
    /// - Returns: 完整的 OSC Bundle 字节，可直接 sendto()
    static func encode(
        gyro: (Float, Float, Float),
        accel: (Float, Float, Float),
        quat: (Float, Float, Float, Float),
        gravity: (Float, Float, Float)
    ) -> Data {
        var bundle = Data(capacity: 200)

        // 1. Bundle 头: "#bundle\0" (8 bytes)
        bundle.append(contentsOf: [0x23, 0x62, 0x75, 0x6E, 0x64, 0x6C, 0x65, 0x00])

        // 2. Timetag: 0x0000000000000001 = "immediate" (8 bytes)
        bundle.append(Data(repeating: 0, count: 7))
        bundle.append(0x01)

        // 3. 四条 OSC Message，每条前加 4-byte big-endian 长度
        let messages: [(String, [Float])] = [
            ("/gyro",       [gyro.0, gyro.1, gyro.2]),
            ("/accel",      [accel.0, accel.1, accel.2]),
            ("/quaternion", [quat.0, quat.1, quat.2, quat.3]),
            ("/gravity",    [gravity.0, gravity.1, gravity.2]),
        ]

        for (address, values) in messages {
            let msg = encodeMessage(address: address, values: values)
            var len = UInt32(msg.count).bigEndian
            withUnsafeBytes(of: &len) { bundle.append(contentsOf: $0) }
            bundle.append(msg)
        }

        return bundle
    }

    // MARK: - Private

    /// 编码单条 OSC Message
    private static func encodeMessage(address: String, values: [Float]) -> Data {
        var msg = Data(capacity: 64)

        // Address: 字符串 + null + 4-byte 对齐
        msg.append(address.data(using: .ascii)!)
        padTo4(&msg)

        // Type tag: ",fff..." + null + 4-byte 对齐
        let typeTag = "," + String(repeating: "f", count: values.count)
        msg.append(typeTag.data(using: .ascii)!)
        padTo4(&msg)

        // Values: big-endian float
        for v in values {
            var be = v.bitPattern.bigEndian
            withUnsafeBytes(of: &be) { msg.append(contentsOf: $0) }
        }

        return msg
    }

    /// 写入 null 终止符并补齐到 4 字节边界
    private static func padTo4(_ data: inout Data) {
        data.append(0) // null terminator
        while data.count & 3 != 0 {
            data.append(0)
        }
    }
}
