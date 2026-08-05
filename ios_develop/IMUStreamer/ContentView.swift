import SwiftUI
import UIKit

// MARK: - View

struct ContentView: View {
    @StateObject private var vm = StreamerViewModel()

    var body: some View {
        NavigationStack {
            Form {
                // ── 目标配置 ──
                Section("目标地址") {
                    HStack {
                        Text("IP").foregroundStyle(.secondary)
                        TextField("192.168.1.100", text: $vm.ipAddress)
                            .keyboardType(.decimalPad)
                            .autocorrectionDisabled()
                            .textInputAutocapitalization(.never)
                            .disabled(vm.isStreaming)
                    }
                    HStack {
                        Text("Port").foregroundStyle(.secondary)
                        TextField("8888", text: $vm.port)
                            .keyboardType(.numberPad)
                            .disabled(vm.isStreaming)
                    }
                    HStack {
                        Text("频率").foregroundStyle(.secondary)
                        Picker("", selection: $vm.targetHz) {
                            Text("60 Hz").tag(60.0)
                            Text("100 Hz").tag(100.0)
                            Text("150 Hz").tag(150.0)
                            Text("200 Hz").tag(200.0)
                        }
                        .pickerStyle(.segmented)
                        .disabled(vm.isStreaming)
                    }
                }

                // ── 控制 ──
                Section {
                    Button(action: vm.isStreaming ? vm.stop : vm.start) {
                        HStack {
                            Spacer()
                            Image(systemName: vm.isStreaming ? "stop.fill" : "play.fill")
                            Text(vm.isStreaming ? "停止发送" : "开始发送")
                                .fontWeight(.semibold)
                            Spacer()
                        }
                    }
                    .tint(vm.isStreaming ? .red : .green)
                    .buttonStyle(.borderedProminent)
                }

                // ── 状态 ──
                Section("状态") {
                    HStack {
                        Circle()
                            .fill(vm.statusColor)
                            .frame(width: 10, height: 10)
                        Text(vm.statusText)
                    }
                    LabeledContent("FPS") {
                        Text(vm.fpsText).fontDesign(.monospaced)
                    }
                    LabeledContent("已发送") {
                        Text("\(vm.packetCount) 包").fontDesign(.monospaced)
                    }
                    LabeledContent("数据量") {
                        Text(vm.bytesFormatted).fontDesign(.monospaced)
                    }
                    LabeledContent("每包大小") {
                        Text("\(vm.packetSize) B").fontDesign(.monospaced)
                    }
                }

                // ── 实时数据预览 ──
                Section("最后数据预览") {
                    dataRow("/gyro",       vm.gyroText)
                    dataRow("/accel",      vm.accelText)
                    dataRow("/quaternion", vm.quatText)
                    dataRow("/gravity",    vm.gravityText)
                }
            }
            .navigationTitle("IMU Streamer")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    if vm.isStreaming {
                        Text("● LIVE").foregroundColor(.red).font(.caption)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func dataRow(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(.caption, design: .monospaced))
                .lineLimit(1)
        }
    }
}

// MARK: - ViewModel

@MainActor
final class StreamerViewModel: ObservableObject {

    // MARK: Published config
    @Published var ipAddress = "192.168.1.100"
    @Published var port = "8888"
    @Published var targetHz = 200.0

    // MARK: Published status
    @Published var isStreaming = false
    @Published var statusText = "就绪"
    @Published var fps: Double = 0
    @Published var packetCount: Int64 = 0
    @Published var byteCount: Int64 = 0
    @Published var packetSize: Int = 0
    @Published var gyroText = "-"
    @Published var accelText = "-"
    @Published var quatText = "-"
    @Published var gravityText = "-"

    var statusColor: Color {
        if isStreaming { return .green }
        return .secondary
    }
    var fpsText: String { String(format: "%.0f Hz", fps) }
    var bytesFormatted: String {
        if byteCount < 1024 { return "\(byteCount) B" }
        if byteCount < 1_048_576 { return String(format: "%.1f KB", Double(byteCount)/1024) }
        return String(format: "%.1f MB", Double(byteCount)/1_048_576)
    }

    // MARK: Private engine
    private let sensor = SensorManager()
    private let sender = OSCSender()
    private var fpsTimer: Timer?

    // Thread-safe counters — sensor 线程写入, timer 线程读取+重置
    private var frameCounter: Int64 = 0
    private var byteCounter: Int64 = 0
    private var counterLock = os_unfair_lock_s()

    /// 最近一帧的副本 — sensor 线程写入, timer 线程读取
    private var latestFrame: SensorManager.IMUFrame?
    private var frameLock = os_unfair_lock_s()

    // MARK: - Actions

    func start() {
        // 1. 校验端口
        guard let portNum = UInt16(port), portNum > 0 else {
            statusText = "端口无效 (1-65535)"
            return
        }

        // 2. 快速预检查 IP 格式
        let octets = ipAddress.split(separator: ".")
        guard octets.count == 4, octets.allSatisfy({ Int($0) != nil }) else {
            statusText = "IP 格式无效 (例: 192.168.1.100)"
            return
        }

        // 3. 配置 UDP socket
        statusText = "连接中..."
        guard sender.configure(host: ipAddress, port: portNum) else {
            statusText = "无法连接 — 检查 IP 和同一 WiFi"
            return
        }

        // 4. 启动传感器（必须成功才继续）
        guard sensor.start(hz: targetHz) else {
            statusText = "传感器不可用 (真机? 权限?)"
            sender.close()
            return
        }

        // 5. 全部就绪，重置计数器
        frameCounter = 0
        byteCounter = 0
        packetCount = 0
        byteCount = 0
        fps = 0

        // 传感器回调 (后台线程, ~200Hz)
        sensor.onFrame = { [weak self] frame in
            guard let self else { return }

            // 1. 编码 OSC Bundle
            let data = OSCEncoder.encode(
                gyro:     (frame.gyroX, frame.gyroY, frame.gyroZ),
                accel:    (frame.accelX, frame.accelY, frame.accelZ),
                quat:     (frame.quatX, frame.quatY, frame.quatZ, frame.quatW),
                gravity:  (frame.gravityX, frame.gravityY, frame.gravityZ)
            )

            // 2. 发送
            self.sender.send(data)

            // 3. 更新计数器 (lock 保护)
            os_unfair_lock_lock(&self.counterLock)
            self.frameCounter += 1
            self.byteCounter += Int64(data.count)
            os_unfair_lock_unlock(&self.counterLock)

            // 4. 保存最新数据副本
            os_unfair_lock_lock(&self.frameLock)
            self.latestFrame = frame
            self.packetSize = data.count
            os_unfair_lock_unlock(&self.frameLock)
        }

        // 启动传感器
        sensor.start(hz: targetHz)

        // 启动 FPS 更新定时器 (4Hz)
        fpsTimer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.updateStats()
            }
        }

        // 保持屏幕常亮
        UIApplication.shared.isIdleTimerDisabled = true

        isStreaming = true
        statusText = "发送中 → \(ipAddress):\(portNum)"
    }

    func stop() {
        sensor.stop()
        fpsTimer?.invalidate()
        fpsTimer = nil
        sender.close()
        UIApplication.shared.isIdleTimerDisabled = false

        isStreaming = false
        statusText = "已停止 (共 \(packetCount) 包)"
    }

    // MARK: - Periodic UI Update (called by Timer on MainActor)

    private func updateStats() {
        // 读取并重置计数器
        os_unfair_lock_lock(&counterLock)
        let frames = frameCounter
        let bytes = byteCounter
        frameCounter = 0
        byteCounter = 0
        os_unfair_lock_unlock(&counterLock)

        // FPS = 帧数 / 时间间隔 (0.25s)
        fps = Double(frames) / 0.25
        packetCount += frames
        byteCount += bytes

        // 读取最新数据预览
        os_unfair_lock_lock(&frameLock)
        if let f = latestFrame {
            gyroText  = String(format: "%+.3f %+.3f %+.3f", Double(f.gyroX), Double(f.gyroY), Double(f.gyroZ))
            accelText = String(format: "%+.3f %+.3f %+.3f", Double(f.accelX), Double(f.accelY), Double(f.accelZ))
            quatText  = String(format: "%+.3f %+.3f %+.3f %+.3f", Double(f.quatX), Double(f.quatY), Double(f.quatZ), Double(f.quatW))
            gravityText = String(format: "%+.3f %+.3f %+.3f", Double(f.gravityX), Double(f.gravityY), Double(f.gravityZ))
        }
        os_unfair_lock_unlock(&frameLock)
    }
}

// MARK: - Preview

#Preview {
    ContentView()
}
