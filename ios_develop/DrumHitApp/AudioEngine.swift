import AVFoundation

/// 低延迟音效播放引擎
/// 预加载 WAV → PCM Buffer，触发时立即播放（打断旧声音）
final class AudioEngine {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var buffer: AVAudioPCMBuffer?
    private var isStarted = false

    /// - Parameter wavName: Bundle 中的 WAV 文件名（不含扩展名），默认 "hit"
    init(wavName: String = "1") {
        guard let url = Bundle.main.url(forResource: wavName, withExtension: "wav") else {
            print("[AudioEngine] 未找到 \(wavName).wav")
            return
        }

        guard let file = try? AVAudioFile(forReading: url) else {
            print("[AudioEngine] 无法读取 \(wavName).wav")
            return
        }

        let format = file.processingFormat
        buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(file.length)
        )
        try? file.read(into: buffer!)

        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)

        print("[AudioEngine] 已加载 \(wavName).wav (\(file.length) 帧 @ \(format.sampleRate)Hz)")
    }

    private func ensureStarted() {
        guard !isStarted else { return }
        do {
            try engine.start()
            isStarted = true
        } catch {
            print("[AudioEngine] 启动失败: \(error)")
        }
    }

    /// 播放音效（打断当前播放，非阻塞）
    func play() {
        guard let buf = buffer else { return }
        ensureStarted()
        player.stop()                                 // 立即打断旧声音
        player.scheduleBuffer(buf, at: nil,
                              options: .interrupts,   // 有新的就打断旧的
                              completionHandler: nil)
        player.play()
    }
}
