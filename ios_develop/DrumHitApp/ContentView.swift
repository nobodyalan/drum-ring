import SwiftUI

/// 最小 UI：击中计数 + 实时概率条 + 状态指示
struct ContentView: View {
    @StateObject private var viewModel = DrumHitViewModel()

    var body: some View {
        VStack(spacing: 24) {
            // 标题
            Text("🥁 鼓击检测")
                .font(.title)
                .foregroundStyle(.secondary)

            // 击中计数
            Text("\(viewModel.hitCount)")
                .font(.system(size: 96, weight: .bold, design: .rounded))
                .foregroundColor(viewModel.armed ? .red : .primary)
                .contentTransition(.numericText())
                .animation(.easeInOut(duration: 0.1), value: viewModel.hitCount)

            // 状态标签
            HStack(spacing: 12) {
                Circle()
                    .fill(viewModel.armed ? Color.red : Color.green)
                    .frame(width: 12, height: 12)
                Text(viewModel.statusText)
                    .font(.headline)
                    .foregroundStyle(.secondary)
            }

            // 实时概率条
            VStack(alignment: .leading, spacing: 4) {
                Text("P(hit): \(viewModel.latestProb, specifier: "%.3f")")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 4)
                            .fill(.quaternary)
                            .frame(height: 8)

                        RoundedRectangle(cornerRadius: 4)
                            .fill(viewModel.barColor)
                            .frame(width: geo.size.width * CGFloat(viewModel.latestProb), height: 8)
                            .animation(.easeOut(duration: 0.05), value: viewModel.latestProb)

                        // probHigh 阈值线
                        Rectangle()
                            .fill(.red.opacity(0.5))
                            .frame(width: 1, height: 16)
                            .offset(x: geo.size.width * CGFloat(viewModel.probHigh), y: -4)
                    }
                }
                .frame(height: 16)

                HStack {
                    Text("0")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                    Spacer()
                    Text("1")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(.horizontal, 32)

            // 参数显示
            HStack(spacing: 24) {
                LabeledContent("高阈值") {
                    Text("\(viewModel.probHigh, specifier: "%.2f")")
                }
                LabeledContent("低阈值") {
                    Text("\(viewModel.probLow, specifier: "%.2f")")
                }
                LabeledContent("冷却") {
                    Text("\(viewModel.cooldownMs)ms")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)

            Spacer()

            // 重置按钮
            Button("重置计数") {
                viewModel.reset()
            }
            .buttonStyle(.bordered)
        }
        .padding()
    }
}

/// ViewModel —— 串联 SensorManager → InferenceEngine → DetectionStateMachine → AudioEngine
@MainActor
final class DrumHitViewModel: ObservableObject {
    @Published var hitCount: Int = 0
    @Published var latestProb: Float = 0
    @Published var armed: Bool = false

    let probHigh: Float = 0.85
    let probLow: Float = 0.40
    let cooldownMs: Int = 400

    var statusText: String { armed ? "ARMED" : "就绪" }
    var barColor: Color {
        if armed { return .red }
        if latestProb > probHigh { return .orange }
        return .blue
    }

    private let sensor = SensorManager()
    private let engine: InferenceEngine
    private let stateMachine: DetectionStateMachine
    private let audio = AudioEngine()

    init() {
        // 1. 加载 CoreML 模型
        guard let mlModel = try? HitDetector().model else {
            fatalError("无法加载 HitDetector.mlpackage，请确认已拖入 Xcode 项目")
        }

        // 2. 读取标准化参数（从 preproc_params.json）
        let params = DrumHitViewModel.loadPreprocParams()
        let mean = params.mean
        let std = params.std

        self.engine = InferenceEngine(mlModel: mlModel, mean: mean, std: std)

        // 3. 状态机
        self.stateMachine = DetectionStateMachine(
            probHigh: probHigh,
            probLow: probLow,
            cooldown: Double(cooldownMs) / 1000.0
        )

        // 4. 连线
        stateMachine.onTrigger = { [weak self] in
            self?.audio.play()
        }

        sensor.onFrame = { [weak self] frame in
            self?.handleFrame(frame)
        }

        sensor.start()
    }

    // MARK: - Private

    private func handleFrame(_ frame: SensorManager.IMUFrame) {
        let prob = engine.predict(frame: frame)

        let triggered = stateMachine.update(prob: prob, timestamp: frame.timestamp)

        // 更新 UI（在 MainActor 上）
        latestProb = prob
        armed = stateMachine.armed
        if triggered {
            hitCount = stateMachine.hitCount
        }
    }

    func reset() {
        stateMachine.resetCount()
        hitCount = 0
    }

    // MARK: - 读取标准化参数

    private struct PreprocParams {
        let mean: [Float]
        let std: [Float]
    }

    private static func loadPreprocParams() -> PreprocParams {
        guard let url = Bundle.main.url(forResource: "preproc_params", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let mean = json["mean"] as? [Double],
              let std = json["std"] as? [Double]
        else {
            fatalError("无法读取 preproc_params.json，请确认已拖入 Xcode 项目")
        }
        return PreprocParams(
            mean: mean.map { Float($0) },
            std: std.map { Float($0) }
        )
    }
}

#Preview {
    ContentView()
}
