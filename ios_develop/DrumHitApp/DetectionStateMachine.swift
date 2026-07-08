import Foundation

/// 双阈值滞回状态机 + 冷却
/// 完全对应 realtime_detect.py:405-420
final class DetectionStateMachine: ObservableObject {
    /// 触发阈值：P(hit) > probHigh → 触发
    let probHigh: Float
    /// 释放阈值：P(hit) < probLow → 解除滞回
    let probLow: Float
    /// 冷却时间 (秒)
    let cooldown: TimeInterval

    /// 是否处于滞回状态（已触发，等待概率回落到低阈值以下）
    @Published private(set) var armed = false
    /// 击中计数
    @Published private(set) var hitCount: Int = 0
    /// 最近一次 P(hit)
    @Published private(set) var latestProb: Float = 0

    private var lastTriggerTime: TimeInterval = 0

    /// 触发时回调（在 update() 所在的线程同步调用）
    var onTrigger: (() -> Void)?

    init(probHigh: Float = 0.85,
         probLow: Float = 0.40,
         cooldown: TimeInterval = 0.4) {
        self.probHigh = probHigh
        self.probLow = probLow
        self.cooldown = cooldown
    }

    /// 喂入最新概率和时间戳。返回是否触发了新击中。
    @discardableResult
    func update(prob: Float, timestamp: TimeInterval) -> Bool {
        latestProb = prob

        let inCooldown = (timestamp - lastTriggerTime) < cooldown

        if !armed {
            if prob > probHigh && !inCooldown {
                armed = true
                lastTriggerTime = timestamp
                hitCount += 1
                onTrigger?()
                return true
            }
        } else {
            if prob < probLow {
                armed = false
            }
        }
        return false
    }

    /// 重置计数器
    func resetCount() {
        hitCount = 0
        armed = false
    }
}
