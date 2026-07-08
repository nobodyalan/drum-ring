import Foundation
import CoreML

/// CoreML 推理引擎：滑动窗口缓冲 → Z-score 标准化 → 模型推理 → P(hit)
/// 完全对应 realtime_detect.py 的 RealtimeDetector.predict()
final class InferenceEngine {
    private let model: MLModel
    private var buffer = RingBuffer<[Float]>(capacity: 15)
    private let mean: [Float]
    private let std: [Float]
    private let inputName: String
    private let outputName: String

    /// - Parameters:
    ///   - mlModel: 从 .mlpackage 加载的 CoreML 模型
    ///   - mean: 6 通道均值（对应训练时的 Z-score mean）
    ///   - std: 6 通道标准差（对应训练时的 Z-score std）
    init(mlModel: MLModel, mean: [Float], std: [Float]) {
        self.model = mlModel
        self.mean = mean
        self.std = std

        // 从模型描述中获取输入/输出名字（兼容 coremltools 自动命名）
        self.inputName = mlModel.modelDescription.inputDescriptionsByName.keys.first!
        self.outputName = mlModel.modelDescription.outputDescriptionsByName.keys.first!

        print("[InferenceEngine] 模型已加载")
        print("  输入: \(inputName)  shape=\(mlModel.modelDescription.inputDescriptionsByName[inputName]!.multiArrayConstraint!.shape)")
        print("  输出: \(outputName)")
        print("  mean: \(mean)")
        print("  std:  \(std)")
    }

    /// 喂入一帧 IMU，返回 P(hit) ∈ [0, 1]
    /// 窗口未满 15 帧时返回 0
    func predict(frame: SensorManager.IMUFrame) -> Float {
        // 通道顺序：[gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]
        buffer.append([frame.gx, frame.gy, frame.gz,
                       frame.ax, frame.ay, frame.az])

        guard buffer.count >= 15 else { return 0.0 }

        // 构建 MLMultiArray (1, 6, 15) + 逐通道 Z-score 标准化
        let array = try! MLMultiArray(shape: [1, 6, 15] as [NSNumber], dataType: .float32)
        for c in 0..<6 {
            let m = mean[c]
            let s = std[c] + 1e-8
            for t in 0..<15 {
                array[[0, c, t] as [NSNumber]] = ((buffer[t][c] - m) / s) as NSNumber
            }
        }

        // 模型推理
        let input = try! MLDictionaryFeatureProvider(
            dictionary: [inputName: array]
        )
        let output = try! model.prediction(from: input)

        // 取 logits，计算 softmax
        guard let logitsArray = output.featureValue(for: outputName)?.multiArrayValue else {
            return 0.0
        }
        let logit0 = Float(truncating: logitsArray[0])
        let logit1 = Float(truncating: logitsArray[1])

        // softmax: P(hit) = exp(logit1) / (exp(logit0) + exp(logit1))
        let exp0 = exp(logit0)
        let exp1 = exp(logit1)
        return exp1 / (exp0 + exp1)
    }
}
