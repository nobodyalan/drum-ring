import Foundation

/// 固定容量环形缓冲区，线程不安全（调用方保证串行访问）
struct RingBuffer<Element> {
    private var storage: [Element?]
    private var writeIndex: Int = 0
    private(set) var count: Int = 0
    let capacity: Int

    init(capacity: Int) {
        self.capacity = capacity
        self.storage = Array(repeating: nil, count: capacity)
    }

    @discardableResult
    mutating func append(_ element: Element) -> Element? {
        let overwritten = storage[writeIndex]
        storage[writeIndex] = element
        writeIndex = (writeIndex + 1) % capacity
        if count < capacity { count += 1 }
        return overwritten
    }

    /// 按插入顺序访问，0 = 最旧，count-1 = 最新
    subscript(index: Int) -> Element {
        guard index >= 0, index < count else {
            fatalError("RingBuffer index \(index) out of bounds [0..<\(count))")
        }
        let readIndex = (writeIndex - count + index + capacity) % capacity
        return storage[readIndex]!
    }
}
