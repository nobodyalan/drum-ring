import Foundation
import Darwin

/// UDP OSC 发送器 — 基于 POSIX socket，适合 200Hz 高频场景
///
/// 为什么用 Darwin socket 而非 Network framework:
///   - NWConnection 的异步 send 在 200Hz 下会增加队列调度开销
///   - POSIX sendto() 是同步调用，微秒级返回，更适合高频发送
///   - 发送失败时静默丢弃 (UDP 语义)
final class OSCSender {

    private var sock: Int32 = -1
    private var addr = sockaddr_in()
    private var lock = os_unfair_lock_s()

    // MARK: - Public

    /// 配置目标地址，成功返回 true
    /// 使用 inet_pton 直接解析 IP（无需 DNS，不阻塞主线程）
    func configure(host: String, port: UInt16) -> Bool {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        // 关闭旧 socket
        _close()

        // 创建 UDP socket
        sock = Darwin.socket(AF_INET, SOCK_DGRAM, 0)
        guard sock >= 0 else { return false }

        // 直接用 inet_pton 解析 IP 地址（微秒级，不阻塞）
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        let result = host.withCString { inet_pton(AF_INET, $0, &addr.sin_addr) }
        guard result == 1 else {
            _close()
            return false
        }

        return true
    }

    /// 发送数据 — 线程安全，可在任意线程调用
    /// 发送失败时静默丢弃 (UDP 不保证送达)
    func send(_ data: Data) {
        os_unfair_lock_lock(&lock)
        let currentSock = sock
        let currentAddr = addr
        os_unfair_lock_unlock(&lock)

        guard currentSock >= 0 else { return }

        data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
            guard let base = ptr.baseAddress else { return }
            withUnsafePointer(to: currentAddr) { sa in
                sa.withMemoryRebound(to: sockaddr.self, capacity: 1) { sap in
                    // 返回值忽略：UDP 丢包是正常的
                    _ = Darwin.sendto(currentSock, base, data.count, 0, sap, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
        }
    }

    /// 关闭 socket
    func close() {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        _close()
    }

    // MARK: - Private

    private func _close() {
        guard sock >= 0 else { return }
        Darwin.close(sock)
        sock = -1
    }
}
