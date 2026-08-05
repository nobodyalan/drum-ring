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
    func configure(host: String, port: UInt16) -> Bool {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        // 关闭旧 socket
        _close()

        // 创建 UDP socket
        sock = Darwin.socket(AF_INET, SOCK_DGRAM, 0)
        guard sock >= 0 else { return false }

        // 解析主机名
        var hint = addrinfo(
            ai_flags: 0,
            ai_family: AF_INET,
            ai_socktype: SOCK_DGRAM,
            ai_protocol: IPPROTO_UDP,
            ai_addrlen: 0,
            ai_canonname: nil,
            ai_addr: nil,
            ai_next: nil
        )
        var result: UnsafeMutablePointer<addrinfo>?
        let portStr = String(port)
        let status = getaddrinfo(host, portStr, &hint, &result)
        guard status == 0, let info = result?.pointee else {
            _close()
            return false
        }

        // 安全复制 sockaddr_in (避免 withMemoryRebound 的 UB)
        memcpy(&addr, info.ai_addr, Int(info.ai_addrlen))
        freeaddrinfo(result)

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
