"""
临时工具: 监听 ZigSim OSC 数据, 打印所有 OSC 地址和值
========================================================
用法:
  python debug_osc.py
  python debug_osc.py --port 8888 --duration 10

ZigSim 连接后运行，会打印所有收到的 OSC 消息地址和数据类型，
用于确认 ZigSim 实际发送了哪些传感器数据。
"""
import struct
import socket
import time
import argparse

def parse_osc_message(chunk: bytes, offset: int) -> tuple[str, list, int]:
    """解析单个 OSC 消息, 返回 (address, values, new_offset)"""
    # 地址字符串
    end = chunk.find(b"\0", offset)
    if end <= offset:
        return "", [], offset
    addr = chunk[offset:end].decode("ascii", errors="replace")
    offset = (end + 4) & ~3  # 对齐到 4 字节

    # 类型标签
    if offset >= len(chunk) or chunk[offset:offset+1] != b",":
        return addr, [], offset
    type_end = chunk.find(b"\0", offset)
    if type_end <= offset:
        return addr, [], offset
    types = chunk[offset+1:type_end].decode("ascii", errors="replace")
    offset = (type_end + 4) & ~3

    values = []
    for t in types:
        if t == 'f':
            vals = struct.unpack(">f", chunk[offset:offset+4])
            values.append(round(float(vals[0]), 6))
            offset += 4
        elif t == 'i':
            vals = struct.unpack(">i", chunk[offset:offset+4])
            values.append(int(vals[0]))
            offset += 4
        elif t == 'd':
            vals = struct.unpack(">d", chunk[offset:offset+8])
            values.append(round(float(vals[0]), 6))
            offset += 8
        elif t == 's':
            s_end = chunk.find(b"\0", offset)
            s = chunk[offset:s_end].decode("utf-8", errors="replace")
            values.append(s)
            offset = (s_end + 4) & ~3
        else:
            break
    return addr, values, offset

def parse_bundle(raw: bytes) -> list[tuple[str, list]]:
    """解析 OSC Bundle, 返回 [(address, values), ...]"""
    results = []
    data = memoryview(raw)
    offset = 0

    # 检查 bundle 头
    if len(data) >= 16 and bytes(data[:8]) == b"#bundle\0":
        offset = 16
    else:
        # 单条消息
        addr, vals, _ = parse_osc_message(raw, 0)
        if addr:
            results.append((addr, vals))
        return results

    while offset < len(data):
        if offset + 4 > len(data):
            break
        el = struct.unpack(">I", data[offset:offset+4])[0]
        offset += 4
        chunk = bytes(data[offset:offset+el])
        offset += el
        addr, vals, _ = parse_osc_message(chunk, 0)
        if addr:
            results.append((addr, vals))
    return results

def main():
    parser = argparse.ArgumentParser(description="Debug ZigSim OSC data")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--duration", type=float, default=0,
                        help="自动停止秒数 (0=手动Ctrl+C)")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", args.port))
        sock.settimeout(1.0)
        print(f"✅ 监听 UDP 0.0.0.0:{args.port}")
    except Exception as e:
        print(f"❌ 绑定失败: {e}")
        return

    print(f"等待 ZigSim 数据... (按 Ctrl+C 停止)\n")

    seen_addresses = {}
    packet_count = 0
    t0 = time.time()

    try:
        while True:
            try:
                raw, addr = sock.recvfrom(65536)
            except socket.timeout:
                if packet_count == 0 and time.time() - t0 > 5:
                    print("⏳ 尚未收到数据, 请确认 ZigSim 正在发送...")
                    t0 = time.time()
                continue

            packet_count += 1
            messages = parse_bundle(raw)

            # 首次 (前 3 个包) 或新地址 → 打印详细信息
            for addr_str, vals in messages:
                key = addr_str
                count = len(vals)
                if key not in seen_addresses:
                    seen_addresses[key] = {"count": 0, "num_values": count, "sample": vals}
                    print(f"  🆕 [{key}] → {count} 个值, 样例: {vals}")
                seen_addresses[key]["count"] += 1

            if packet_count <= 3:
                print(f"  📦 包 #{packet_count}: {len(messages)} 条消息, "
                      f"地址: {[m[0] for m in messages]}")
                print()

            if args.duration > 0 and time.time() - t0 > args.duration:
                break

    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

    print(f"\n{'='*60}")
    print(f"汇总: 共收到 {packet_count} 个 OSC 包")
    print(f"发现 {len(seen_addresses)} 种 OSC 地址:")
    print(f"{'='*60}")
    for addr_str, info in sorted(seen_addresses.items()):
        print(f"  {addr_str:40s} | {info['num_values']} 个值 "
              f"| 出现 {info['count']} 次 "
              f"| 样例: {info['sample']}")
    print(f"{'='*60}")

    # 通道扩展建议
    print(f"\n📊 通道扩展建议:")
    gyro = any("/gyro" in a for a in seen_addresses)
    accel = any("/accel" in a for a in seen_addresses)
    quat = any(("/quat" in a.lower() or "/attitude" in a.lower())
               for a in seen_addresses)
    grav = any(("/grav" in a.lower() or "/gravity" in a.lower())
               for a in seen_addresses)
    mag = any(("/mag" in a.lower() or "/compass" in a.lower())
              for a in seen_addresses)

    total = 6  # gyro(3) + accel(3)
    print(f"  基础: gyro(3) + accel(3) = 6 通道")
    if quat:
        total += 4
        print(f"  ✅ 四元数 (4) → +4 通道, 总计 {total}")
    else:
        print(f"  ❌ 未检测到四元数 — 请在 ZigSim 设置中开启")
    if grav:
        total += 3
        print(f"  ✅ 重力向量 (3) → +3 通道, 总计 {total}")
    else:
        print(f"  ❌ 未检测到重力向量")
    if mag:
        total += 3
        print(f"  ✅ 磁力计 (3) → +3 通道, 总计 {total}")
    print(f"\n  衍生特征 (从现有6轴计算): accel_mag(1) + gyro_mag(1) + jerk(1) = +3")
    print(f"  最大可用: {total + 3} 通道")

if __name__ == "__main__":
    main()
