from rosbags.highlevel import AnyReader
from pathlib import Path
import csv
import tqdm

# ==================== 配置 ====================
BAG_PATH = Path("/home/bhzhang/Documents/tools/dvs_mcemvs/data/DSEC/interlaken_00-odometry/pose.bag")  # 你的实际路径
OUTPUT_TXT = Path("interlaken_00_pose_gt.txt")  # 输出文件

POSE_TOPIC = "/pose"  # 从你输出确认的 topic

# ==================== 提取 ====================
with AnyReader([BAG_PATH]) as reader:
    connections = [c for c in reader.connections if c.topic == POSE_TOPIC]

    if not connections:
        print(f"未找到 topic: {POSE_TOPIC}")
        print("可用 topics：")
        for c in reader.connections:
            print(f" - {c.topic} ({c.msgtype})")
        exit(1)

    conn = connections[0]
    print(f"开始提取：topic = {conn.topic} ({conn.msgtype})，共 {conn.msgcount} 条消息")

    with open(OUTPUT_TXT, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        count = 0

        for _, timestamp_ns, rawdata in tqdm.tqdm(reader.messages(connections=[conn]), total=conn.msgcount):
            msg = reader.deserialize(rawdata, conn.msgtype)

            # 时间戳（秒，保留9位小数，与 DSEC 微秒级兼容）
            t_sec = timestamp_ns / 1e9

            # PoseStamped 结构：msg.header.stamp + msg.pose
            p = msg.pose.position
            q = msg.pose.orientation

            # TUM 格式：timestamp tx ty tz qx qy qz qw
            row = [
                f"{t_sec:.9f}",
                f"{p.x:.6f}",
                f"{p.y:.6f}",
                f"{p.z:.6f}",
                f"{q.x:.6f}",
                f"{q.y:.6f}",
                f"{q.z:.6f}",
                f"{q.w:.6f}"
            ]
            writer.writerow(row)
            count += 1

        print(f"提取完成！共 {count} 条 pose，保存到：{OUTPUT_TXT}")

print("文件格式：TUM style (timestamp tx ty tz qx qy qz qw)")
print("坐标系：通常为 ENU (x前, y左, z上)，与 DSEC event 相机对齐")