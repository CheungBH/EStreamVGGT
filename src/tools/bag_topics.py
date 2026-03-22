from rosbags.highlevel import AnyReader
from pathlib import Path

bag_path = Path("/home/bhzhang/Documents/tools/dvs_mcemvs/data/DSEC/interlaken_00-odometry/pose.bag")

with AnyReader([bag_path]) as reader:
    conn = [c for c in reader.connections if c.topic == "/pose"][0]
    timestamps = []
    for _, ts_ns, _ in reader.messages(connections=[conn]):
        timestamps.append(ts_ns / 1e9)  # 转秒
    print(f"Pose bag 时间范围: {min(timestamps):.3f} ~ {max(timestamps):.3f} 秒")
    print(f"总时长: {max(timestamps) - min(timestamps):.1f} 秒")
    print(f"第一条时间戳示例: {timestamps[0]:.9f}")