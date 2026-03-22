import cv2
import numpy as np
import h5py
import yaml
from pathlib import Path
import hdf5plugin
import tqdm
from rosbags.highlevel import AnyReader  # 这个是关键！高层次读取器
from scipy.spatial.transform import Rotation as R  # 用于 quaternion → matrix

# ==================== 配置 ====================
DATA_ROOT = Path("/home/bhzhang/Documents/datasets/DSEC")          # 你的 DSEC 根目录
SEQ_NAME = "zurich_city_09_a"                   # 你的序列名
OUTPUT_DIR = Path(f"output_{SEQ_NAME}")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

USE_EVENT_VIEW = False
DISP_FOLDER = "disparity/event" if USE_EVENT_VIEW else "disparity/image"

# 路径（根据你的解压结构调整）
img_dir = DATA_ROOT / "RGB_event/train" / SEQ_NAME / "images/left/distorted"
event_h5 = DATA_ROOT / "RGB_event/train" / SEQ_NAME / "events/left/events.h5"
disp_dir = DATA_ROOT / "train_disparity" / SEQ_NAME / DISP_FOLDER
calib_file = DATA_ROOT / "train_calibration" / SEQ_NAME / "calibration/cam_to_cam.yaml"

base_seq = SEQ_NAME.rsplit('_', 1)[0] if '_' in SEQ_NAME[-2:] else SEQ_NAME
bag_dir = DATA_ROOT / "lidar_imu" / "data" / base_seq
bag_file = list(bag_dir.glob("*.bag"))[0] if bag_dir.exists() else None

# 加载 Q
with open(calib_file, 'r') as f:
    calib = yaml.safe_load(f)
q_key = "cams_03" if USE_EVENT_VIEW else "cams_12"
Q = np.array(calib['disparity_to_depth'][q_key], dtype=np.float32)

# RGB 文件列表
rgb_files = sorted(img_dir.glob("*.png"))
frame_ids = [f.stem for f in rgb_files]
print(f"Found {len(frame_ids)} frames")

# 事件加载（一次性或分块，根据内存）
with h5py.File(event_h5, 'r') as f:
    events = {
        't': f['events']['t'][:],
        'x': f['events']['x'][:],
        'y': f['events']['y'][:],
        'p': f['events']['p'][:],
    }
    # t_offset = f.get('t_offset', [0])[0]
    # events['t'] += t_offset

# 事件累积函数（示例：固定 50ms 窗口，实际应从 timestamps.txt 取 t）
def accumulate_events(t_start_us, t_end_us, shape=(480, 640)):
    mask = (events['t'] >= t_start_us) & (events['t'] < t_end_us)
    if not np.any(mask):
        return np.zeros((*shape, 3), np.uint8)
    x, y, p = events['x'][mask], events['y'][mask], events['p'][mask]
    pos = np.zeros(shape, np.uint8)
    neg = np.zeros(shape, np.uint8)
    pos[y[p==1], x[p==1]] = 255
    neg[y[p==0], x[p==0]] = 255
    return np.stack([pos, neg, np.zeros_like(pos)], -1)


# ==================== ROS bag 处理 ====================
pose_topic = '/lio_sam/mapping/odometry'  # 先用命令 rosbags-info your.bag 确认实际 topic 名！！！
# 常见 DSEC pose topic：/lio_pose, /odometry, /gps/odometry, /pose 等

bag_reader = None
connections_pose = []  # 提前过滤 connection

if bag_file:
    bag_reader = AnyReader([bag_file])
    bag_reader.__enter__()

    # 关键：过滤出对应 topic 的 Connection 对象
    connections_pose = [c for c in bag_reader.connections if c.topic == pose_topic]

    if not connections_pose:
        print(f"警告：未找到 topic {pose_topic}，pose 将为 identity")
        bag_reader.__exit__(None, None, None)
        bag_reader = None

# 主循环
t_prev_us = events['t'][0] if len(events['t']) > 0 else 0

for idx, fid in enumerate(tqdm.tqdm(frame_ids)):
    # RGB
    rgb_path = img_dir / f"{fid}.png"
    rgb = cv2.imread(str(rgb_path))
    if rgb is None: continue
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Depth
    disp_path = disp_dir / f"{fid}.png"
    if not disp_path.exists(): continue
    disp_u16 = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)
    disp_f = disp_u16.astype(np.float32) / 256.0
    valid = disp_u16 > 0
    points_3d = cv2.reprojectImageTo3D(disp_f, Q)
    depth = points_3d[..., 2]
    depth[~valid] = 0
    depth = np.clip(depth, 0.1, 80.0)

    # Event frame (50ms 示例)
    t_curr_us = t_prev_us + 50000
    event_frame = accumulate_events(t_prev_us, t_curr_us)
    t_prev_us = t_curr_us

    # Pose
    pose = np.eye(4, dtype=np.float32)
    if bag_reader and connections_pose:
        min_dt = float('inf')
        closest_msg = None

        for connection, timestamp_ns, rawdata in bag_reader.messages(connections=connections_pose):
            msg = bag_reader.deserialize(rawdata, connection.msgtype)
            t_msg_us = timestamp_ns // 1000  # ns → us
            dt = abs(t_msg_us - t_curr_us)
            if dt < min_dt:
                min_dt = dt
                closest_msg = msg
            if min_dt < 5000:  # 5ms 内
                break

        if closest_msg and min_dt < 5000:
            # 假设是 nav_msgs.msg.Odometry
            try:
                p = closest_msg.pose.pose.position
                q = closest_msg.pose.pose.orientation
            except AttributeError:
                # 备选：geometry_msgs.msg.PoseStamped
                try:
                    p = closest_msg.pose.position
                    q = closest_msg.pose.orientation
                except AttributeError:
                    print(f"Pose msg 格式未知，跳过: {type(closest_msg)}")
                    continue

            pose[0:3, 3] = [p.x, p.y, p.z]
            rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            pose[0:3, 0:3] = rot

    # 保存
    out_h5 = OUTPUT_DIR / f"frame_{fid}.h5"
    with h5py.File(out_h5, 'w') as hf:
        hf.create_dataset('rgb', data=rgb, compression="gzip")
        hf.create_dataset('event_frame', data=event_frame, compression="gzip")
        hf.create_dataset('depth', data=depth, compression="gzip")
        hf.create_dataset('pose', data=pose)

if bag_reader:
    bag_reader.__exit__(None, None, None)

print(f"完成！输出文件夹：{OUTPUT_DIR}")