import os
import os.path as osp
import argparse
import numpy as np
import cv2
import h5py
import yaml
import tqdm
import hdf5plugin
from pathlib import Path
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as R
import bisect
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def aggregate_events_xy_p(xs, ys, ps, H, W):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    if xs.size == 0:
        return img
    pos = ps > 0
    neg = ps == 0
    img[ys[pos], xs[pos]] = [255, 0, 0]   # Blue for positive
    img[ys[neg], xs[neg]] = [0, 0, 255]   # Red for negative
    return img


def process_subsequence(args, base_name, sub_name):
    seq_name = f"{base_name}_{sub_name}" if sub_name else base_name
    print(f"\n=== 处理子序列: {seq_name} ===")

    seq_dst = osp.join(args.dst, seq_name)
    os.makedirs(seq_dst, exist_ok=True)

    DATA_ROOT = Path(args.src)

    # ====================== 路径 ======================
    img_dir = DATA_ROOT / "RGB_event/train" / seq_name / "images/left/distorted"
    event_h5 = DATA_ROOT / "RGB_event/train" / seq_name / "events/left/events.h5"
    disp_dir = DATA_ROOT / "train_disparity" / seq_name / (
        "disparity/event" if args.use_event_view else "disparity/image")
    calib_file = DATA_ROOT / "train_calibration" / seq_name / "calibration/cam_to_cam.yaml"

    pose_bag_file = DATA_ROOT / "pose" / (base_name + ".bag")

    # ====================== Calibration ======================
    with open(calib_file, 'r') as f:
        calib = yaml.safe_load(f)

    # K
    if args.use_event_view:
        cam_key = 'camRect0' if 'camRect0' in calib['intrinsics'] else 'cam0'
    else:
        cam_key = 'camRect1' if 'camRect1' in calib['intrinsics'] else 'cam1'

    K_list = calib['intrinsics'][cam_key]['camera_matrix']
    if len(K_list) == 4:
        fx, fy, cx, cy = K_list
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    else:
        K = np.array(K_list).reshape(3, 3).astype(np.float32)

    # Q
    q_key = "cams_03" if args.use_event_view else "cams_12"
    Q = np.array(calib['disparity_to_depth'][q_key], dtype=np.float32)

    # ====================== RGB & Event ======================
    rgb_files = sorted(img_dir.glob("*.png"))
    frame_ids = [f.stem for f in rgb_files]
    print(f"找到 {len(frame_ids)} 帧")

    sample = cv2.imread(str(rgb_files[0]))
    H, W = sample.shape[:2]

    with h5py.File(event_h5, 'r') as f:
        events = {
            't': f['events']['t'][:],
            'x': f['events']['x'][:],
            'y': f['events']['y'][:],
            'p': f['events']['p'][:],
        }

    # ====================== 读取时间戳 ======================
    timestamp_file = img_dir.parent / "timestamps.txt"
    with open(timestamp_file, 'r') as f:
        img_timestamps_us = [int(line.strip()) for line in f if line.strip()]

    t_prev_us = img_timestamps_us[0]

    # ====================== 读取 pose.bag ======================
    pose_dict = {}
    pose_times = []
    if pose_bag_file.exists():
        with AnyReader([pose_bag_file]) as reader:
            for conn in reader.connections:
                if conn.topic == "/pose":
                    for _, ts_ns, raw in reader.messages(connections=[conn]):
                        msg = reader.deserialize(raw, conn.msgtype)
                        t_sec = ts_ns / 1e9
                        try:
                            p = msg.pose.pose.position
                            q = msg.pose.pose.orientation
                        except AttributeError:
                            p = msg.pose.position
                            q = msg.pose.orientation
                        pose = np.eye(4, dtype=np.float32)
                        pose[:3, 3] = [p.x, p.y, p.z]
                        rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                        pose[:3, :3] = rot
                        pose_dict[t_sec] = pose
                        pose_times.append(t_sec)
        pose_times.sort()
        print(f"从 pose.bag 读取到 {len(pose_dict)} 条 pose")

    # ====================== 主循环 ======================
    for frame_idx, fid in enumerate(tqdm.tqdm(frame_ids, desc=seq_name)):

        if frame_idx + 1 < len(img_timestamps_us):
            t_curr_us = img_timestamps_us[frame_idx + 1]
        else:
            # 最后一帧：用前一帧的间隔推算
            if len(img_timestamps_us) >= 2:
                t_curr_us = t_prev_us + (img_timestamps_us[-1] - img_timestamps_us[-2])
            else:
                t_curr_us = t_prev_us + 50000

        # Depth
        disp_path = disp_dir / f"{fid}.png"
        if not disp_path.exists():
            t_prev_us = t_curr_us
            continue

        disp_u16 = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)
        if disp_u16 is None:
            t_prev_us = t_curr_us
            continue

        # Pose
        t_sec = t_curr_us / 1e6
        pose = None
        if pose_times:
            bisect_pos = bisect.bisect_left(pose_times, t_sec)
            if bisect_pos == 0:
                best_t = pose_times[0]
            elif bisect_pos == len(pose_times):
                best_t = pose_times[-1]
            else:
                t1 = pose_times[bisect_pos - 1]
                t2 = pose_times[bisect_pos]
                best_t = t1 if abs(t1 - t_sec) < abs(t2 - t_sec) else t2
            pose = pose_dict[best_t]

        if pose is None:
            print(f"警告: {fid} 没有对应 pose，跳过")
            t_prev_us = t_curr_us
            continue

        # 1. RGB
        rgb = cv2.imread(str(img_dir / f"{fid}.png"))
        cv2.imwrite(osp.join(seq_dst, f"{fid}.png"), rgb)

        # 2. Depth — 修复: 不用 clip，保留稀疏有效区域，无效像素设为 0
        disp_f = disp_u16.astype(np.float32) / 256.0
        valid_mask = disp_u16 > 0
        depth = cv2.reprojectImageTo3D(disp_f, Q)[..., 2]
        depth[~valid_mask] = 0.0   # 无效视差 → 0
        depth[depth < 0.1] = 0.0  # 过滤负数和极近点
        depth[depth > 80.0] = 0.0  # 过滤超远距离
        cv2.imwrite(osp.join(seq_dst, f"{fid}.exr"), depth.astype(np.float32))

        # 3. Event
        mask = (events['t'] >= t_prev_us) & (events['t'] < t_curr_us)
        ev_img = aggregate_events_xy_p(
            events['x'][mask], events['y'][mask], events['p'][mask], H, W)
        cv2.imwrite(osp.join(seq_dst, f"{fid}_event.png"), ev_img)

        t_prev_us = t_curr_us

        # 4. 保存 camera params
        np.savez(osp.join(seq_dst, f"{fid}.npz"),
                 intrinsics=K,
                 cam2world=pose)

    print(f"子序列 {seq_name} 处理完成 → {seq_dst}")


def run(args):
    DATA_ROOT = Path(args.src)
    base_name = args.name

    train_img_root = DATA_ROOT / "RGB_event/train"
    sub_dirs = sorted([d for d in train_img_root.iterdir()
                       if d.is_dir() and d.name.startswith(base_name + "_")])

    if not sub_dirs:
        print(f"未找到子序列，尝试直接处理 {base_name}")
        process_subsequence(args, base_name, "")
        return

    print(f"发现 {len(sub_dirs)} 个子序列：")
    for d in sub_dirs:
        print("  -", d.name)

    for sub_dir in sub_dirs:
        sub_name = sub_dir.name.replace(base_name + "_", "")
        process_subsequence(args, base_name, sub_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="DSEC 根目录")
    parser.add_argument("--dst", type=str, required=True, help="输出根目录")
    parser.add_argument("--name", type=str, required=True, help="基序列名，例如 interlaken_00")
    parser.add_argument("--use_event_view", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()