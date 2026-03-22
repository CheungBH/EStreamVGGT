import os
import os.path as osp
import argparse
import numpy as np
import cv2
import hdf5plugin
import h5py
import yaml
import tqdm
from pathlib import Path
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as R

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def aggregate_events_xy_p(xs, ys, ps, H, W):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    if xs.size == 0:
        return img

    pos = ps > 0
    neg = ps == 0

    img[ys[pos], xs[pos]] = [255, 0, 0]     # Blue for positive
    img[ys[neg], xs[neg]] = [0, 0, 255]     # Red for negative

    return img


class SimpleIMUIntegrator:
    """简单 IMU 积分（仅短期可用，长时漂移严重）"""

    def __init__(self, g=9.81):
        self.q = np.array([1.0, 0, 0, 0])  # w x y z
        self.v = np.zeros(3)
        self.p = np.zeros(3)
        self.g = np.array([0, 0, -g])
        self.last_t_sec = None

    def update(self, t_sec, acc, gyro):
        if self.last_t_sec is None:
            self.last_t_sec = t_sec
            return np.eye(4)

        dt = t_sec - self.last_t_sec
        if dt <= 0:
            return self.get_pose()

        # 1. 姿态更新（简单积分）
        angle = gyro * dt
        if np.linalg.norm(angle) > 1e-8:
            delta_rot = R.from_rotvec(angle)
            current_rot = R.from_quat(self.q[[1, 2, 3, 0]])  # xyzw
            new_rot = current_rot * delta_rot
            self.q = new_rot.as_quat()[[3, 0, 1, 2]]  # wxyz

        # 2. 加速度转世界系
        R_mat = R.from_quat(self.q[[1, 2, 3, 0]]).as_matrix()
        acc_world = R_mat @ acc

        # 3. 减重力
        acc_linear = acc_world + self.g

        # 4. 速度 & 位置
        v_old = self.v.copy()
        self.v += acc_linear * dt
        self.p += (v_old + self.v) / 2 * dt

        self.last_t_sec = t_sec
        return self.get_pose()

    def get_pose(self):
        R_mat = R.from_quat(self.q[[1, 2, 3, 0]]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = self.p
        return T.astype(np.float32)


def process_subsequence(args, base_name, sub_name):
    """处理单个子序列，如 interlaken_00_a"""
    seq_name = f"{base_name}_{sub_name}" if sub_name else base_name
    print(f"\n处理子序列: {seq_name}")

    seq_dst = osp.join(args.dst, seq_name)
    os.makedirs(seq_dst, exist_ok=True)

    DATA_ROOT = Path(args.src)

    # 路径（根据你的结构调整）
    img_dir = DATA_ROOT / "RGB_event/train" / seq_name / "images/left/distorted"
    event_h5 = DATA_ROOT / "RGB_event/train" / seq_name / "events/left/events.h5"
    disp_dir = DATA_ROOT / "train_disparity" / seq_name / (
        "disparity/event" if args.use_event_view else "disparity/image")
    calib_file = DATA_ROOT / "train_calibration" / seq_name / "calibration/cam_to_cam.yaml"

    # bag 是基序列的
    bag_dir = DATA_ROOT / "lidar_imu" / "data" / base_name
    bag_file = list(bag_dir.glob("*.bag"))[0] if bag_dir.exists() else None

    # 加载 calibration
    if not calib_file.exists():
        print(f"警告：calib 文件不存在 {calib_file}，使用单位内参")
        K = np.eye(3, dtype=np.float32)
        Q = np.zeros((4, 4), dtype=np.float32)
    else:
        with open(calib_file, 'r') as f:
            calib = yaml.safe_load(f)

        # 内参 K
        if args.use_event_view:
            cam_key = 'camRect0' if 'camRect0' in calib['intrinsics'] else 'cam0'
        else:
            cam_key = 'camRect1' if 'camRect1' in calib['intrinsics'] else 'cam1'

        if cam_key in calib['intrinsics']:
            K_list = calib['intrinsics'][cam_key]['camera_matrix']
            print(f"使用相机内参: {cam_key}, 值: {K_list}")

            if len(K_list) == 4:
                fx, fy, cx, cy = K_list
                K = np.array([
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float32)
            elif len(K_list) == 9:
                K = np.array(K_list).reshape(3, 3).astype(np.float32)
            else:
                raise ValueError(f"不支持的 camera_matrix 长度: {len(K_list)}，内容: {K_list}")
        else:
            print(f"警告：未找到 {cam_key} 内参，使用单位矩阵")
            K = np.eye(3, dtype=np.float32)

        # Q
        q_key = "cams_03" if args.use_event_view else "cams_12"
        if 'disparity_to_depth' in calib and q_key in calib['disparity_to_depth']:
            Q = np.array(calib['disparity_to_depth'][q_key], dtype=np.float32)
        else:
            print(f"警告：未找到 {q_key}，depth 将不可用")
            Q = np.zeros((4, 4), dtype=np.float32)

    # RGB 文件列表
    rgb_files = sorted(img_dir.glob("*.png"))
    if not rgb_files:
        print(f"无 RGB 图像: {img_dir}，跳过")
        return

    frame_ids = [f.stem for f in rgb_files]
    print(f"找到 {len(frame_ids)} 帧")

    # 取第一张图确定分辨率
    sample_rgb = cv2.imread(str(rgb_files[0]))
    if sample_rgb is None:
        print("无法读取 RGB 示例图，跳过")
        return
    H, W = sample_rgb.shape[:2]

    # 加载 events
    if not event_h5.exists():
        print(f"无 event 文件: {event_h5}，event 将为空")
        events = {'t': np.array([]), 'x': np.array([]), 'y': np.array([]), 'p': np.array([])}
    else:
        with h5py.File(event_h5, 'r') as f:
            events = {
                't': f['events']['t'][:],
                'x': f['events']['x'][:],
                'y': f['events']['y'][:],
                'p': f['events']['p'][:],
            }

    # IMU 积分器
    imu_integrator = SimpleIMUIntegrator()

    # bag 中读取 IMU
    bag_reader = None
    connections_imu = []
    if bag_file:
        try:
            bag_reader = AnyReader([bag_file])
            bag_reader.__enter__()
            connections_imu = [c for c in bag_reader.connections if c.topic == '/imu/data']
            if connections_imu:
                print("找到 IMU topic /imu/data，将进行积分")
            else:
                print("未找到 IMU topic /imu/data，pose 将为 identity")
        except Exception as e:
            print(f"打开 bag 失败: {e}，pose 将为 identity")
            bag_reader = None

    # 主循环
    t_prev_us = events['t'][0] if len(events['t']) > 0 else 0
    event_window_us = int(args.event_window_ms * 1000)

    for idx, fid in enumerate(tqdm.tqdm(frame_ids, desc=seq_name)):
        # 1. RGB
        rgb_path = img_dir / f"{fid}.png"
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            continue
        cv2.imwrite(osp.join(seq_dst, f"{fid}.png"), rgb)

        # 2. Depth
        disp_path = disp_dir / f"{fid}.png"
        if disp_path.exists():
            disp_u16 = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)
            disp_f = disp_u16.astype(np.float32) / 256.0
            valid = disp_u16 > 0
            points_3d = cv2.reprojectImageTo3D(disp_f, Q)
            depth = points_3d[..., 2]
            depth[~valid] = 0
            depth = np.clip(depth, 0.1, 80.0)
            cv2.imwrite(osp.join(seq_dst, f"{fid}.exr"), depth.astype(np.float32))
        else:
            cv2.imwrite(osp.join(seq_dst, f"{fid}.exr"), np.zeros((H, W), dtype=np.float32))

        # 3. Event
        t_curr_us = t_prev_us + event_window_us
        mask = (events['t'] >= t_prev_us) & (events['t'] < t_curr_us)
        ev_img = aggregate_events_xy_p(events['x'][mask], events['y'][mask], events['p'][mask], H, W)
        cv2.imwrite(osp.join(seq_dst, f"{fid}_event.png"), ev_img)
        t_prev_us = t_curr_us

        # 4. Pose（IMU 积分） - 关键修复：使用 int64 避免溢出
        pose = imu_integrator.get_pose()

        if bag_reader and connections_imu:
            min_dt = float('inf')
            closest_msg = None

            t_curr_us_int64 = np.int64(t_curr_us)

            for conn, timestamp_ns, raw in bag_reader.messages(connections=connections_imu):
                t_msg_ns_int64 = np.int64(timestamp_ns)
                t_msg_us_int64 = t_msg_ns_int64 // np.int64(1000)

                dt_int64 = abs(t_msg_us_int64 - t_curr_us_int64)
                dt = float(dt_int64)

                if dt < min_dt:
                    min_dt = dt
                    closest_msg = bag_reader.deserialize(raw, conn.msgtype)

                if dt < 5000:
                    break

            if closest_msg and min_dt < 5000:
                acc = np.array([closest_msg.linear_acceleration.x,
                                closest_msg.linear_acceleration.y,
                                closest_msg.linear_acceleration.z], dtype=np.float64)
                gyro = np.array([closest_msg.angular_velocity.x,
                                 closest_msg.angular_velocity.y,
                                 closest_msg.angular_velocity.z], dtype=np.float64)
                t_sec = float(t_curr_us) / 1e6
                pose = imu_integrator.update(t_sec, acc, gyro)

        # 保存
        np.savez(osp.join(seq_dst, f"{fid}.npz"),
                 intrinsics=K,
                 cam2world=pose)

    if bag_reader:
        bag_reader.__exit__(None, None, None)

    print(f"子序列 {seq_name} 处理完成 → {seq_dst}")


def run(args):
    DATA_ROOT = Path(args.src)
    base_name = args.name

    # 自动发现子序列
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
    parser = argparse.ArgumentParser(description="DSEC 数据预处理为 EStreamVGGT 格式（支持多子序列）")
    parser.add_argument("--src", type=str, required=True, help="DSEC 根目录")
    parser.add_argument("--dst", type=str, required=True, help="输出根目录")
    parser.add_argument("--name", type=str, required=True, help="基序列名，例如 interlaken_00 或 zurich_city_09")
    parser.add_argument("--event_window_ms", type=float, default=50.0, help="事件累积窗口 ms")
    parser.add_argument("--use_event_view", action="store_true", help="使用 event 视图的 disparity (cams_03)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()