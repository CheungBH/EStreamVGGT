import h5py
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import scipy.ndimage as ndimage

def aggregate_events_xy_p(xs, ys, ps, H, W):
    """蓝正红负 3通道 event 图"""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    if xs.size == 0:
        return img
    pos = ps > 0
    neg = ps == 0
    img[ys[pos], xs[pos]] = [255, 0, 0]   # Blue 正
    img[ys[neg], xs[neg]] = [0, 0, 255]   # Red 负
    return img

def load_pose_gt(path):
    """加载 pose_gt.h5 → {ts: 4x4 pose}"""
    poses = {}
    with h5py.File(path, 'r') as f:
        ts = f['ts'][:].astype(np.float64)
        pose_data = f.get('pose') or f.get('T') or f.get('poses')
        if pose_data is None:
            raise KeyError("pose_gt.h5 中未找到 'pose' / 'T' / 'poses'")
        pose_data = np.array(pose_data)
        if pose_data.ndim == 2:
            pose_data = pose_data.reshape(-1, 4, 4)
        for t, p in zip(ts, pose_data):
            poses[float(t)] = p.astype(np.float32)
    return poses

def get_closest_key(dic, target):
    return min(dic.keys(), key=lambda k: abs(k - target))

def project_depth_to_rgb(depth_event, K_event, T_event_to_rgb, rgb_H, rgb_W):
    """Depth 从 event 投影到 RGB 相机"""
    H_e, W_e = depth_event.shape
    y_e, x_e = np.mgrid[0:H_e, 0:W_e]
    pts = np.stack([x_e.ravel(), y_e.ravel(), np.ones_like(x_e.ravel())], axis=1)
    pts3d = pts * depth_event.ravel()[:, None]
    pts3d_rgb = (T_event_to_rgb[:3, :3] @ pts3d.T + T_event_to_rgb[:3, 3:]).T
    K_rgb = np.array([[K_event[0], 0, K_event[2]],
                      [0, K_event[1], K_event[3]],
                      [0, 0, 1]], dtype=np.float32)
    pts2d = (K_rgb @ pts3d_rgb.T).T
    pts2d = pts2d[:, :2] / (pts2d[:, 2:3] + 1e-8)  # 防除0
    depth_rgb = np.zeros((rgb_H, rgb_W), dtype=np.float32)
    valid = (pts2d[:,0] >= 0) & (pts2d[:,0] < rgb_W) & (pts2d[:,1] >= 0) & (pts2d[:,1] < rgb_H)
    x_valid = pts2d[valid, 0].astype(int)
    y_valid = pts2d[valid, 1].astype(int)
    depth_rgb[y_valid, x_valid] = pts3d_rgb[valid, 2]
    depth_rgb = ndimage.gaussian_filter(depth_rgb, sigma=1)
    return np.clip(depth_rgb, 0, None)

def preprocess_m3ed(seq_name: str,
                    data_root="/home/bhzhang/Documents/datasets/m3ed",
                    out_root="/home/bhzhang/Documents/m3ed_vggt_ready/train"):

    seq_dir = Path(data_root) / seq_name
    h5_path    = seq_dir / f"{seq_name}_data.h5"
    depth_path = seq_dir / f"{seq_name}_depth_gt.h5"
    pose_path  = seq_dir / f"{seq_name}_pose_gt.h5"

    out_scene = Path(out_root) / seq_name
    for sub in ["rgb", "event", "depth", "pose"]:
        (out_scene / sub).mkdir(parents=True, exist_ok=True)

    # 加载主数据
    with h5py.File(h5_path, 'r') as f:
        rgb_data = f['/ovc/rgb/data'][:]
        rgb_ts   = f['/ovc/rgb/ts'][:].astype(np.float64)

        # 关键：官方 event 索引映射（累积索引）
        ts_map_left = f['/ovc/rgb/ts_map_prophesee_left_t'][:].astype(int)

        events = {
            'x': f['/prophesee/left/x'][:],
            'y': f['/prophesee/left/y'][:],
            't': f['/prophesee/left/t'][:].astype(np.float64) / 1e9,  # ns → s
            'p': f['/prophesee/left/p'][:]
        }

        calib_e = f['/prophesee/left/calib']
        resolution = calib_e['resolution'][:]
        H_e, W_e = min(resolution), max(resolution)  # 保险处理顺序
        K_e = calib_e['intrinsics'][:]

        T_cam_to_e = f['/ovc/rgb/calib/T_to_prophesee_left'][:]
        T_e_to_rgb = np.linalg.inv(T_cam_to_e)

        H_rgb, W_rgb = rgb_data.shape[1:3]

    # GT
    with h5py.File(depth_path, 'r') as f:
        depth_ts = f['ts'][:].astype(np.float64)
        depth_maps = f['depth'][:]

    poses_dict = load_pose_gt(pose_path)

    metadata = {"sequence": seq_name, "views": []}

    print(f"🚀 处理 {seq_name} | RGB 帧数: {len(rgb_ts)} | 使用 ts_map_prophesee_left_t 严格对齐 event")

    total_events = len(events['x'])
    prev_end = 0

    for i in tqdm(range(len(rgb_ts)), desc=seq_name):
        ts = rgb_ts[i]

        # RGB 保存（转 RGB → 存 BGR）
        rgb = cv2.cvtColor(rgb_data[i], cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(out_scene / "rgb" / f"rgb_{i:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # Event：使用 ts_map 索引（官方对齐）
        start_idx = prev_end
        end_idx   = ts_map_left[i]

        # 边界检查
        end_idx = min(end_idx, total_events)
        if end_idx <= start_idx:
            # 极少见：无新事件 → 空图
            event_img = np.zeros((H_e, W_e, 3), dtype=np.uint8)
        else:
            xs = events['x'][start_idx:end_idx]
            ys = events['y'][start_idx:end_idx]
            ps = events['p'][start_idx:end_idx]
            event_img = aggregate_events_xy_p(xs, ys, ps, H_e, W_e)

        cv2.imwrite(str(out_scene / "event" / f"event_{i:06d}.png"), event_img)

        # 更新 prev_end
        prev_end = end_idx

        # Depth（投影）
        idx_d = np.argmin(np.abs(depth_ts - ts))
        depth_e = depth_maps[idx_d].astype(np.float32)
        depth_rgb = project_depth_to_rgb(depth_e, K_e, T_e_to_rgb, H_rgb, W_rgb)
        cv2.imwrite(str(out_scene / "depth" / f"depth_{i:06d}.png"), (depth_rgb * 1000).astype(np.uint16))

        # Pose（最近）
        closest_ts = get_closest_key(poses_dict, ts)
        pose = poses_dict[closest_ts]
        np.savetxt(out_scene / "pose" / f"pose_{i:06d}.txt", pose)

        metadata["views"].append({
            "idx": i,
            "ts": float(ts),
            "rgb": f"rgb_{i:06d}.png",
            "event": f"event_{i:06d}.png",
            "depth": f"depth_{i:06d}.png",
            "pose": f"pose_{i:06d}.txt"
        })

        # 验证打印（每 1000 帧）
        if i % 1000 == 0 or i == len(rgb_ts)-1:
            events_this = end_idx - start_idx
            print(f"  帧 {i:06d} | RGB ts={ts:.6f} | events={events_this} | 累计 event idx={end_idx}/{total_events}")

    # 保存 metadata + calib
    calib_info = {
        "K_event": K_e.tolist(),
        "T_e_to_rgb": T_e_to_rgb.tolist(),
        "H_rgb": int(H_rgb), "W_rgb": int(W_rgb),
        "H_event": int(H_e), "W_event": int(W_e)
    }
    with open(out_scene / "metadata.json", "w") as f:
        json.dump({"sequence": seq_name, "calib": calib_info, "views": metadata["views"]}, f, indent=2)

    print(f"✅ {seq_name} 完成！输出帧数 = {len(rgb_ts)}（RGB/Event/Depth/Pose 严格对齐）")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M3ED → VGGT 四模态预处理（使用 ts_map 严格对齐）")
    parser.add_argument("--seq", nargs="+", default=["falcon_forest_into_forest_1"])
    parser.add_argument("--data_root", default="/home/bhzhang/Documents/datasets/m3ed")
    parser.add_argument("--out_root", default="/home/bhzhang/Documents/m3ed_vggt_ready/train")
    args = parser.parse_args()

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    for s in args.seq:
        preprocess_m3ed(s, args.data_root, args.out_root)