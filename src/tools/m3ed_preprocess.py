import os
import os.path as osp
import argparse
import numpy as np
import cv2
import h5py
import glob
from pathlib import Path
from tqdm import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

H_EVT, W_EVT = 720, 1280  # prophesee/left resolution


def aggregate_events(xs, ys, ps, H, W):
    """Aggregate events into an RGB image (Blue=positive, Red=negative)."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    if xs.size == 0:
        return img
    pos = ps > 0
    neg = ~pos
    img[ys[pos], xs[pos]] = [255, 0, 0]  # Blue for positive
    img[ys[neg], xs[neg]] = [0, 0, 255]  # Red for negative
    return img


def process_sequence(seq_dir, dst_dir):
    seq_name = osp.basename(seq_dir)
    print(f"\n=== Processing: {seq_name} ===")

    seq_dst = osp.join(dst_dir, seq_name)
    os.makedirs(seq_dst, exist_ok=True)

    # ── File paths ────────────────────────────────────────────────────────────
    data_h5_path   = osp.join(seq_dir, f"{seq_name}_data.h5")
    depth_h5_path  = osp.join(seq_dir, f"{seq_name}_depth_gt.h5")
    pose_h5_path   = osp.join(seq_dir, f"{seq_name}_pose_gt.h5")

    for p in [data_h5_path, depth_h5_path, pose_h5_path]:
        if not osp.exists(p):
            print(f"  Missing: {p}, skipping.")
            return

    with h5py.File(data_h5_path,  'r') as fdata, \
         h5py.File(depth_h5_path, 'r') as fdepth, \
         h5py.File(pose_h5_path,  'r') as fpose:

        # ── Pose (Cn_T_C0): world2cam relative to frame 0 ────────────────────
        # Cn_T_C0[i] takes a point from C0 frame to Cn frame (world2cam)
        # cam2world = inv(Cn_T_C0)
        Cn_T_C0 = fpose['Cn_T_C0'][:]          # (N, 4, 4), float64
        depth_ts = fdepth['ts'][:]              # (N,), microseconds int64
        N = len(depth_ts)
        print(f"  Depth/Pose frames: {N}")

        # ── Depth ─────────────────────────────────────────────────────────────
        depths = fdepth['depth/prophesee/left']  # (N, 720, 1280), lazy-load

        # ── Intrinsics (prophesee/left): [fx, fy, cx, cy] ────────────────────
        intr = fdata['prophesee/left/calib/intrinsics'][:]  # (4,)
        fx, fy, cx, cy = float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3])
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float32)

        # ── RGB frames ────────────────────────────────────────────────────────
        rgb_data = fdata['ovc/rgb/data']         # (M, 800, 1280, 3)
        rgb_ts   = fdata['ovc/ts'][:]            # (M,), microseconds int64
        M = len(rgb_ts)
        print(f"  RGB frames: {M}")

        # ── Events (prophesee/left) ───────────────────────────────────────────
        evt_t      = fdata['prophesee/left/t']           # lazy, int64 us
        evt_x      = fdata['prophesee/left/x']           # lazy, uint16
        evt_y      = fdata['prophesee/left/y']           # lazy, uint16
        evt_p      = fdata['prophesee/left/p']           # lazy, int8
        ms_map_idx = fdata['prophesee/left/ms_map_idx']  # (T_ms,), event idx per ms

        n_events_total = len(evt_t)
        n_ms = len(ms_map_idx)
        print(f"  Total events: {n_events_total:,}")

        # ── Pre-build nearest RGB index for each depth frame ──────────────────
        # For each depth ts, find nearest RGB frame
        rgb_ts_sorted = rgb_ts  # already sorted
        nearest_rgb = np.searchsorted(rgb_ts_sorted, depth_ts, side='left')
        # clamp and pick closest
        nearest_rgb = np.clip(nearest_rgb, 0, M - 1)
        for i in range(N):
            if nearest_rgb[i] > 0:
                if abs(rgb_ts[nearest_rgb[i] - 1] - depth_ts[i]) < \
                   abs(rgb_ts[nearest_rgb[i]]     - depth_ts[i]):
                    nearest_rgb[i] -= 1

        # ── Main loop ─────────────────────────────────────────────────────────
        for i in tqdm(range(N), desc=seq_name):
            frame_id = f"{i:06d}"

            # 1. Pose → cam2world = inv(Cn_T_C0[i])
            w2c = Cn_T_C0[i].astype(np.float32)   # world2cam (relative to frame 0)
            c2w = np.linalg.inv(w2c).astype(np.float32)  # cam2world

            # 2. Depth
            depth = depths[i].astype(np.float32)   # (720, 1280)
            # M3ED depth is in meters, clip to valid range
            depth[depth <= 0] = 0.0  # 先过滤无效
            depth[depth > 80.0] = 0.0  # 过滤超远距离，与其他数据集对齐

            # 3. RGB (nearest frame)
            rgb_idx = nearest_rgb[i]
            rgb = rgb_data[rgb_idx]                # (800, 1280, 3), uint8
            # Resize RGB to match event camera resolution (720, 1280)
            rgb_resized = cv2.resize(rgb, (W_EVT, H_EVT), interpolation=cv2.INTER_LINEAR)

            # 4. Events (window: [ts[i-1], ts[i]) or [ts[i], ts[i]+window])
            # 事件窗口：始终用相邻两帧的真实时间戳
            if i == 0:
                if N > 1:
                    frame_interval = int(depth_ts[1] - depth_ts[0])
                else:
                    frame_interval = 50000
                t_start_us = max(0, int(depth_ts[0]) - frame_interval)
                t_end_us   = int(depth_ts[0])
            else:
                t_start_us = int(depth_ts[i - 1])
                t_end_us   = int(depth_ts[i])

            # Use ms_map_idx for efficient slicing
            t_start_ms = int(t_start_us // 1000)
            t_end_ms   = int(t_end_us   // 1000)
            t_start_ms = max(0, min(t_start_ms, n_ms - 1))
            t_end_ms   = max(0, min(t_end_ms,   n_ms - 1))

            idx_start = int(ms_map_idx[t_start_ms])
            idx_end   = int(ms_map_idx[t_end_ms]) if t_end_ms < n_ms else n_events_total

            if idx_end > idx_start:
                xs = evt_x[idx_start:idx_end].astype(np.int32)
                ys = evt_y[idx_start:idx_end].astype(np.int32)
                ps = evt_p[idx_start:idx_end].astype(np.int32)
                # Fine-grained filter within the ms window
                ts_slice = evt_t[idx_start:idx_end]
                mask = (ts_slice >= t_start_us) & (ts_slice < t_end_us)
                ev_img = aggregate_events(xs[mask], ys[mask], ps[mask], H_EVT, W_EVT)
            else:
                ev_img = np.zeros((H_EVT, W_EVT, 3), dtype=np.uint8)

            # ── Save ──────────────────────────────────────────────────────────
            # RGB
            cv2.imwrite(osp.join(seq_dst, f"{frame_id}.png"), rgb_resized)
            # Event
            cv2.imwrite(osp.join(seq_dst, f"{frame_id}_event.png"), ev_img)
            # Depth
            cv2.imwrite(osp.join(seq_dst, f"{frame_id}.exr"), depth)
            # Camera params
            np.savez(osp.join(seq_dst, f"{frame_id}.npz"),
                     intrinsics=K,
                     cam2world=c2w)

    print(f"  Done → {seq_dst}  ({N} frames)")


def run(args):
    src = Path(args.src)

    if args.name:
        # Process single sequence
        seq_dirs = [src / args.name]
    else:
        # Process all sequences under src
        seq_dirs = sorted([d for d in src.iterdir() if d.is_dir()])

    print(f"Found {len(seq_dirs)} sequence(s)")
    for seq_dir in seq_dirs:
        process_sequence(str(seq_dir), args.dst)

    print("\nAll done.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess M3ED dataset")
    parser.add_argument("--src", type=str, required=True,
                        help="M3ED root dir (contains sequence folders)")
    parser.add_argument("--dst", type=str, required=True,
                        help="Output root dir")
    parser.add_argument("--name", type=str, default="",
                        help="Single sequence name (e.g. falcon_forest_into_forest_1). "
                             "If empty, process all sequences under --src.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()