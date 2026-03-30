import os
import os.path as osp
import argparse
import numpy as np
import cv2
import glob
import bisect

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def aggregate_events_xy_p(xs, ys, ps, H, W):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    if xs.size == 0:
        return img
    pos = ps > 0
    neg = ps == 0
    img[ys[pos], xs[pos]] = [255, 0, 0]  # Blue for positive
    img[ys[neg], xs[neg]] = [0, 0, 255]  # Red for negative
    return img


def carla_to_opencv_cam2world(x, y, z, pitch, yaw, roll):
    """
    Convert CARLA left-handed (X-forward, Y-right, Z-up) pose
    to OpenCV right-handed (X-right, Y-down, Z-forward) cam2world.
    """
    pitch_rad = np.deg2rad(pitch)
    yaw_rad   = np.deg2rad(yaw)
    roll_rad  = np.deg2rad(roll)

    cy, sy = np.cos(yaw_rad),   np.sin(yaw_rad)
    cr, sr = np.cos(roll_rad),  np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)

    # CARLA rotation matrix (camera to world in CARLA frame)
    R_carla = np.array([
        [cp*cy,  cy*sp*sr - sy*cr,  -cy*sp*cr - sy*sr],
        [cp*sy,  sy*sp*sr + cy*cr,  -sy*sp*cr + cy*sr],
        [sp,     -cp*sr,             cp*cr            ]
    ], dtype=np.float32)

    fwd_carla   = R_carla[:, 0]
    right_carla = R_carla[:, 1]
    up_carla    = R_carla[:, 2]

    cv_right_in_carla = right_carla
    cv_down_in_carla  = -up_carla
    cv_fwd_in_carla   = fwd_carla

    def carla_vec_to_cv_world(v):
        return np.array([v[1], -v[2], v[0]], dtype=np.float32)

    col_0 = carla_vec_to_cv_world(cv_right_in_carla)
    col_1 = carla_vec_to_cv_world(cv_down_in_carla)
    col_2 = carla_vec_to_cv_world(cv_fwd_in_carla)

    R_cv = np.column_stack((col_0, col_1, col_2))

    t_carla = np.array([x, y, z], dtype=np.float32)
    t_cv    = carla_vec_to_cv_world(t_carla)

    P_cv = np.eye(4, dtype=np.float32)
    P_cv[:3, :3] = R_cv
    P_cv[:3, 3]  = t_cv
    return P_cv


def find_nearest_idx(sorted_timestamps, query):
    """在已排序的时间戳数组中找最近的索引。"""
    idx = bisect.bisect_left(sorted_timestamps, query)
    if idx == 0:
        return 0
    if idx >= len(sorted_timestamps):
        return len(sorted_timestamps) - 1
    # 比较左右哪个更近
    if abs(sorted_timestamps[idx - 1] - query) <= abs(sorted_timestamps[idx] - query):
        return idx - 1
    return idx


def run(args):
    seq_dirs = sorted(glob.glob(osp.join(args.src, "sequence_*")))
    if not seq_dirs:
        raise RuntimeError(f"No sequence_* directories found in {args.src}")

    print(f"Found {len(seq_dirs)} sequences in {args.src}")

    for seq_path in seq_dirs:
        seq_name = osp.basename(seq_path)
        seq_dst  = osp.join(args.dst, args.src.split("/")[-1] + "-" + seq_name)
        os.makedirs(seq_dst, exist_ok=True)
        print(f"\nProcessing: {seq_name}")

        # ── Paths ────────────────────────────────────────────────────────────
        rgb_dir    = osp.join(seq_path, "rgb",    "data")
        depth_dir  = osp.join(seq_path, "depth",  "frames")
        events_dir = osp.join(seq_path, "events", "frames")

        vehicle_data_dir = osp.join(seq_path, "vehicle_data")
        pos_file  = osp.join(vehicle_data_dir, "position.txt")
        ori_file  = osp.join(vehicle_data_dir, "orientation.txt")
        veh_ts_file = osp.join(vehicle_data_dir, "timestamps.txt")

        # RGB timestamp file (25 Hz, one entry per RGB frame)
        rgb_ts_file = osp.join(seq_path, "rgb", "data", "timestamps.txt")
        # Fallback: some versions store it one level up
        if not osp.exists(rgb_ts_file):
            rgb_ts_file = osp.join(seq_path, "rgb", "timestamps.txt")

        # ── RGB frames ───────────────────────────────────────────────────────
        rgb_files  = sorted(glob.glob(osp.join(rgb_dir, "*.png")))
        num_frames = len(rgb_files)
        print(f"  RGB frames: {num_frames}")

        sample_img = cv2.imread(rgb_files[0])
        H, W = sample_img.shape[:2]

        # ── Intrinsics (CARLA 90° FOV) ───────────────────────────────────────
        focal = W / 2.0
        K_mat = np.array([
            [focal, 0,     W / 2.0],
            [0,     focal, H / 2.0],
            [0,     0,     1      ]
        ], dtype=np.float32)

        # ── Event frames ─────────────────────────────────────────────────────
        event_files = sorted(glob.glob(osp.join(events_dir, "*.png")))
        depth_files = sorted(glob.glob(osp.join(depth_dir,  "*.npy")))

        # ── Vehicle data (1000 Hz) ────────────────────────────────────────────
        positions    = np.loadtxt(pos_file)      # (N_veh, 3)
        orientations = np.loadtxt(ori_file)      # (N_veh, 3)
        veh_timestamps = np.loadtxt(veh_ts_file) # (N_veh,)  seconds

        # ── RGB timestamps (25 Hz) ────────────────────────────────────────────
        if osp.exists(rgb_ts_file):
            raw = np.loadtxt(rgb_ts_file)
            # Format may be "index timestamp" or just "timestamp"
            if raw.ndim == 2:
                rgb_timestamps = raw[:, 1]   # second column
            else:
                rgb_timestamps = raw         # single column
            print(f"  RGB timestamps loaded: {len(rgb_timestamps)} entries, "
                  f"interval ≈ {np.mean(np.diff(rgb_timestamps))*1000:.1f} ms")
        else:
            # Fallback: assume uniform 25 Hz starting from veh_timestamps[0]
            print("  WARNING: RGB timestamps.txt not found, assuming 25 Hz")
            dt = 1.0 / 25.0
            rgb_timestamps = veh_timestamps[0] + np.arange(num_frames) * dt

        # Sanity check
        assert len(rgb_timestamps) >= num_frames, \
            f"RGB timestamps ({len(rgb_timestamps)}) < num_frames ({num_frames})"

        # ── Main loop ─────────────────────────────────────────────────────────
        veh_ts_list = veh_timestamps.tolist()  # for bisect

        for i in range(num_frames):
            frame_id = f"{i:06d}"

            # ── Pose: find nearest vehicle_data entry by RGB timestamp ─────
            rgb_ts = rgb_timestamps[i]
            veh_idx = find_nearest_idx(veh_ts_list, rgb_ts)

            p = positions[veh_idx]      # [x, y, z] in CARLA frame
            o = orientations[veh_idx]   # [pitch, yaw, roll] in degrees

            cam2world = carla_to_opencv_cam2world(
                p[0], p[1], p[2],
                o[0], o[1], o[2]
            ).astype(np.float32)

            # ── RGB ───────────────────────────────────────────────────────────
            img = cv2.imread(rgb_files[i])
            cv2.imwrite(osp.join(seq_dst, f"{frame_id}.png"), img)

            # ── Event ─────────────────────────────────────────────────────────
            ev_img = cv2.imread(event_files[i])
            cv2.imwrite(osp.join(seq_dst, f"{frame_id}_event.png"), ev_img)

            # ── Depth (read .npy, filter invalid) ────────────────────────────
            depth_map = np.load(depth_files[i]).astype(np.float32)
            depth_map[depth_map >= 999.0] = 0.0   # CARLA invalid sentinel
            depth_map[depth_map > 80.0]   = 0.0   # clip far range
            cv2.imwrite(osp.join(seq_dst, f"{frame_id}.exr"), depth_map)

            # ── Camera params ─────────────────────────────────────────────────
            np.savez(osp.join(seq_dst, f"{frame_id}.npz"),
                     intrinsics=K_mat,
                     cam2world=cam2world)

            if i > 0 and i % 50 == 0:
                # Print displacement to verify timestamps are correctly aligned
                prev_data = np.load(osp.join(seq_dst, f"{(i-1):06d}.npz"))
                curr_t = cam2world[:3, 3]
                prev_t = prev_data["cam2world"][:3, 3]
                disp = np.linalg.norm(curr_t - prev_t)
                print(f"  frame {i}/{num_frames}  disp={disp:.4f}m  "
                      f"veh_idx={veh_idx}  rgb_ts={rgb_ts:.3f}s")

        print(f"  Done -> {seq_dst}")

    print("\nEventScape preprocessing completed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True,
                        help="Path to dataset root containing sequence_* dirs")
    parser.add_argument("--dst", type=str, required=True,
                        help="Output directory")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()