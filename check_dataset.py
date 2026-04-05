# python
import os
import os.path as osp
import numpy as np
import cv2

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# =====================================================================
# Configuration: fill in your actual paths (can be the sequence directory
# itself or its parent directory)
# =====================================================================
DATASETS = {
    "Waymo": {
        "root": "/media/bhzhang/ACER/dataset/waymo_sample_processed_stream/validation/segment-14333744981238305769_5658_260_5678_260_with_camera_labels.tfrecord",
    },
    "EventScape": {
        "root": "/home/bhzhang/Documents/datasets/streamVGGT/eventscape/train/Town01-sequence_0",
    },
    "DSEC": {
        "root": "/home/bhzhang/Documents/datasets/streamVGGT/DSEC/train/interlaken_00_c",
    },
    "M3ED": {
        "root": "/home/bhzhang/Documents/datasets/streamVGGT/m3ed/train/falcon_forest_into_forest_1",
    },
}

N_FRAMES_PER_SEQ = 5  # Number of frames to sample per sequence for checking
# =====================================================================


def collect_frames(root):
    """
    Supports two structures:
      1. root is the sequence directory itself (contains .exr/.npz files)
      2. root is the parent directory (contains multiple sequence subdirectories)
    """
    files_in_root = os.listdir(root)
    has_exr_directly = any(f.endswith(".exr") for f in files_in_root)

    if has_exr_directly:
        candidate_dirs = [root]
    else:
        candidate_dirs = sorted([
            osp.join(root, d) for d in files_in_root
            if osp.isdir(osp.join(root, d))
        ])

    all_seqs = []
    for seq_dir in candidate_dirs:
        exr_files = sorted([f for f in os.listdir(seq_dir) if f.endswith(".exr")])
        pairs = []
        for exr in exr_files:
            base = exr[:-4]
            npz = base + ".npz"
            if osp.exists(osp.join(seq_dir, npz)):
                pairs.append((osp.join(seq_dir, exr), osp.join(seq_dir, npz)))
        if pairs:
            all_seqs.append((osp.basename(seq_dir), pairs))
    return all_seqs


def check_depth(exr_path):
    d = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    valid = d[d > 0]
    if len(valid) == 0:
        return {"min": 0, "max": 0, "mean": 0, "valid_ratio": 0, "shape": d.shape}
    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "valid_ratio": float(len(valid) / d.size),
        "shape": d.shape,
    }


def check_pose(npz_path):
    data = np.load(npz_path)
    c2w = data["cam2world"]
    Rmat = c2w[:3, :3]
    t = c2w[:3, 3]
    det = float(np.linalg.det(Rmat))
    ortho_err = float(np.max(np.abs(Rmat @ Rmat.T - np.eye(3))))
    return {
        "translation": t,
        "det_R": det,
        "ortho_err": ortho_err,
        "intrinsics": data["intrinsics"],
    }


def check_pose_continuity(pairs):
    diffs = []
    prev_t = None
    for _, npz_path in pairs[:30]:
        data = np.load(npz_path)
        t = data["cam2world"][:3, 3]
        if prev_t is not None:
            diffs.append(float(np.linalg.norm(t - prev_t)))
        prev_t = t
    return diffs


def print_separator(name):
    print("\n" + "=" * 60)
    print(f"  {name}")
    print("=" * 60)


def main():
    for ds_name, cfg in DATASETS.items():
        root = cfg["root"]
        print_separator(ds_name)

        if not osp.exists(root):
            print(f"  X Path does not exist: {root}")
            continue

        all_seqs = collect_frames(root)
        if not all_seqs:
            print(f"  X No sequences containing exr+npz were found")
            continue

        print(f"  Found {len(all_seqs)} sequence(s)")

        depth_stats_all = []
        pose_issues = []

        for seq_name, pairs in all_seqs:
            print(f"\n  [seq] {seq_name}  ({len(pairs)} frames)")
            sample_pairs = pairs[:N_FRAMES_PER_SEQ]

            # Depth
            print(f"    [Depth]")
            for exr_path, _ in sample_pairs:
                stats = check_depth(exr_path)
                if stats is None:
                    print(f"      X Failed to read: {osp.basename(exr_path)}")
                    continue
                if stats["valid_ratio"] < 0.01:
                    flag = "WARN"
                elif stats["mean"] > 150:
                    flag = "WARN"
                else:
                    flag = "OK"
                print(f"      [{flag}] {osp.basename(exr_path)}: "
                      f"min={stats['min']:.2f}m  max={stats['max']:.2f}m  "
                      f"mean={stats['mean']:.2f}m  valid={stats['valid_ratio']:.3f}  "
                      f"shape={stats['shape']}")
                depth_stats_all.append(stats)

            # Pose
            print(f"    [Pose]")
            for _, npz_path in sample_pairs:
                p = check_pose(npz_path)
                det_ok = abs(p["det_R"] - 1.0) < 1e-3
                ortho_ok = p["ortho_err"] < 1e-3
                flag = "OK" if det_ok and ortho_ok else "ERR"
                t = p["translation"]
                K = p["intrinsics"]
                print(f"      [{flag}] {osp.basename(npz_path)}: "
                      f"t=[{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}]  "
                      f"det(R)={p['det_R']:.4f}  ortho_err={p['ortho_err']:.2e}")
                print(f"           K: fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  "
                      f"cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
                if not det_ok or not ortho_ok:
                    pose_issues.append(npz_path)

            # Pose continuity
            diffs = check_pose_continuity(pairs)
            if diffs:
                mn, mx, avg = min(diffs), max(diffs), float(np.mean(diffs))
                if mx > 10:
                    cont_flag = "WARN: inter-frame translation >10m, possible jump"
                elif mx < 1e-6:
                    cont_flag = "WARN: inter-frame translation almost zero"
                else:
                    cont_flag = "OK"
                print(f"    [Continuity] min={mn:.4f}m  max={mx:.4f}m  "
                      f"mean={avg:.4f}m  [{cont_flag}]")

        # Summary
        if depth_stats_all:
            mean_depth = float(np.mean([s["mean"] for s in depth_stats_all]))
            mean_ratio = float(np.mean([s["valid_ratio"] for s in depth_stats_all]))
            if mean_depth > 150:
                depth_flag = "WARN: depth mean too large"
            elif mean_ratio < 0.01:
                depth_flag = "WARN: very few valid pixels"
            else:
                depth_flag = "OK"
            print(f"\n  [Depth Summary] mean={mean_depth:.2f}m  "
                  f"valid_ratio={mean_ratio:.3f}  [{depth_flag}]")

        if pose_issues:
            print(f"  [Pose ERR] The following rotation matrices are invalid:")
            for p in pose_issues:
                print(f"     {p}")
        else:
            print(f"  [Pose Summary] All rotation matrices are valid [OK]")

    print("\n" + "=" * 60)
    print("  Check complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()