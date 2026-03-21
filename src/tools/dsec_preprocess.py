import os
import os.path as osp
import argparse
import numpy as np
import cv2
import glob
import h5py
from PIL import Image

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def read_intrinsics(path):
    # Hard logic for DSEC calibration format
    import yaml
    with open(path, 'r') as f:
        calib = yaml.safe_load(f)
    K = np.array(calib['cam_to_cam']['cam0']['K']).reshape(3,3).astype(np.float32)
    return K


def read_extrinsics_per_frame(path, num_frames):
    # Hard logic for DSEC poses format (which is a csv or txt file typically for poses)
    # The extrinsics path here should point to the ground truth poses file or equivalent
    arr = np.loadtxt(path, delimiter=',').astype(np.float32)
    mats = []
    for i in range(min(arr.shape[0], num_frames)):
        flat = arr[i]
        M = np.eye(4, dtype=np.float32)
        # Assuming standard format [tx, ty, tz, qx, qy, qz, qw]
        from scipy.spatial.transform import Rotation as R
        M[:3, 3] = flat[:3]
        M[:3, :3] = R.from_quat(flat[3:7]).as_matrix()
        mats.append(M)
    
    # Pad if not enough poses
    while len(mats) < num_frames:
        mats.append(mats[-1] if mats else np.eye(4, dtype=np.float32))
        
    return mats


def load_events(events_h5):
    with h5py.File(events_h5, "r") as f:
        x = f["events/x"][:]
        y = f["events/y"][:]
        t = f["events/t"][:] / 1e6  # converted to seconds
        p = f["events/p"][:]
    return x, y, t, p


def aggregate_events(x, y, t, p, t0, t1, H, W):
    mask = (t >= t0) & (t < t1)
    xs = x[mask]
    ys = y[mask]
    ps = p[mask]
    img = np.zeros((H, W, 3), dtype=np.uint8)
    if xs.size == 0:
        return img
    pos = ps > 0
    neg = ~pos
    img[ys[pos], xs[pos], 0] = 255
    img[ys[neg], xs[neg], 1] = 255
    img[..., 2] = np.maximum(img[..., 0], img[..., 1])
    return img


def disparity_to_depth(disp, fx, baseline):
    disp_f = disp.astype(np.float32)
    depth = np.zeros_like(disp_f, dtype=np.float32)
    valid = disp_f > 1e-6
    depth[valid] = fx * baseline / disp_f[valid]
    return depth


def save_exr(path, depth):
    cv2.imwrite(path, depth.astype(np.float32))


def run(args):
    seq_dst = osp.join(args.dst, args.name)
    os.makedirs(seq_dst, exist_ok=True)
    imgs = sorted(glob.glob(osp.join(args.src, args.images_glob)))
    if not imgs:
        raise RuntimeError("no images found")
    H0, W0 = np.array(Image.open(imgs[0])).shape[:2]
    K = read_intrinsics(osp.join(args.src, args.intrinsics))
    mats = read_extrinsics_per_frame(osp.join(args.src, args.extrinsics), len(imgs))
    
    x, y, te, p = load_events(osp.join(args.src, args.events_h5))
    ts = np.loadtxt(osp.join(args.src, args.timestamps_txt)).astype(np.float64) / 1e6 # assuming timestamps are in microseconds

    disp_paths = sorted(glob.glob(osp.join(args.src, args.disparity_glob)))
    
    # Pre-calculate disparities mapping
    # Assuming disparity images map 1-to-1 with the rgb images
    disp_files = []
    if args.disparity_glob:
        disp_files = sorted(glob.glob(osp.join(args.src, args.disparity_glob)))
    
    for i, impath in enumerate(imgs):
        rgb = cv2.cvtColor(cv2.imread(impath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        frame_id = osp.splitext(osp.basename(impath))[0]
        rgb_out = osp.join(seq_dst, frame_id + ".png")
        cv2.imwrite(rgb_out, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        t1 = ts[i]
        t0 = t1 - args.event_window_ms / 1000.0
        ev = aggregate_events(x, y, te, p, t0, t1, H0, W0)
        ev_out = osp.join(seq_dst, frame_id + "_event.png")
        cv2.imwrite(ev_out, cv2.cvtColor(ev, cv2.COLOR_RGB2BGR))
        
        disp = cv2.imread(disp_files[i], cv2.IMREAD_ANYDEPTH)
        depth = disparity_to_depth(disp, K[0, 0], args.baseline)
        save_exr(osp.join(seq_dst, frame_id + ".exr"), depth)
        
        np.savez(osp.join(seq_dst, frame_id + ".npz"), intrinsics=K.astype(np.float32), cam2world=mats[i].astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True)
    ap.add_argument("--dst", type=str, required=True)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--images_glob", type=str, required=True)
    ap.add_argument("--events_h5", type=str, required=True)
    ap.add_argument("--timestamps_txt", type=str, required=True)
    ap.add_argument("--intrinsics", type=str, required=True)
    ap.add_argument("--extrinsics", type=str, required=True)
    ap.add_argument("--disparity_glob", type=str, required=True)
    ap.add_argument("--baseline", type=float, default=0.1)
    ap.add_argument("--event_window_ms", type=float, default=30.0)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
