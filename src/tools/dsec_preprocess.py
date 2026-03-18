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
    if path.endswith(".npz"):
        z = np.load(path)
        if "intrinsics" in z:
            K = z["intrinsics"].astype(np.float32)
        else:
            K = z["K"].astype(np.float32)
    elif path.endswith(".npy"):
        K = np.load(path).astype(np.float32)
    else:
        arr = np.loadtxt(path).astype(np.float32)
        if arr.size == 9:
            K = arr.reshape(3, 3)
        else:
            fx, fy, cx, cy = arr.flatten()[:4]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return K


def read_extrinsics_per_frame(path, num_frames):
    mats = []
    if path.endswith(".npz"):
        z = np.load(path)
        if "cam2worlds" in z:
            mats = [z["cam2worlds"][i].astype(np.float32) for i in range(len(z["cam2worlds"]))]
        elif "world2cams" in z:
            mats = [np.linalg.inv(z["world2cams"][i]).astype(np.float32) for i in range(len(z["world2cams"]))]
    else:
        arr = np.loadtxt(path).astype(np.float32)
        if arr.ndim == 2 and arr.shape[1] in (12, 16):
            step = arr.shape[1]
            for i in range(arr.shape[0]):
                flat = arr[i]
                if step == 12:
                    M = np.eye(4, dtype=np.float32)
                    M[:3, :4] = flat.reshape(3, 4)
                else:
                    M = flat.reshape(4, 4)
                mats.append(M)
    if not mats:
        mats = [np.eye(4, dtype=np.float32) for _ in range(num_frames)]
    if len(mats) != num_frames:
        if len(mats) > num_frames:
            mats = mats[:num_frames]
        else:
            mats = mats + [mats[-1].copy() for _ in range(num_frames - len(mats))]
    return mats


def load_events(events_h5):
    with h5py.File(events_h5, "r") as f:
        x = f["events/xs"][:]
        y = f["events/ys"][:]
        t = f["events/ts"][:]
        p = f["events/ps"][:]
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
    if args.events_h5:
        x, y, te, p = load_events(osp.join(args.src, args.events_h5))
        if args.timestamps_txt:
            ts = np.loadtxt(osp.join(args.src, args.timestamps_txt)).astype(np.float64)
        else:
            ts = np.linspace(te.min(), te.max(), num=len(imgs), dtype=np.float64)
    else:
        x = y = te = p = None
        ts = np.zeros((len(imgs),), dtype=np.float64)
    disp_paths = sorted(glob.glob(osp.join(args.src, args.disparity_glob))) if args.disparity_glob else []
    for i, impath in enumerate(imgs):
        rgb = cv2.cvtColor(cv2.imread(impath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        frame_id = osp.splitext(osp.basename(impath))[0]
        rgb_out = osp.join(seq_dst, frame_id + ".png")
        cv2.imwrite(rgb_out, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if x is not None:
            t1 = ts[i]
            t0 = t1 - args.event_window_ms / 1000.0
            ev = aggregate_events(x, y, te, p, t0, t1, H0, W0)
            ev_out = osp.join(seq_dst, frame_id + "_event.png")
            cv2.imwrite(ev_out, cv2.cvtColor(ev, cv2.COLOR_RGB2BGR))
        if disp_paths and i < len(disp_paths):
            disp = cv2.imread(disp_paths[i], cv2.IMREAD_ANYDEPTH)
            depth = disparity_to_depth(disp, K[0, 0], args.baseline)
            save_exr(osp.join(seq_dst, frame_id + ".exr"), depth)
        else:
            save_exr(osp.join(seq_dst, frame_id + ".exr"), np.zeros((H0, W0), dtype=np.float32))
        np.savez(osp.join(seq_dst, frame_id + ".npz"), intrinsics=K.astype(np.float32), cam2world=mats[i].astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True)
    ap.add_argument("--dst", type=str, required=True)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--images_glob", type=str, default="images/*.png")
    ap.add_argument("--events_h5", type=str, default="")
    ap.add_argument("--timestamps_txt", type=str, default="")
    ap.add_argument("--intrinsics", type=str, required=True)
    ap.add_argument("--extrinsics", type=str, required=True)
    ap.add_argument("--disparity_glob", type=str, default="")
    ap.add_argument("--baseline", type=float, default=0.1)
    ap.add_argument("--event_window_ms", type=float, default=30.0)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
