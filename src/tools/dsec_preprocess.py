import os
import os.path as osp
import argparse
import numpy as np
import cv2
import glob
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
    
    img[ys[pos], xs[pos]] = [255, 0, 0]
    img[ys[neg], xs[neg]] = [0, 0, 255]
    
    return img

def run(args):
    # Initialize output directory
    seq_dst = osp.join(args.dst, args.name)
    os.makedirs(seq_dst, exist_ok=True)
    
    DATA_ROOT = Path(args.src)
    SEQ_NAME = args.name
    USE_EVENT_VIEW = args.use_event_view
    
    DISP_FOLDER = "disparity/event" if USE_EVENT_VIEW else "disparity/image"
    
    # Set up paths based on Grok's working structure
    img_dir = DATA_ROOT / "RGB_event/train" / SEQ_NAME / "images/left/distorted"
    event_h5 = DATA_ROOT / "RGB_event/train" / SEQ_NAME / "events/left/events.h5"
    disp_dir = DATA_ROOT / "train_disparity" / SEQ_NAME / DISP_FOLDER
    calib_file = DATA_ROOT / "train_calibration" / SEQ_NAME / "calibration/cam_to_cam.yaml"
    
    base_seq = SEQ_NAME.rsplit('_', 1)[0] if '_' in SEQ_NAME[-2:] else SEQ_NAME
    bag_dir = DATA_ROOT / "lidar_imu" / "data" / base_seq
    bag_file = list(bag_dir.glob("*.bag"))[0] if bag_dir.exists() and len(list(bag_dir.glob("*.bag"))) > 0 else None

    # Load intrinsics (K) and Disparity-to-Depth mapping (Q)
    with open(calib_file, 'r') as f:
        calib = yaml.safe_load(f)
        
    K = np.array(calib['cam_to_cam']['cam0']['K']).reshape(3,3).astype(np.float32)
    
    q_key = "cams_03" if USE_EVENT_VIEW else "cams_12"
    Q = np.array(calib['disparity_to_depth'][q_key], dtype=np.float32)
    
    # Load RGB frame paths
    rgb_files = sorted(img_dir.glob("*.png"))
    if not rgb_files:
        raise RuntimeError(f"No RGB images found in {img_dir}")
        
    frame_ids = [f.stem for f in rgb_files]
    print(f"Found {len(frame_ids)} frames")
    
    # Get image dimensions from first image
    sample_rgb = cv2.imread(str(rgb_files[0]))
    H, W = sample_rgb.shape[:2]
    
    # Load events into memory
    print("Loading events from H5...")
    with h5py.File(event_h5, 'r') as f:
        events = {
            't': f['events']['t'][:],
            'x': f['events']['x'][:],
            'y': f['events']['y'][:],
            'p': f['events']['p'][:],
        }

    # ROS bag setup for Poses
    pose_topic = '/lio_sam/mapping/odometry'
    bag_reader = None
    connections_pose = []
    
    if bag_file:
        bag_reader = AnyReader([bag_file])
        bag_reader.__enter__()
        connections_pose = [c for c in bag_reader.connections if c.topic == pose_topic]
        if not connections_pose:
            print(f"Warning: Topic {pose_topic} not found in bag. Poses will be identity.")
            bag_reader.__exit__(None, None, None)
            bag_reader = None
    else:
        print("Warning: No ROS bag found. Poses will be identity.")

    # Main processing loop
    t_prev_us = events['t'][0] if len(events['t']) > 0 else 0
    event_window_us = int(args.event_window_ms * 1000)

    for idx, fid in enumerate(tqdm.tqdm(frame_ids)):
        # 1. RGB
        rgb_path = img_dir / f"{fid}.png"
        rgb = cv2.imread(str(rgb_path))
        if rgb is None: continue
        # Write to EStreamVGGT format
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
            # Save zero depth if not available
            cv2.imwrite(osp.join(seq_dst, f"{fid}.exr"), np.zeros((H, W), dtype=np.float32))

        # 3. Events
        t_curr_us = t_prev_us + event_window_us
        mask = (events['t'] >= t_prev_us) & (events['t'] < t_curr_us)
        ev_img = aggregate_events_xy_p(events['x'][mask], events['y'][mask], events['p'][mask], H, W)
        cv2.imwrite(osp.join(seq_dst, f"{fid}_event.png"), ev_img)
        t_prev_us = t_curr_us

        # 4. Pose
        pose = np.eye(4, dtype=np.float32)
        if bag_reader and connections_pose:
            min_dt = float('inf')
            closest_msg = None

            for connection, timestamp_ns, rawdata in bag_reader.messages(connections=connections_pose):
                msg = bag_reader.deserialize(rawdata, connection.msgtype)
                t_msg_us = timestamp_ns // 1000
                dt = abs(t_msg_us - t_curr_us)
                if dt < min_dt:
                    min_dt = dt
                    closest_msg = msg
                if min_dt < 5000:  # within 5ms
                    break

            if closest_msg and min_dt < 5000:
                try:
                    p = closest_msg.pose.pose.position
                    q = closest_msg.pose.pose.orientation
                except AttributeError:
                    try:
                        p = closest_msg.pose.position
                        q = closest_msg.pose.orientation
                    except AttributeError:
                        p = None

                if p is not None:
                    pose[0:3, 3] = [p.x, p.y, p.z]
                    rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                    pose[0:3, 0:3] = rot

        # Save Intrinsics and Pose
        np.savez(osp.join(seq_dst, f"{fid}.npz"), 
                 intrinsics=K, 
                 cam2world=pose)

    if bag_reader:
        bag_reader.__exit__(None, None, None)
        
    print(f"DSEC EStreamVGGT Preprocessing completed at: {seq_dst}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="DSEC Root Directory (e.g. /home/bhzhang/Documents/datasets/DSEC)")
    ap.add_argument("--dst", type=str, required=True, help="Output root directory")
    ap.add_argument("--name", type=str, required=True, help="Sequence name (e.g. zurich_city_09_a)")
    ap.add_argument("--event_window_ms", type=float, default=50.0, help="Event accumulation window in ms")
    ap.add_argument("--use_event_view", action="store_true", help="Use event view disparity mapping (cams_03)")
    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
