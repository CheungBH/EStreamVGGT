import os
import os.path as osp
import argparse
import numpy as np
import cv2
import h5py

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def aggregate_events_xy_p(xs, ys, ps, H, W):
    img = np.zeros((H, W, 3), dtype=np.uint8)

    pos = ps > 0
    neg = ps == 0
    
    img[ys[pos], xs[pos]] = [255, 0, 0]
    img[ys[neg], xs[neg]] = [0, 0, 255]
    
    return img

def find_nearest_idx(timestamps, target_t):
    return np.argmin(np.abs(timestamps.astype(np.float64) - target_t))

def run(args):
    os.makedirs(args.dst, exist_ok=True)
    
    print(f"Reading MVSEC HDF5: {args.data_h5}")
    f = h5py.File(args.data_h5, 'r')
    
    # In MVSEC, left camera images are in 'davis/left/image_raw'
    img_data = f['davis/left/image_raw'][:]
    img_ts = f['davis/left/image_raw_ts'][:]
    
    # Events
    print("Loading events...")
    ev_x = f['davis/left/events/x'][:]
    ev_y = f['davis/left/events/y'][:]
    ev_p = f['davis/left/events/p'][:]
    
    # Map from image index to event index
    ts_map = f['davis/left/image_raw_event_inds'][:]
    
    # Intrinsics
    # For MVSEC, we load intrinsics from calibration txt file
    import yaml
    with open(args.calib, 'r') as cf:
        calib = yaml.safe_load(cf)
    K = np.array(calib['cam0']['intrinsics'])
    K_mat = np.array([
        [K[0], 0, K[2]],
        [0, K[1], K[3]],
        [0, 0, 1]
    ], dtype=np.float32)
        
    num_frames = len(img_data)
    print(f"Processing {num_frames} frames...")
    
    # Pose loading
    f_pose = h5py.File(args.pose_h5, 'r')
    pose_data = f_pose['davis/left/pose'][:]
    pose_ts = f_pose['davis/left/pose_ts'][:]

    # Depth loading
    f_depth = h5py.File(args.depth_h5, 'r')
    depth_data = f_depth['davis/left/depth_image_raw'][:]
    depth_ts = f_depth['davis/left/depth_image_raw_ts'][:]
            
    for i in range(num_frames):
        ts_target = img_ts[i]
        frame_id = f"{i:06d}"
        
        # 1. Save Image
        img_np = img_data[i]
        if img_np.ndim == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        H, W = img_np.shape[:2]
        cv2.imwrite(osp.join(args.dst, f"{frame_id}.png"), img_np)
        
        # 2. Save Events
        start = int(ts_map[i])
        if i + 1 < len(ts_map):
            stop = int(ts_map[i + 1])
        else:
            stop = ev_x.shape[0]

        if start < stop:
            ev_img = aggregate_events_xy_p(ev_x[start:stop], ev_y[start:stop], ev_p[start:stop], H, W)
        else:
            ev_img = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.imwrite(osp.join(args.dst, f"{frame_id}_event.png"), ev_img)
        
        # 3. Save Poses
        idx_pose = find_nearest_idx(pose_ts, ts_target)
        pos_quat = pose_data[idx_pose] # [x, y, z, qx, qy, qz, qw]
        M = np.eye(4, dtype=np.float32)
        M[:3, 3] = pos_quat[:3]
        from scipy.spatial.transform import Rotation as R
        M[:3, :3] = R.from_quat(pos_quat[3:]).as_matrix()
        cam2world = M
            
        np.savez(osp.join(args.dst, f"{frame_id}.npz"), 
                 intrinsics=K_mat.astype(np.float32), 
                 cam2world=cam2world.astype(np.float32))
                 
        # 4. Save Depth
        idx_depth = find_nearest_idx(depth_ts, ts_target)
        depth_map = depth_data[idx_depth].astype(np.float32)
        cv2.imwrite(osp.join(args.dst, f"{frame_id}.exr"), depth_map)
            
        if i % 100 == 0:
            print(f"Processed {i}/{num_frames} frames")
            
    f.close()
    f_pose.close()
    f_depth.close()
    print("MVSEC Preprocessing completed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_h5", type=str, required=True, help="MVSEC Data HDF5 (e.g. outdoor_day1_data.hdf5)")
    parser.add_argument("--depth_h5", type=str, required=True, help="MVSEC Depth HDF5")
    parser.add_argument("--pose_h5", type=str, required=True, help="MVSEC Pose/GT HDF5")
    parser.add_argument("--calib", type=str, required=True, help="Calibration yaml file")
    parser.add_argument("--dst", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
