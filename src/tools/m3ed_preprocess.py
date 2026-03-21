import os
import os.path as osp
import argparse
import numpy as np
import cv2
import h5py
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation as R

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def read_calibration(yaml_path):
    """
    Reads calibration.yaml and returns intrinsic K and extrinsic T_cam_lidar.
    """
    with open(yaml_path, 'r') as f:
        calib = yaml.safe_load(f)
    
    # Defaults or extract from yaml
    # Assuming standard M3ED yaml format or generic format
    K = np.eye(3, dtype=np.float32)
    if 'intrinsics' in calib:
        intrinsics = calib['intrinsics']
        if isinstance(intrinsics, list) and len(intrinsics) >= 4:
            fx, fy, cx, cy = intrinsics[:4]
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
        elif isinstance(intrinsics, dict):
            K[0, 0] = intrinsics.get('fx', 1.0)
            K[1, 1] = intrinsics.get('fy', 1.0)
            K[0, 2] = intrinsics.get('cx', 0.0)
            K[1, 2] = intrinsics.get('cy', 0.0)
    elif 'cam0' in calib and 'intrinsics' in calib['cam0']:
        intrinsics = calib['cam0']['intrinsics']
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = intrinsics[:4]
    
    T_cam_lidar = np.eye(4, dtype=np.float32)
    if 'T_cam_lidar' in calib:
        T_cam_lidar = np.array(calib['T_cam_lidar'], dtype=np.float32).reshape(4, 4)
    elif 'extrinsic' in calib:
        T_cam_lidar = np.array(calib['extrinsic'], dtype=np.float32).reshape(4, 4)

    return K, T_cam_lidar

def ros_to_opencv_pose(pos, quat):
    """
    Converts ROS pose (x, y, z, qx, qy, qz, qw) to OpenCV 4x4 cam2world matrix.
    ROS standard: x-forward, y-left, z-up
    OpenCV standard: x-right, y-down, z-forward
    """
    # Create transformation matrix from ROS World to ROS Camera
    T_ros = np.eye(4, dtype=np.float32)
    T_ros[:3, :3] = R.from_quat(quat).as_matrix() # quat is [x,y,z,w]
    T_ros[:3, 3] = pos
    
    # Transformation from ROS Camera to OpenCV Camera
    # OpenCV z is ROS x, OpenCV x is ROS -y, OpenCV y is ROS -z
    R_ros2cv = np.array([
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [1,  0,  0,  0],
        [0,  0,  0,  1]
    ], dtype=np.float32)
    
    # Cam2World in OpenCV: T_world_cv = T_world_ros @ R_cv2ros
    # R_cv2ros is inverse of R_ros2cv
    R_cv2ros = np.linalg.inv(R_ros2cv)
    
    T_cv = T_ros @ R_cv2ros
    return T_cv

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
    img[ys[pos], xs[pos], 0] = 255 # Red for positive
    img[ys[neg], xs[neg], 1] = 255 # Green for negative
    img[..., 2] = np.maximum(img[..., 0], img[..., 1]) # Yellow for overlap
    return img

def project_lidar_to_depth(points, K, T_cam_lidar, H, W):
    """
    Project LiDAR points to a 2D depth map.
    points: (N, 3) or (N, 4) in LiDAR coordinate frame.
    """
    depth_map = np.zeros((H, W), dtype=np.float32)
    if points.size == 0:
        return depth_map
        
    pts_3d = points[:, :3]
    # To homogeneous
    pts_3d_h = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    
    # Transform to camera frame
    pts_cam = (T_cam_lidar @ pts_3d_h.T).T
    
    # Filter points behind the camera
    valid = pts_cam[:, 2] > 0
    pts_cam = pts_cam[valid]
    
    if pts_cam.size == 0:
        return depth_map
        
    # Project to 2D
    pts_2d = (K @ pts_cam[:, :3].T).T
    u = (pts_2d[:, 0] / pts_2d[:, 2]).astype(np.int32)
    v = (pts_2d[:, 1] / pts_2d[:, 2]).astype(np.int32)
    z = pts_cam[:, 2]
    
    # Filter points outside image
    valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid_uv]
    v = v[valid_uv]
    z = z[valid_uv]
    
    # Populate depth map (keep closest points)
    # Sort by depth descending so closer points overwrite farther points
    sort_idx = np.argsort(z)[::-1]
    u = u[sort_idx]
    v = v[sort_idx]
    z = z[sort_idx]
    
    depth_map[v, u] = z
    return depth_map

def find_nearest_idx(timestamps, target_t):
    return np.argmin(np.abs(timestamps - target_t))

def run(args):
    os.makedirs(args.dst, exist_ok=True)
    
    print(f"Reading HDF5 file: {args.data_h5}")
    f = h5py.File(args.data_h5, 'r')
    
    # Load calibration
    K, T_cam_lidar = read_calibration(args.calib_yaml)
    print(f"Loaded Intrinsics K:\n{K}")
    
    # 1. Images
    img_group = f.get('images/left') or f.get('ovc/left') or f.get('rgb/image_raw')
    if img_group is None:
        raise ValueError("Cannot find image group in HDF5")
        
    img_data = img_group['image'][:] if 'image' in img_group else img_group['data'][:]
    img_ts = img_group['ts'][:] if 'ts' in img_group else img_group['t'][:]
    
    # 2. Events
    ev_group = f.get('events/left') or f.get('prophesee/left') or f.get('events')
    has_events = ev_group is not None
    if has_events:
        print("Loading events...")
        ev_x = ev_group['x'][:]
        ev_y = ev_group['y'][:]
        ev_t = ev_group['ts'][:] if 'ts' in ev_group else ev_group['t'][:]
        ev_p = ev_group['p'][:]
    
    # 3. Poses
    pose_group = f.get('poses/gt') or f.get('poses/stamped_poses')
    has_poses = pose_group is not None
    if has_poses:
        print("Loading poses...")
        pose_pos = pose_group['position'][:] if 'position' in pose_group else pose_group['tx_ty_tz'][:]
        pose_ori = pose_group['orientation'][:] if 'orientation' in pose_group else pose_group['qx_qy_qz_qw'][:]
        pose_ts = pose_group['ts'][:] if 'ts' in pose_group else pose_group['t'][:]
    
    # 4. Depth / LiDAR
    depth_group = f.get('depth/left')
    has_depth = depth_group is not None
    if has_depth:
        print("Loading depth maps...")
        depth_data = depth_group['depth'][:] if 'depth' in depth_group else depth_group['data'][:]
        depth_ts = depth_group['ts'][:] if 'ts' in depth_group else depth_group['t'][:]
        
    lidar_group = f.get('lidar/points') or f.get('ouster')
    has_lidar = lidar_group is not None and not has_depth
    if has_lidar:
        print("Loading LiDAR points...")
        lidar_data = lidar_group['points'][:] if 'points' in lidar_group else lidar_group['data'][:]
        lidar_ts = lidar_group['ts'][:] if 'ts' in lidar_group else lidar_group['t'][:]

    num_frames = len(img_data)
    print(f"Processing {num_frames} frames...")
    
    for i in range(num_frames):
        ts_target = img_ts[i]
        frame_id = f"{i:06d}"
        
        # Decode image
        if img_data.dtype == np.uint8 and img_data.ndim == 1:
            # JPEG encoded
            img_np = cv2.imdecode(img_data[i], cv2.IMREAD_COLOR)
        else:
            img_np = img_data[i]
            if img_np.shape[0] == 3: # CHW to HWC
                img_np = img_np.transpose(1, 2, 0)
        
        H, W = img_np.shape[:2]
        cv2.imwrite(osp.join(args.dst, f"{frame_id}.png"), img_np)
        
        # Process Events
        if has_events:
            t1 = ts_target
            t0 = t1 - (args.event_window_ms * 1e-3) # assuming timestamps are in seconds
            # if timestamps are in microseconds, adjust accordingly
            if ev_t[0] > 1e12: # likely microseconds
                t0 = t1 - (args.event_window_ms * 1e3)
            
            ev_img = aggregate_events(ev_x, ev_y, ev_t, ev_p, t0, t1, H, W)
            cv2.imwrite(osp.join(args.dst, f"{frame_id}_event.png"), ev_img)
            
        # Process Poses
        cam2world = np.eye(4, dtype=np.float32)
        if has_poses:
            idx_pose = find_nearest_idx(pose_ts, ts_target)
            pos = pose_pos[idx_pose]
            quat = pose_ori[idx_pose] # [x, y, z, w]
            cam2world = ros_to_opencv_pose(pos, quat)
            
        # Save npz
        np.savez(osp.join(args.dst, f"{frame_id}.npz"), 
                 intrinsics=K.astype(np.float32), 
                 cam2world=cam2world.astype(np.float32))
                 
        # Process Depth
        depth_map = np.zeros((H, W), dtype=np.float32)
        if has_depth:
            idx_depth = find_nearest_idx(depth_ts, ts_target)
            depth_map = depth_data[idx_depth].astype(np.float32)
        elif has_lidar:
            idx_lidar = find_nearest_idx(lidar_ts, ts_target)
            points = lidar_data[idx_lidar]
            depth_map = project_lidar_to_depth(points, K, T_cam_lidar, H, W)
            
        cv2.imwrite(osp.join(args.dst, f"{frame_id}.exr"), depth_map.astype(np.float32))
        
        if i % 100 == 0:
            print(f"Processed {i}/{num_frames} frames")
            
    f.close()
    print("M3ED Preprocessing completed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_h5", type=str, required=True, help="Path to M3ED HDF5 file")
    parser.add_argument("--calib_yaml", type=str, required=True, help="Path to calibration YAML file")
    parser.add_argument("--dst", type=str, required=True, help="Output directory")
    parser.add_argument("--event_window_ms", type=float, default=30.0, help="Event aggregation window in milliseconds")
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
