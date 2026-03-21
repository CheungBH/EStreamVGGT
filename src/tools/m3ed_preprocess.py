import os
import os.path as osp
import argparse
import numpy as np
import cv2
import h5py
import yaml
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

    return aggregate_events_xy_p(xs, ys, ps, H, W)


def aggregate_events_xy_p(xs, ys, ps, H, W):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    if xs.size == 0:
        return img

    pos = ps > 0
    neg = ~pos
    img[ys[pos], xs[pos], 0] = 255
    img[ys[neg], xs[neg], 1] = 255
    img[..., 2] = np.maximum(img[..., 0], img[..., 1])
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

def find_group_by_keywords(f, keywords):
    """
    Helper function to find a group or dataset in an HDF5 file
    by matching any of the given keywords in the keys.
    """
    for key in f.keys():
        for kw in keywords:
            if kw in key.lower():
                return f[key]
    # If not found at top level, check one level deeper
    for key in f.keys():
        if isinstance(f[key], h5py.Group):
            for subkey in f[key].keys():
                for kw in keywords:
                    if kw in subkey.lower():
                        return f[key][subkey]
    return None

def run(args):
    os.makedirs(args.dst, exist_ok=True)
    
    print(f"Reading data HDF5: {args.data_h5}")
    f = h5py.File(args.data_h5, 'r')
    
    f_depth = None
    f_pose = None
    if args.depth_h5:
        print(f"Reading depth GT HDF5: {args.depth_h5}")
        f_depth = h5py.File(args.depth_h5, 'r')
    if args.pose_h5:
        print(f"Reading pose GT HDF5: {args.pose_h5}")
        f_pose = h5py.File(args.pose_h5, 'r')

    depth_src = f_depth if f_depth is not None else f
    pose_src = f_pose if f_pose is not None else f

    # Load calibration
    K, T_cam_lidar = read_calibration(args.calib_yaml)
    print(f"Loaded Intrinsics K:\n{K}")
    
    # 1. Images
    img_group = f.get('images/left') or f.get('/ovc/left') or f.get('rgb/image_raw') or find_group_by_keywords(f, ['image', 'ovc', 'rgb'])
    if img_group is None:
        raise ValueError("Cannot find image group in HDF5")
        
    img_data = img_group['image'][:] if 'image' in img_group else img_group['data'][:]
    img_ts = img_group['ts'][:] if 'ts' in img_group else img_group['t'][:]
    
    # 2. Events
    ev_group = f.get("/prophesee/left") or f.get("prophesee/left") or f.get("events/left") or f.get("events") or find_group_by_keywords(f, ['event', 'prophesee'])
    has_events = ev_group is not None
    ts_map = None
    if has_events:
        print("Loading events...")
        ev_x = ev_group["x"][:]
        ev_y = ev_group["y"][:]
        ev_t = ev_group["ts"][:] if "ts" in ev_group else ev_group["t"][:]
        ev_p = ev_group["p"][:]
        ts_map = f.get("/ovc/ts_map_prophesee_left_t") or f.get("ovc/ts_map_prophesee_left_t")
        if ts_map is not None:
            ts_map = np.asarray(ts_map, dtype=np.int64)
    
    # 3. Poses
    pose_group = pose_src.get("poses/gt") or pose_src.get("/poses/gt") or pose_src.get("poses/stamped_poses") or pose_src.get("/poses/stamped_poses") or find_group_by_keywords(pose_src, ['pose', 'odometry', 'gt'])
    has_poses = pose_group is not None
    if has_poses:
        print("Loading poses...")
        pose_pos = pose_group['position'][:] if 'position' in pose_group else (pose_group['tx_ty_tz'][:] if 'tx_ty_tz' in pose_group else pose_group['data'][:, :3])
        pose_ori = pose_group['orientation'][:] if 'orientation' in pose_group else (pose_group['qx_qy_qz_qw'][:] if 'qx_qy_qz_qw' in pose_group else pose_group['data'][:, 3:7])
        pose_ts = pose_group['ts'][:] if 'ts' in pose_group else pose_group['t'][:]
    
    # 4. Depth / LiDAR
    depth_group = depth_src.get("depth/left") or depth_src.get("/depth/left") or find_group_by_keywords(depth_src, ['depth'])
    has_depth = depth_group is not None
    if has_depth:
        print("Loading depth maps...")
        depth_data = depth_group['depth'][:] if 'depth' in depth_group else depth_group['data'][:]
        depth_ts = depth_group['ts'][:] if 'ts' in depth_group else depth_group['t'][:]
        
    lidar_group = f.get('lidar/points') or f.get('ouster') or f.get('/ouster') or find_group_by_keywords(f, ['lidar', 'ouster'])
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
        img_i = img_data[i]
        if isinstance(img_i, np.ndarray) and img_i.dtype == np.uint8 and img_i.ndim == 1:
            img_np = cv2.imdecode(img_i, cv2.IMREAD_COLOR)
        else:
            img_np = np.asarray(img_i)
            if img_np.ndim == 3 and img_np.shape[0] == 3:
                img_np = img_np.transpose(1, 2, 0)
        
        H, W = img_np.shape[:2]
        cv2.imwrite(osp.join(args.dst, f"{frame_id}.png"), img_np)
        
        # Process Events
        if has_events:
            if args.event_mode in ("between_images", "center_n_events") and ts_map is not None:
                center = int(ts_map[i])
                if args.event_mode == "between_images":
                    start = center
                    if i + 1 < len(ts_map):
                        stop = int(ts_map[i + 1])
                    else:
                        stop = min(center + int(args.n_events), ev_x.shape[0])
                else:
                    half = int(args.n_events) // 2
                    start = max(0, center - half)
                    stop = min(ev_x.shape[0], center + half)

                ev_img = aggregate_events_xy_p(ev_x[start:stop], ev_y[start:stop], ev_p[start:stop], H, W)
            else:
                t1 = ts_target
                if ev_t[0] > 1e12:
                    t0 = t1 - (args.event_window_ms * 1e3)
                else:
                    t0 = t1 - (args.event_window_ms * 1e-3)
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
    if f_depth is not None:
        f_depth.close()
    if f_pose is not None:
        f_pose.close()
    print("M3ED Preprocessing completed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_h5", type=str, required=True, help="M3ED 序列的传感器 HDF5（包含图像/事件/雷达等）")
    parser.add_argument("--depth_h5", type=str, default="", help="(可选) 官方单独提供的 depth GT HDF5；提供则优先用于生成 .exr")
    parser.add_argument("--pose_h5", type=str, default="", help="(可选) 官方单独提供的 pose GT HDF5；提供则优先用于生成 cam2world")
    parser.add_argument("--calib_yaml", type=str, required=True, help="相机/雷达标定 yaml（用于 K 和 T_cam_lidar）")
    parser.add_argument("--dst", type=str, required=True, help="输出目录（会生成四件套）")
    
    parser.add_argument(
        "--event_mode",
        type=str,
        default="between_images",
        choices=("between_images", "center_n_events", "time_window_ms"),
        help="事件聚合策略：优先用 /ovc/ts_map_prophesee_left_t 按相邻两帧聚合；无映射时可退回固定时间窗",
    )
    parser.add_argument("--n_events", type=int, default=200000, help="center_n_events 模式下使用")
    parser.add_argument("--event_window_ms", type=float, default=30.0, help="time_window_ms 模式下使用")
    
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
