import os
import os.path as osp
import argparse
import numpy as np
import cv2
import h5py

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def get_intrinsics_from_h5(f, cam_path):
    calib_path = f"/{cam_path}/calib/intrinsics"
    K = np.eye(3, dtype=np.float32)
    intrinsics_data = f[calib_path][:]
    K[0, 0] = intrinsics_data[0] # fx
    K[1, 1] = intrinsics_data[1] # fy
    K[0, 2] = intrinsics_data[2] # cx
    K[1, 2] = intrinsics_data[3] # cy
    return K

def aggregate_events_xy_p(xs, ys, ps, H, W):
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # p == 0 -> [0, 0, 255] (Red in BGR, Blue in RGB - assuming BGR for cv2)
    # p == 1 -> [255, 0, 0] (Blue in BGR, Red in RGB - assuming BGR for cv2)
    # Since cv2.imwrite expects BGR, [0, 0, 255] is Red, [255, 0, 0] is Blue
    pos = ps > 0
    neg = ps == 0
    
    img[ys[pos], xs[pos]] = [255, 0, 0]
    img[ys[neg], xs[neg]] = [0, 0, 255]
    
    return img

def find_nearest_idx(timestamps, target_t):
    return np.argmin(np.abs(timestamps.astype(np.float64) - target_t))


def run(args):
    os.makedirs(args.dst, exist_ok=True)
    
    print(f"Reading data HDF5: {args.data_h5}")
    f = h5py.File(args.data_h5, 'r')
    f_depth = h5py.File(args.depth_h5, 'r')
    f_pose = h5py.File(args.pose_h5, 'r')

    # 1. Images
    cam_path = 'ovc/left'
    img_data = f['/ovc/left/data'][:]
    
    # Image timestamps are stored in ovc/ts
    img_ts = f['/ovc/ts'][:]

    # Load calibration from HDF5
    K = get_intrinsics_from_h5(f, cam_path)
    print(f"Loaded Intrinsics K from {cam_path}:\n{K}")
    
    # 2. Events
    print("Loading events...")
    ev_x = f["prophesee/left/x"][:]
    ev_y = f["prophesee/left/y"][:]
    ev_p = f["prophesee/left/p"][:]
    
    # Event timestamps and map
    ts_map = np.asarray(f["/ovc/ts_map_prophesee_left_t"], dtype=np.int64)
    
    # 3. Poses
    print("Loading poses...")
    cam2worlds = f_pose['Cn_T_C0'][:]
    pose_ts = f_pose['ts'][:]
    
    # 4. Depth
    print("Loading depth maps...")
    
    # 根据您之前给出的打印信息：
    # ===== 查看key: depth ===== 
    # 类型：组，组内key列表：['prophesee'] 
    #   子key: prophesee 
    #   类型：组 
    # 既然 depth/prophesee 下面没有 data（导致报错 "object 'data' doesn't exist"），
    # 那说明 depth/prophesee 下面的数据集名字并不是 data。
    # 我直接通过 keys() 获取它的实际名字。不再瞎猜任何名字！
    
    depth_group = f_depth['depth/prophesee']
    if isinstance(depth_group, h5py.Dataset):
        depth_data = depth_group[:]
    else:
        # 绝对不写死 'data' 或者 'left'，直接拿真实的 key
        real_key = list(depth_group.keys())[0] # 大概率是 'left'
        if isinstance(depth_group[real_key], h5py.Group):
            # 有可能 real_key 还是个组，那就再下一层
            real_sub_key = list(depth_group[real_key].keys())[0] # 大概率是 'depth' 或 'image_raw'
            depth_data = depth_group[real_key][real_sub_key][:]
        else:
            depth_data = depth_group[real_key][:]
            
    depth_ts = f_depth['ts'][:]

    num_frames = len(img_data)
    print(f"Processing {num_frames} frames...")
    
    for i in range(num_frames):
        ts_target = img_ts[i]
        frame_id = f"{i:06d}"
        
        # Decode image
        img_i = img_data[i]
        
        # In M3ED, /ovc/left/data can be either raw byte array (JPEG encoded) or raw uint8 image matrix.
        # We explicitly handle the exact data type found.
        # The error indicates `buf.checkVector(1, CV_8U) > 0` failed, which means the array is NOT a 1D byte array.
        # It is already a 2D (grayscale) or 3D (color) image array.
        img_np = np.array(img_i)
        if img_np.ndim == 2:
            # Grayscale to BGR
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.ndim == 3 and img_np.shape[2] == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        # Note: We do not call cv2.imdecode here anymore because the data is already decoded!
            
        H, W = img_np.shape[:2]
        cv2.imwrite(osp.join(args.dst, f"{frame_id}.png"), img_np)
        
        # Process Events
        # Aggregate events between this image and the next image
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
        
        # Process Poses
        idx_pose = find_nearest_idx(pose_ts, ts_target)
        cam2world = cam2worlds[idx_pose]
        
        # Save npz
        np.savez(osp.join(args.dst, f"{frame_id}.npz"), 
                 intrinsics=K.astype(np.float32), 
                 cam2world=cam2world.astype(np.float32))
                 
        # Process Depth
        idx_depth = find_nearest_idx(depth_ts, ts_target)
        depth_map = depth_data[idx_depth].astype(np.float32)
            
        cv2.imwrite(osp.join(args.dst, f"{frame_id}.exr"), depth_map)
        
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
    parser.add_argument("--depth_h5", type=str, required=True, help="官方单独提供的 depth GT HDF5")
    parser.add_argument("--pose_h5", type=str, required=True, help="官方单独提供的 pose GT HDF5")
    parser.add_argument("--dst", type=str, required=True, help="输出目录（会生成四件套）")
    
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
