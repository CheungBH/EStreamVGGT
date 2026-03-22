import os
import os.path as osp
import argparse
import numpy as np
import cv2
import glob

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def aggregate_events_xy_p(xs, ys, ps, H, W):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    pos = ps > 0
    neg = ps == 0
    
    img[ys[pos], xs[pos]] = [255, 0, 0]
    img[ys[neg], xs[neg]] = [0, 0, 255]
    
    return img

def run(args):
    os.makedirs(args.dst, exist_ok=True)
    
    # EventScape directory structure based on image:
    # <sequence_name>/
    #   depth/
    #   events/
    #   rgb/
    #   semantic/
    #   vehicle_data/
    
    rgb_dir = osp.join(args.src, "rgb")
    depth_dir = osp.join(args.src, "depth")
    events_dir = osp.join(args.src, "events")
    vehicle_data_dir = osp.join(args.src, "vehicle_data")
    
    if not osp.exists(rgb_dir):
        raise RuntimeError(f"RGB directory not found at {rgb_dir}")
        
    rgb_files = sorted(glob.glob(osp.join(rgb_dir, "*.png")))
    if not rgb_files:
        raise RuntimeError("No RGB images found.")
        
    num_frames = len(rgb_files)
    print(f"Processing {num_frames} frames from {args.src}")
    
    # Intrinsic calculation for CARLA (90 FOV)
    sample_img = cv2.imread(rgb_files[0])
    H, W = sample_img.shape[:2]
    focal = W / 2.0
    K_mat = np.array([
        [focal, 0, W/2.0],
        [0, focal, H/2.0],
        [0, 0, 1]
    ], dtype=np.float32)

    event_files = sorted(glob.glob(osp.join(events_dir, "*.npz")))
    depth_files = sorted(glob.glob(osp.join(depth_dir, "*.npz"))) 
    if not depth_files:
        depth_files = sorted(glob.glob(osp.join(depth_dir, "*.png")))
        
    # Read transforms from vehicle_data/transforms.npz
    transform_file = osp.join(vehicle_data_dir, "transforms.npz")
    has_pose = False
    poses = []
    if osp.exists(transform_file):
        try:
            transform_data = np.load(transform_file)
            # Find the first array in the npz file
            keys = list(transform_data.keys())
            poses = transform_data[keys[0]]
            has_pose = True
        except Exception as e:
            print(f"Warning: Could not read poses from {transform_file}: {e}")
    else:
        print(f"Warning: Transforms file not found at {transform_file}")
        
    for i in range(num_frames):
        frame_id = f"{i:06d}"
        
        # 1. RGB
        img = cv2.imread(rgb_files[i])
        cv2.imwrite(osp.join(args.dst, f"{frame_id}.png"), img)
        
        # 2. Events
        ev_img = np.zeros((H, W, 3), dtype=np.uint8)
        if i < len(event_files):
            ev_data = np.load(event_files[i])
            if 'events' in ev_data:
                events = ev_data['events']
                if events.dtype.names is not None:
                    ev_x = events['x']
                    ev_y = events['y']
                    ev_p = events['p']
                    ev_img = aggregate_events_xy_p(ev_x, ev_y, ev_p, H, W)
            elif 'x' in ev_data:
                ev_x = ev_data['x']
                ev_y = ev_data['y']
                ev_p = ev_data['p']
                ev_img = aggregate_events_xy_p(ev_x, ev_y, ev_p, H, W)
                
        cv2.imwrite(osp.join(args.dst, f"{frame_id}_event.png"), ev_img)
        
        # 3. Poses
        if has_pose and i < len(poses):
            cam2world = poses[i].astype(np.float32)
        else:
            cam2world = np.eye(4, dtype=np.float32)
            
        np.savez(osp.join(args.dst, f"{frame_id}.npz"), 
                 intrinsics=K_mat.astype(np.float32), 
                 cam2world=cam2world)
                 
        # 4. Depth
        if i < len(depth_files):
            d_file = depth_files[i]
            if d_file.endswith('.npz'):
                d_data = np.load(d_file)
                if 'depth' in d_data:
                    depth_map = d_data['depth'].astype(np.float32)
                else:
                    keys = list(d_data.keys())
                    depth_map = d_data[keys[0]].astype(np.float32)
            else:
                d_img = cv2.imread(d_file, cv2.IMREAD_UNCHANGED)
                if d_img.ndim == 3: # RGB encoded depth
                    R = d_img[:,:,2].astype(np.float32)
                    G = d_img[:,:,1].astype(np.float32)
                    B = d_img[:,:,0].astype(np.float32)
                    normalized = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
                    depth_map = normalized * 1000.0 # Assuming max distance 1000m
                else:
                    depth_map = d_img.astype(np.float32)
                    
            cv2.imwrite(osp.join(args.dst, f"{frame_id}.exr"), depth_map)
            
        if i % 100 == 0:
            print(f"Processed {i}/{num_frames} frames")
            
    print("EventScape Preprocessing completed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Path to sequence directory (e.g., .../sequence_35)")
    parser.add_argument("--dst", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
