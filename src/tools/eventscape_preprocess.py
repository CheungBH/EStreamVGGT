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
    
    # Red for negative (BGR: [0, 0, 255]), Blue for positive (BGR: [255, 0, 0])
    # Background is black (0, 0, 0)
    img[ys[pos], xs[pos]] = [255, 0, 0]
    img[ys[neg], xs[neg]] = [0, 0, 255]
    
    return img

def run(args):
    # args.src is expected to be the dataset root, e.g. /path/to/dataset_example
    # Under args.src, there are multiple sequence directories like sequence_8, sequence_9, etc.
    
    seq_dirs = sorted(glob.glob(osp.join(args.src, "sequence_*")))
    if not seq_dirs:
        raise RuntimeError(f"No sequence_* directories found in {args.src}")
        
    print(f"Found {len(seq_dirs)} sequences in {args.src}")
    
    for seq_path in seq_dirs:
        seq_name = osp.basename(seq_path)
        seq_dst = osp.join(args.dst, args.src.split("/")[-1] + "-" + seq_name)
        os.makedirs(seq_dst, exist_ok=True)
        print(f"Processing sequence: {seq_name}")
        
        # Using the RAM_Net eventscape format:
        # <dataset_root>/sequence_*/
        #   rgb/data/ (contains *_image.png)
        #   depth/frames/ (contains 0000.png, etc)
        #   events/frames/ (contains 0000.png, etc)
        
        rgb_dir = osp.join(seq_path, "rgb", "data")
        depth_dir = osp.join(seq_path, "depth", "frames")
        events_dir = osp.join(seq_path, "events", "frames")
        
        rgb_files = sorted(glob.glob(osp.join(rgb_dir, "*.png")))
        num_frames = len(rgb_files)
        print(f"  Processing {num_frames} frames from {seq_name}")
        
        # Intrinsic calculation for CARLA (90 FOV)
        sample_img = cv2.imread(rgb_files[0])
        H, W = sample_img.shape[:2]
        focal = W / 2.0
        K_mat = np.array([
            [focal, 0, W/2.0],
            [0, focal, H/2.0],
            [0, 0, 1]
        ], dtype=np.float32)

        # In this format, events are already rendered as images (Voxel Grid or similar)
        event_files = sorted(glob.glob(osp.join(events_dir, "*.png")))
        depth_files = sorted(glob.glob(osp.join(depth_dir, "*.png"))) 
            
        # Check if poses exist in this format
        # RAM_Net EventScape saves pose as 'position.txt' (x, y, z) and 'orientation.txt' (pitch, yaw, roll)
        vehicle_data_dir = osp.join(seq_path, "vehicle_data")
        pos_file = osp.join(vehicle_data_dir, "position.txt")
        ori_file = osp.join(vehicle_data_dir, "orientation.txt")
        
        # The txt files use spaces as delimiters, not commas.
        positions = np.loadtxt(pos_file)
        orientations = np.loadtxt(ori_file)
        
        def carla_to_opencv_cam2world(x, y, z, pitch, yaw, roll):
            # CARLA uses left-handed coordinates: X-forward, Y-right, Z-up
            # OpenCV uses right-handed coordinates: X-right, Y-down, Z-forward
            pitch, yaw, roll = np.deg2rad([pitch, yaw, roll])

            cy, sy = np.cos(yaw), np.sin(yaw)
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)

            # Construct CARLA left-handed cam2world matrix
            P_carla = np.eye(4, dtype=np.float32)
            P_carla[0, 0] = cp * cy
            P_carla[0, 1] = cy * sp * sr - sy * cr
            P_carla[0, 2] = -cy * sp * cr - sy * sr
            P_carla[0, 3] = x

            P_carla[1, 0] = cp * sy
            P_carla[1, 1] = sy * sp * sr + cy * cr
            P_carla[1, 2] = -sy * sp * cr + cy * sr
            P_carla[1, 3] = y

            P_carla[2, 0] = sp
            P_carla[2, 1] = -cp * sr
            P_carla[2, 2] = cp * cr
            P_carla[2, 3] = z

            # Transformation matrix from OpenCV camera to CARLA camera
            T = np.array([
                [0,  0,  1,  0],
                [1,  0,  0,  0],
                [0, -1,  0,  0],
                [0,  0,  0,  1]
            ], dtype=np.float32)

            # Inverse transformation
            T_inv = np.array([
                [0,  1,  0,  0],
                [0,  0, -1,  0],
                [1,  0,  0,  0],
                [0,  0,  0,  1]
            ], dtype=np.float32)

            # Convert to OpenCV right-handed cam2world matrix
            P_cv = T_inv @ P_carla @ T
            return P_cv

        poses = []
        for p, o in zip(positions, orientations):
            # p: [x, y, z], o: [pitch, yaw, roll]
            P_cv = carla_to_opencv_cam2world(p[0], p[1], p[2], o[0], o[1], o[2])
            poses.append(P_cv)
            
        for i in range(num_frames):
            frame_id = f"{i:06d}"
            
            # 1. RGB
            img = cv2.imread(rgb_files[i])
            cv2.imwrite(osp.join(seq_dst, f"{frame_id}.png"), img)
            
            # 2. Events (Already PNGs in this dataset version)
            ev_img = cv2.imread(event_files[i])
            cv2.imwrite(osp.join(seq_dst, f"{frame_id}_event.png"), ev_img)
            
            # 3. Poses
            cam2world = poses[i].astype(np.float32)
                
            np.savez(osp.join(seq_dst, f"{frame_id}.npz"), 
                     intrinsics=K_mat.astype(np.float32), 
                     cam2world=cam2world)
                     
            # 4. Depth
            d_file = depth_files[i]
            d_img = cv2.imread(d_file, cv2.IMREAD_UNCHANGED)
            if d_img.ndim == 3: # RGB encoded depth
                R_ch = d_img[:,:,2].astype(np.float32)
                G_ch = d_img[:,:,1].astype(np.float32)
                B_ch = d_img[:,:,0].astype(np.float32)
                normalized = (R_ch + G_ch * 256.0 + B_ch * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
                depth_map = normalized * 1000.0 # Assuming max distance 1000m
            else:
                depth_map = d_img.astype(np.float32)
                
            cv2.imwrite(osp.join(seq_dst, f"{frame_id}.exr"), depth_map)
            
            if i > 0 and i % 100 == 0:
                print(f"  Processed {i}/{num_frames} frames")
                
    print("EventScape Preprocessing completed for all sequences.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Path to dataset root (e.g., RAM_Net/dataset_example, which contains sequence_* folders)")
    parser.add_argument("--dst", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
