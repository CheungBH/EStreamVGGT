import h5py

data_h5 = "/home/bhzhang/Documents/datasets/m3ed/falcon_forest_into_forest_1/falcon_forest_into_forest_1_data.h5"
depth_h5 = "/home/bhzhang/Documents/datasets/m3ed/falcon_forest_into_forest_1/falcon_forest_into_forest_1_depth_gt.h5"
pose_h5 = "/home/bhzhang/Documents/datasets/m3ed/falcon_forest_into_forest_1/falcon_forest_into_forest_1_pose_gt.h5"


def print_h5_structure(path, max_depth=4):
    def visitor(name, obj):
        depth = name.count('/')
        if depth > max_depth:
            return
        indent = '  ' * depth
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"{indent}{name}/")
    print(f"\n=== {path} ===")
    with h5py.File(path, 'r') as f:
        f.visititems(visitor)

print_h5_structure(depth_h5)
print_h5_structure(pose_h5)
print_h5_structure(data_h5)
