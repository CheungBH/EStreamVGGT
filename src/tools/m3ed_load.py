import h5py
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_h5", required=True, help="H5 file path with sensor data")
parser.add_argument("--depth_h5", required=True, help="H5 file path with sensor data")
parser.add_argument("--pose_h5", required=True, help="H5 file path with sensor data")

parser.add_argument("--idx", type=int, default=100, help="Image index to load")
parser.add_argument("--n_events", type=int, default=200000, help="Number of events to load (centered on the image)")
args = parser.parse_args()

# Start by loading a file through h5py. The compression used is available within the base h5py package.
f = h5py.File(args.data_h5, 'r')
f_depth = h5py.File(args.depth_h5, 'r')

import h5py
import numpy as np

# 假设你的args.depth_h5是文件路径，这里先替换成实际路径演示
file_path = args.pose_h5  # 或直接写死路径：file_path = "你的文件路径.h5"

with h5py.File(file_path, 'r') as f1:
    # 遍历所有顶层key
    for key in f1.keys():
        print(f"\n===== 查看key: {key} =====")
        obj = f1[key]

        # 1. 如果是数据集（Dataset）：查看具体数据
        if isinstance(obj, h5py.Dataset):
            print(f"类型：数据集")
            print(f"形状：{obj.shape}")
            print(f"数据类型：{obj.dtype}")

            # 查看前5行/前几个数据（避免数据量太大）
            print("前5条数据：")
            # 根据维度适配显示方式
            if len(obj.shape) == 1:
                # 一维数据（如ts）：直接打印前5个值
                print(obj[:5])
            elif len(obj.shape) >= 2:
                # 多维数据（如Cn_T_C0）：打印第一个样本的完整数据
                print(obj[0])  # 打印第0个样本（889个中的第一个）

            # 可选：提取整个数据集到numpy数组
            # data_array = np.array(obj)
            # print(f"数据总长度：{len(data_array)}")

        # 2. 如果是组（Group）：查看组内的嵌套key和数据
        elif isinstance(obj, h5py.Group):
            print(f"类型：组，组内key列表：{list(obj.keys())}")
            # 遍历组内的所有key
            for sub_key in obj.keys():
                sub_obj = obj[sub_key]
                print(f"\n  子key: {sub_key}")
                print(f"  类型：{'数据集' if isinstance(sub_obj, h5py.Dataset) else '组'}")
                if isinstance(sub_obj, h5py.Dataset):
                    print(f"  形状：{sub_obj.shape}")
                    print(f"  前5条数据：{sub_obj[:5]}")

# 补充：单独查看某个指定key的完整数据
with h5py.File(file_path, 'r') as f1:
    # 示例1：查看depth组内的内容（你的输出显示depth有1个成员）
    if 'depth' in f1:
        print("\n===== 单独查看depth组 =====")
        depth_group = f1['depth']
        print(f"depth组内的key：{list(depth_group.keys())}")
        # 假设depth组内有一个叫'data'的key（根据实际情况替换）
        if 'data' in depth_group:
            print(f"depth/data的形状：{depth_group['data'].shape}")
            print(f"depth/data的前5条数据：{depth_group['data'][:5]}")

    # 示例2：提取Cn_T_C0的所有数据到numpy数组
    if 'Cn_T_C0' in f1:
        cn_data = np.array(f1['Cn_T_C0'])
        print(f"\n===== 提取Cn_T_C0到numpy数组 =====")
        print(f"数组形状：{cn_data.shape}")  # 输出(889, 4, 4)
        print(f"数组类型：{cn_data.dtype}")  # 输出float64（<f8对应float64）
        print(f"第100个样本的数据：\n{cn_data[100]}")


# Grab the image itself
left_image = f['/ovc/left/data'][args.idx]

# Find the index in the event stream that correlates with the image time
left_event_idx = f['/ovc/ts_map_prophesee_left_t'][args.idx]

# Compute the start and stop index within the event stream
half_n_events = args.n_events // 2
start = left_event_idx - half_n_events
stop = left_event_idx + half_n_events

# Load 200k events surrounding the central timestamp
left_events_x = f['/prophesee/left/x'][start:stop]
left_events_y = f['/prophesee/left/y'][start:stop]
left_events_t = f['/prophesee/left/t'][start:stop]
left_events_p = f['/prophesee/left/p'][start:stop]