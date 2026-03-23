import h5py

h5_path = "/Users/cheungbh/Documents/PhDCode/EStreamVGGT/data/car_urban_day_city_hall/car_urban_day_city_hall_data.h5"

with h5py.File(h5_path, 'r') as f:
    print("=== 根目录所有 keys ===")
    print(list(f.keys()))

    print("\n=== /ovc 组（如果存在）===")
    if '/ovc' in f:
        print(list(f['/ovc'].keys()))

        print("\n=== /ovc/rgb 组（如果存在）===")
        if '/ovc/rgb' in f['/ovc']:
            rgb_group = f['/ovc/rgb']
            print(list(rgb_group.keys()))
            # 检查 ts 和 ts_map
            if 'ts' in rgb_group:
                print("  /ovc/rgb/ts 存在，shape:", rgb_group['ts'].shape)
            if 'data' in rgb_group:
                print("  /ovc/rgb/data 存在，shape:", rgb_group['data'].shape)
            if 'ts_map_prophesee_left_t' in rgb_group:
                print("  /ovc/rgb/ts_map_prophesee_left_t 存在，shape:", rgb_group['ts_map_prophesee_left_t'].shape)

    print("\n=== 顶层 ts_map_prophesee_left_t 是否存在 ===")
    if '/ts_map_prophesee_left_t' in f:
        print("存在，shape:", f['/ts_map_prophesee_left_t'].shape)
    else:
        print("顶层无 ts_map_prophesee_left_t")

    print("\n=== /prophesee/left 组 ===")
    if '/prophesee/left' in f:
        print(list(f['/prophesee/left'].keys()))