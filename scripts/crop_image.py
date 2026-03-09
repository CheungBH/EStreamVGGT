from pathlib import Path
import imageio.v2 as iio
import numpy as np

INPUT_DIR = "/path/to/folder"
OVERWRITE = True  # 设为 False 则以 *_top236.png 另存

def crop_top_236(img: np.ndarray) -> np.ndarray:
    h = img.shape[0]
    return img[:min(236, h)]

def save_cropped(src: Path):
    img = iio.imread(src)
    cropped = crop_top_236(img)
    if OVERWRITE:
        iio.imwrite(src, cropped)
        print(f"覆盖保存: {src}")
    else:
        dst = src.with_name(src.stem + "_top236.png")
        iio.imwrite(dst, cropped)
        print(f"另存为: {dst}")

def main():
    root = Path(INPUT_DIR).expanduser().resolve()
    targets = list(root.rglob("4_event.png")) + list(root.rglob("5_event.png"))
    if not targets:
        print("未找到目标文件")
        return
    for p in sorted(targets):
        try:
            save_cropped(p)
        except Exception as e:
            print(f"失败: {p} -> {e}")

if __name__ == "__main__":
    main()