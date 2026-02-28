import re
from pathlib import Path
from typing import Dict, List, Tuple


INPUT_DIR = "/path/to/your/waymo/frames"  # 修改为你的图片目录
OUTPUT_DIR = "/path/to/your/output/videos"  # 修改为你的输出目录
FPS = 10
CODEC = "mp4v"
EXTS = ["jpg", "jpeg", "png"]
PATTERN = r"^(?P<frame>\d+)[_\-](?P<view>\d+)\.(?P<ext>.+)$"
PREFIX = "view"
DRY_RUN = False


def list_images(input_dir: Path, exts: List[str]) -> List[Path]:
    files = []
    for ext in exts:
        files.extend(sorted(input_dir.glob(f"*.{ext}")))
        files.extend(sorted(input_dir.glob(f"*.{ext.upper()}")))
    files = sorted(set(files))
    return files


def parse_name(path: Path, pattern: re.Pattern) -> Tuple[int, int]:
    m = pattern.match(path.name)
    if not m:
        raise ValueError(f"无法解析文件名: {path.name}")
    frame = int(m.group("frame"))
    view = int(m.group("view"))
    return frame, view


def group_by_view(paths: List[Path], pattern: re.Pattern) -> Dict[int, List[Tuple[int, Path]]]:
    groups: Dict[int, List[Tuple[int, Path]]] = {}
    for p in paths:
        try:
            frame, view = parse_name(p, pattern)
        except Exception:
            continue
        groups.setdefault(view, []).append((frame, p))
    for v in groups:
        groups[v].sort(key=lambda x: x[0])
    return groups


def write_videos_with_cv2(groups: Dict[int, List[Tuple[int, Path]]], out_dir: Path, fps: int, codec: str, prefix: str):
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    for view, frames in sorted(groups.items()):
        first_img = None
        for _, p in frames:
            img = cv2.imread(str(p))
            if img is not None:
                first_img = img
                break
        if first_img is None:
            print(f"[跳过] 视角 {view}: 找不到可读首帧")
            continue
        h, w = first_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out_path = out_dir / f"{prefix}_{view}.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if not writer.isOpened():
            print(f"[警告] 无法打开写入器: {out_path}")
            continue
        for _, p in frames:
            img = cv2.imread(str(p))
            if img is None:
                continue
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h))
            writer.write(img)
        writer.release()
        print(f"[完成] 写出 {out_path}，帧数 {len(frames)}")


def write_videos_with_imageio(groups: Dict[int, List[Tuple[int, Path]]], out_dir: Path, fps: int, prefix: str):
    import imageio.v2 as iio
    out_dir.mkdir(parents=True, exist_ok=True)
    for view, frames in sorted(groups.items()):
        out_path = out_dir / f"{prefix}_{view}.mp4"
        try:
            with iio.get_writer(str(out_path), fps=fps) as writer:
                for _, p in frames:
                    try:
                        img = iio.imread(p)
                    except Exception:
                        continue
                    writer.append_data(img)
            print(f"[完成] 写出 {out_path}，帧数 {len(frames)}")
        except Exception as e:
            print(f"[失败] 视角 {view}: {e}")


def main():
    input_dir = Path(INPUT_DIR).expanduser().resolve()
    if not input_dir.exists():
        print("输入目录不存在")
        return
    output_dir = Path(OUTPUT_DIR).expanduser().resolve()
    exts = [e.strip(".").lower() for e in EXTS if str(e).strip()]
    pattern = re.compile(PATTERN)
    imgs = list_images(input_dir, exts)
    groups = group_by_view(imgs, pattern)
    if not groups:
        print("未找到可解析的图片")
        return
    for v, items in sorted(groups.items()):
        print(f"视角 {v}: {len(items)} 帧，示例 {items[0][1].name}")
    if DRY_RUN:
        print("干运行完成")
        return
    tried_cv2 = False
    try:
        import cv2  # noqa: F401
        tried_cv2 = True
    except Exception:
        tried_cv2 = False
    if tried_cv2:
        try:
            write_videos_with_cv2(groups, output_dir, FPS, CODEC, PREFIX)
            return
        except Exception as e:
            print(f"[警告] OpenCV 写视频失败，改用 imageio: {e}")
    write_videos_with_imageio(groups, output_dir, FPS, PREFIX)


if __name__ == "__main__":
    main()
