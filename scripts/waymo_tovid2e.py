import re
from pathlib import Path
from typing import Dict, List, Tuple

FPS = 10
EXTS = ["jpg", "jpeg", "png"]
PATTERN = r"^(?P<frame>\d+)[_\-](?P<view>\d+)\.(?P<ext>.+)$"
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
        raise ValueError(path.name)
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


def export_frame_folders(groups: Dict[int, List[Tuple[int, Path]]], out_root: Path, tag: str, fps: int):
    import imageio.v2 as iio
    out_root.mkdir(parents=True, exist_ok=True)
    for view, frames in sorted(groups.items()):
        view_dir = out_root / f"{tag}_view{view}"
        if view_dir.exists():
            print(f"{view_dir} skipped")
            continue
        imgs_dir = view_dir / "imgs"
        imgs_dir.mkdir(parents=True, exist_ok=True)
        (view_dir / "fps.txt").write_text(str(fps))
        idx = 1
        for _, p in frames:
            try:
                img = iio.imread(p)
            except Exception:
                continue
            out_path = imgs_dir / f"img_{idx:08d}.png"
            iio.imwrite(out_path, img)
            idx += 1
        print(f"{view_dir} {idx-1}")


def main(input_dir, output_root):
    input_dir = Path(input_dir).expanduser().resolve()
    if not input_dir.exists():
        print("输入目录不存在")
        return
    output_root = Path(output_root).expanduser().resolve()
    exts = [e.strip(".").lower() for e in EXTS if str(e).strip()]
    pattern = re.compile(PATTERN)
    imgs = list_images(input_dir, exts)
    groups = group_by_view(imgs, pattern)
    if not groups:
        print("未找到可解析的图片")
        return
    if DRY_RUN:
        for v, items in sorted(groups.items()):
            print(f"{input_dir.name}_view{v} {len(items)}")
        return
    export_frame_folders(groups, output_root, input_dir.name, FPS)


if __name__ == "__main__":
    ROOT_DIR = "/path/to/folder_of_folders"
    OUTPUT_ROOT = "/path/to/output"
    root_path = Path(ROOT_DIR).expanduser().resolve()
    out_root = Path(OUTPUT_ROOT).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    subfolders = sorted([p for p in root_path.iterdir() if p.is_dir()])
    if not subfolders:
        print("未找到子目录")
    for sub in subfolders:
        print(sub.name)
        main(sub, out_root)
