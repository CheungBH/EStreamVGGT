import re
import shutil
from pathlib import Path


def process(input_folder: str, output_folder: str):
    in_root = Path(input_folder).expanduser().resolve()
    out_root = Path(output_folder).expanduser().resolve()
    if not in_root.exists():
        return
    out_root.mkdir(parents=True, exist_ok=True)
    rg = re.compile(r"^(?P<root>.+)_view(?P<view>\d+)$")
    for sub in sorted([p for p in in_root.iterdir() if p.is_dir()]):
        m = rg.match(sub.name)
        if not m:
            continue
        root = m.group("root")
        view = m.group("view")
        dst_dir = out_root / root
        dst_dir.mkdir(parents=True, exist_ok=True)
        imgs = sorted(sub.glob("*.png"))
        for p in imgs:
            nm = p.stem
            try:
                idx = int(nm)
            except Exception:
                continue
            new_name = f"{str(idx + 1).zfill(5)}_{view}_event.png"
            dst = dst_dir / new_name
            if dst.exists():
                continue
            shutil.copy2(str(p), str(dst))


if __name__ == "__main__":
    INPUT_FOLDER = "/path/to/input"
    OUTPUT_FOLDER = "/path/to/output"
    process(INPUT_FOLDER, OUTPUT_FOLDER)
