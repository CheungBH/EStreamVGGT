import random
import shutil
from pathlib import Path


def split_subfolders(input_dir: str, output_dir: str, n_parts: int, seed: int):
    root = Path(input_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    subfolders = [p for p in root.iterdir() if p.is_dir()]
    if not subfolders or n_parts < 1:
        return
    # if seed is not None:
    #     random.seed(seed)
    # random.shuffle(subfolders)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n_parts):
        (out / f"part{i}").mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(subfolders):
        part = idx % n_parts
        dest_dir = out / f"part{part}"
        dest = dest_dir / src.name
        if dest.exists():
            dest = dest_dir / f"{src.name}_{idx}"
        shutil.move(str(src), str(dest))


if __name__ == "__main__":
    INPUT_DIR = "/path/to/folder_of_folders"
    OUTPUT_DIR = "output"
    N_PARTS = 5
    split_subfolders(INPUT_DIR, OUTPUT_DIR, N_PARTS, seed=42)