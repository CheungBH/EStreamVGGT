from pathlib import Path
import shutil

def move_to_folder(src_folder: str, target_folder: str):
    src = Path(src_folder).expanduser().resolve()
    tgt = Path(target_folder).expanduser().resolve()
    if not src.exists() or not tgt.exists():
        return
    subs_src = {p.name for p in src.iterdir() if p.is_dir()}
    subs_tgt = {p.name for p in tgt.iterdir() if p.is_dir()}
    common = sorted(subs_src & subs_tgt)
    for name in common:
        sdir = src / name
        tdir = tgt / name
        for p in sdir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(sdir)
                dest = tdir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(p), str(dest))

if __name__ == "__main__":
    SRC = "/path/to/src_folder"
    TGT = "/path/to/target_folder"
    move_to_folder(SRC, TGT)
