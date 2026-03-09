from pathlib import Path

def compare_subfolder_counts(dir_a: str, dir_b: str, img_subdir: str = "img", exts: tuple | None = None):
    A = Path(dir_a).expanduser().resolve()
    B = Path(dir_b).expanduser().resolve()
    if not A.exists() or not B.exists():
        print("目录不存在")
        return

    # 子文件夹集合（只比较两边都存在的）
    subs_a = {p.name for p in A.iterdir() if p.is_dir()}
    subs_b = {p.name for p in B.iterdir() if p.is_dir()}
    common = sorted(subs_a & subs_b)
    if not common:
        print("无同名子文件夹")
        return

    def count_files(p: Path):
        if not p.exists():
            return 0
        if exts is None:
            return sum(1 for x in p.iterdir() if x.is_file())
        exts_lc = tuple(e.lower() for e in exts)
        return sum(1 for x in p.iterdir() if x.is_file() and x.suffix.lower().lstrip(".") in exts_lc)

    mismatches = 0
    for name in common:
        a_img_dir = A / name / img_subdir
        b_dir = B / name
        ca = count_files(a_img_dir)
        cb = count_files(b_dir)
        if ca != cb:
            mismatches += 1
            print(f"{name}: A({img_subdir})={ca}, B={cb}")

    print(f"不一致的子文件夹数量: {mismatches}")

if __name__ == "__main__":
    DIR_A = "/path/to/A"
    DIR_B = "/path/to/B"
    compare_subfolder_counts(DIR_A, DIR_B, img_subdir="img", exts=None)