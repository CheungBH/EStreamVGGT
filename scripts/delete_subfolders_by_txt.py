from pathlib import Path
import shutil

def delete_by_txt(roots: list[str], names_txt: str, dry_run: bool = False):
    nt = Path(names_txt).expanduser().resolve()
    if not nt.exists():
        print("名单文件不存在")
        return
    names = [line.strip() for line in nt.read_text().splitlines() if line.strip()]
    total = 0
    for r in roots:
        root = Path(r).expanduser().resolve()
        if not root.exists():
            continue
        for name in names:
            target = root / name
            if target.exists() and target.is_dir():
                total += 1
                print(f"删除 {target}" if not dry_run else f"预览 {target}")
                if not dry_run:
                    shutil.rmtree(target)
    print(f"计数 {total}")

if __name__ == "__main__":
    FOLDERS_LIST = ["/path/to/root1", "/path/to/root2"]
    NAMES_TXT = "mismatch.txt"
    delete_by_txt(FOLDERS_LIST, NAMES_TXT, dry_run=False)
