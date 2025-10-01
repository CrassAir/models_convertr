import os
from pathlib import Path


def remap_line(line: str, max_class_id: int = 7) -> str:
    parts = line.strip().split()
    if not parts:
        return ""
    try:
        cid = int(parts[0])
    except ValueError:
        return line

    # If class id outside [0, max_class_id], try to fold 8..15 back into 0..7
    if cid > max_class_id:
        if 8 <= cid <= 15:
            cid = cid - 8
        else:
            # Fallback: modulo into range
            cid = cid % (max_class_id + 1)
        parts[0] = str(cid)
        return " ".join(parts)
    return line


def fix_dir(labels_dir: Path, max_class_id: int = 7) -> int:
    fixed_files = 0
    for txt in labels_dir.glob("*.txt"):
        with open(txt, "r", encoding="utf-8") as f:
            lines = f.readlines()

        changed = False
        new_lines = []
        for ln in lines:
            orig = ln.rstrip("\n")
            new = remap_line(orig, max_class_id)
            new_lines.append(new + "\n")
            if new != orig:
                changed = True

        if changed:
            with open(txt, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            fixed_files += 1
    return fixed_files


if __name__ == "__main__":
    # Dataset root relative to this script
    root = Path(__file__).resolve().parent / "ttn_dataset"
    train_labels = root / "train" / "labels"
    val_labels = root / "val" / "labels"

    total_fixed = 0
    for d in [train_labels, val_labels]:
        if d.exists():
            fixed = fix_dir(d, max_class_id=7)
            print(f"Fixed {fixed} files in {d}")
            total_fixed += fixed
        else:
            print(f"Skip missing: {d}")

    # Remove possible caches so YOLO rebuilds them
    for cache in [root / "train" / "labels.cache", root / "val" / "labels.cache"]:
        if cache.exists():
            try:
                os.remove(cache)
                print(f"Removed cache: {cache}")
            except OSError:
                pass

    print(f"Done. Total files changed: {total_fixed}")


