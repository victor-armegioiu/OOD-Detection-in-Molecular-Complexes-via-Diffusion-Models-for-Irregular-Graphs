import json
import shutil
from pathlib import Path

# --- user inputs ---
split_json = Path("single_use_scripts/training_validation_ids.json")           # your JSON file
pdbbind_root = Path("..", "PDBbind_v2020")            # folder with all subfolders
out_train = pdbbind_root.parent / "PDBbind_train"
out_valid = pdbbind_root.parent / "PDBbind_valid"

# --- load split file ---
with open(split_json) as f:
    split = json.load(f)

# --- helper to copy subsets ---
def copy_subset(ids, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for id_ in ids:
        src = pdbbind_root / id_
        dst = out_dir / id_
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            print(f"[WARN] {src} not found")

# --- perform copies ---
copy_subset(split["train"], out_train)
copy_subset(split["validation"], out_valid)

print(f"✅ Created {out_train} and {out_valid}")
