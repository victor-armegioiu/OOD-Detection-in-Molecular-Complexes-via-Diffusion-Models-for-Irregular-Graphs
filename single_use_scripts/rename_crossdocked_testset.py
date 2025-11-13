import os
import re

folder = "../benchmarks/processed_crossdocked/test"

for fname in os.listdir(folder):
    old_path = os.path.join(folder, fname)
    if not os.path.isfile(old_path):
        continue

    # Extract extension
    base, ext = os.path.splitext(fname)
    if ext.lower() not in [".pdb", ".sdf"]:
        continue

    # Split by "-" and keep first 8 segments
    parts = re.split(r"-", base)
    if len(parts) < 8:
        # Skip weird files
        continue

    new_base = "-".join(parts[:8])
    new_name = new_base + ext.lower()
    new_path = os.path.join(folder, new_name)

    # Handle collisions: if file exists, append increment
    counter = 1
    final_path = new_path
    while os.path.exists(final_path):
        final_path = os.path.join(folder, f"{new_base}_{counter}{ext.lower()}")
        counter += 1

    print(f"Renaming:\n  {fname}\n→ {os.path.basename(final_path)}")
    os.rename(old_path, final_path)
