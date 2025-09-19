from pathlib import Path
import sys

# Add the parent directory of 'moldiff' to sys.path
ROOT = Path(r"c:\Users\paesc\OneDrive\docs\master_thesis\code\molecular-diffusion")
def add_root_to_syspath():
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
        print(f"Added {ROOT} to sys.path")