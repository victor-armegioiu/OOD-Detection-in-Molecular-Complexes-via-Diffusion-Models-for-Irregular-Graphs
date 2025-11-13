import torch
from torch_geometric.loader import DataLoader
import json
import os
from collections import Counter

# attach path of root folder
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from moldiff.Dataset import PDBbind_Dataset



############
# Code below extracts IDs from torch file
############

def create_json_from_graph_dataset(
    dataset_train_path = "datasets/datasetmd_pdbbind_train.pt",
    dataset_val_path = "datasets/datasetmd_pdbbind_validation.pt",
    out_json_path = "single_use_scripts/training_validation_ids.json",
    is_MD: bool = True,
):


    def _load_dataset(path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return torch.load(path, map_location="cpu")


    def _extract_ids_from_loaded(ds):
        # try simple indexing/iteration first (common for lists / InMemoryDataset)
        ids = []
        try:
            if hasattr(ds, "__len__") and len(ds) > 0 and hasattr(ds[0], "id"):
                for item in ds:
                    ids.append(item.id)
                return ids
        except Exception:
            pass

        # fallback to building one full batch via DataLoader (matches original approach)
        try:
            batch_size = len(ds) if hasattr(ds, "__len__") and len(ds) > 0 else 1
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False, follow_batch=["lig_coords", "prot_coords"])
            batch = next(iter(dl))
            ids = batch.id
        except Exception as e:
            raise RuntimeError(f"Could not extract ids from dataset: {e}")

        # normalize to Python list
        if torch.is_tensor(ids):
            ids = ids.tolist()
        elif isinstance(ids, (tuple, set)):
            ids = list(ids)
        elif not isinstance(ids, list):
            ids = [ids]
        # convert any 0-dim tensors in list to scalars
        normalized = []
        for x in ids:
            if torch.is_tensor(x) and x.numel() == 1:
                normalized.append(x.item())
            else:
                normalized.append(x)
        return normalized


    def get_unique_ids(path, is_MD=False):
        ds = _load_dataset(path)
        ids = _extract_ids_from_loaded(ds)

        # If MD ids are long strings, normalize them by truncating to the first 4 chars
        if is_MD:
            ids = [str(x)[:4].lower() for x in ids]

        try:
            unique = sorted(set(ids))
        except TypeError:
            # mixed/unorderable types -> stringify for stable output
            unique = sorted(set(map(str, ids)))
        return unique


    train_ids = get_unique_ids(dataset_train_path, is_MD=is_MD)
    val_ids = get_unique_ids(dataset_val_path, is_MD=is_MD)

    output = {"train": train_ids, "validation": val_ids}

    with open(out_json_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Found {len(train_ids)} unique training ids and {len(val_ids)} unique validation ids.")
    print(f"Wrote ids to {out_json_path}")

############
# Code below compares two json files with training/validation ids
############

def compare_json_files(
    file_a = "json/PDBbind_data_split_pdbbind.json",
    file_b = "training_validation_ids.json"
):

    if not os.path.exists(file_a):
        raise FileNotFoundError(file_a)
    if not os.path.exists(file_b):
        raise FileNotFoundError(file_b)

    with open(file_a, "r") as fa, open(file_b, "r") as fb:
        ja = json.load(fa)
        jb = json.load(fb)

    def to_counter(lst):
        if not isinstance(lst, list):
            raise TypeError("expected a list")
        out = []
        for x in lst:
            try:
                s = json.dumps(x, sort_keys=True)
            except Exception:
                s = str(x)
            out.append(s)
        return Counter(out)

    for key in ("train", "validation"):
        if key not in ja or key not in jb:
            print(f"key '{key}' missing in one of the files")
            continue
        ca = to_counter(ja[key])
        cb = to_counter(jb[key])
        if ca == cb:
            print(f"{key}: MATCH ({sum(ca.values())} items)")
        else:
            print(f"{key}: MISMATCH")
            only_a = ca - cb
            only_b = cb - ca
            if only_a:
                print(f"  only in {file_a}: {list(only_a.elements())[:10]}{' ...' if sum(only_a.values())>10 else ''}")
            if only_b:
                print(f"  only in {file_b}: {list(only_b.elements())[:10]}{' ...' if sum(only_b.values())>10 else ''}")

if __name__ == "__main__":
    create_json_from_graph_dataset()
    # compare_json_files()
