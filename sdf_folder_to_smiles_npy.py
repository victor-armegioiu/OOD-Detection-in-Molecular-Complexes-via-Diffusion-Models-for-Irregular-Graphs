#!/usr/bin/env python3
from rdkit import Chem
from pathlib import Path
import numpy as np
import argparse, glob
from tqdm import tqdm

def rdmol_to_smiles(rdmol):
    mol = Chem.Mol(rdmol)
    Chem.RemoveStereochemistry(mol)
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)  # canonicalization is implicit; stereochem already stripped

def main(data_dir):
    data_dir = Path(data_dir)
    sdf_files = glob.glob(str(data_dir / "**/*.sdf"), recursive=True)
    if not sdf_files:
        raise SystemExit(f"No SDF files found under {data_dir}")

    seen = set()
    n_total = 0
    n_valid = 0

    for sdf in tqdm(sdf_files, desc="SDF"):
        suppl = Chem.SDMolSupplier(sdf, removeHs=False, sanitize=True)
        for m in suppl:
            n_total += 1
            if m is None:
                continue
            try:
                smi = rdmol_to_smiles(m)
            except Exception:
                continue
            if smi:
                n_valid += 1
                seen.add(smi)

    out = data_dir / "train_smiles.npy"
    np.save(out, np.array(sorted(seen), dtype=object))
    print(f"Wrote {out} with {len(seen)} unique SMILES "
          f"(from {n_valid}/{n_total} valid molecules)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="Directory containing all training SDFs (scanned recursively)")
    args = ap.parse_args()
    main(args.data_dir)
