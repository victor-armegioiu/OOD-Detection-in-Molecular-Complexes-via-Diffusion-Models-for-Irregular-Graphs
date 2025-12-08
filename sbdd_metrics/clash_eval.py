from pathlib import Path
import subprocess
import re
import os

import pandas as pd
from rdkit import Chem
import gemmi

from posebusters.modules.intermolecular_distance import check_intermolecular_distance

from tqdm import tqdm

def make_relative_path(path: Path, base: Path) -> Path:
    """
    Return `path` expressed relative to directory `base`.
    
    If `path` is inside `base`, use Path.relative_to() for a clean relative path.
    Otherwise fall back to os.path.relpath().

    Parameters
    ----------
    path : Path
        The absolute or arbitrary filesystem path to convert.
    base : Path
        The reference directory (e.g., cwd of a subprocess).

    Returns
    -------
    Path
        A relative path from `base` to `path`.
    """
    path = Path(path)
    base = Path(base)

    try:
        # Cleanest form (no "../.."), only works when inside base
        return path.relative_to(base)
    except ValueError:
        # General fallback: produce a valid relative path even across directories
        return Path(os.path.relpath(path, base))


def classify_residue(resname: str) -> str:
    r = resname.upper()
    if r in {"ARG", "LYS", "HIS"}:
        return "charged_pos"
    if r in {"ASP", "GLU"}:
        return "charged_neg"
    if r in {"SER", "THR", "ASN", "GLN"}:
        return "polar"
    if r in {"PHE", "TYR", "TRP"}:
        return "aromatic"
    if r in {"ALA", "VAL", "LEU", "ILE", "MET", "PRO"}:
        return "hydrophobic"
    if r in {"CYS"}:
        return "special"
    return "other"

def build_protein_atom_table_gemmi(pocket_pdb: Path) -> pd.DataFrame:
    """
    Build a table mapping protein_atom_id -> residue + atom metadata using gemmi.
    Assumes atom order matches what was used for clash detection.
    """
    st = gemmi.read_structure(str(pocket_pdb))
    model = st[0]  # single model
    records = []
    atom_idx = 0

    for chain in model:
        chain_id = chain.name
        for res in chain:
            resname = res.name
            resid = res.seqid.num
            for atom in res:
                atom_name = atom.name.strip()
                is_backbone = atom_name in {"N", "CA", "C", "O"}
                records.append(
                    {
                        "protein_atom_id": atom_idx,
                        "protein_atom_name": atom_name,
                        "protein_resname": resname,
                        "protein_resid": resid,
                        "protein_chain": chain_id,
                        "protein_is_backbone": is_backbone,
                        "protein_region": "backbone" if is_backbone else "sidechain",
                        "protein_res_class": classify_residue(resname),
                    }
                )
                atom_idx += 1

    return pd.DataFrame.from_records(records)




def build_dpocket_ready_complex(
    crossdocked_root: Path,
    pocket_id: str,
    lig_resname: str = None,
    lig_chain: str = "Z",
    lig_resid: int = 1,
    out_root: Path|None = None
) -> Path:
    """
    Fuse CrossDocked receptor (pocket_id.pdb) and ligand (pocket_id.sdf)
    into a single PDB suitable for dpocket.

    Returns path to the complex PDB.
    """
    if out_root is None:
        out_root = crossdocked_root

    rec_pdb = crossdocked_root / f"{pocket_id}.pdb"
    lig_sdf = crossdocked_root / f"{pocket_id}.sdf"
    if not rec_pdb.exists():
        raise FileNotFoundError(f"Receptor PDB not found: {rec_pdb}")
    if not lig_sdf.exists():
        raise FileNotFoundError(f"Ligand SDF not found: {lig_sdf}")

    out_pdb = out_root / f"{pocket_id}_complex_for_dpocket.pdb"

    if out_pdb.exists():
        return out_pdb

    # --- read receptor PDB as text ---
    with open(rec_pdb, "r") as f:
        rec_lines = [
            l.rstrip("\n") for l in f
            if l.startswith(("ATOM", "HETATM", "TER", "END"))
        ]

    # --- max atom serial ---
    rec_serials = [
        int(l[6:11]) for l in rec_lines
        if l.startswith(("ATOM", "HETATM")) and l[6:11].strip().isdigit()
    ]
    start_serial = max(rec_serials) if rec_serials else 0

    # --- ligand from SDF ---
    suppl = Chem.SDMolSupplier(str(lig_sdf), sanitize=False)
    lig = suppl[0]
    if lig is None:
        raise ValueError(f"Could not read ligand from {lig_sdf}")

    if lig_resname is None:
        # heuristic: take 4th field from pocket_id "14gs-A-rec-20gs-cbd-lig-tt-min"
        parts = pocket_id.split("-")
        if len(parts) >= 5:
            lig_resname = parts[4][:3].upper()
        else:
            lig_resname = "LIG"

    lig_res = lig_resname[:3].upper()

    # --- assign PDB residue info to ligand atoms ---
    for atom in lig.GetAtoms():
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName(lig_res)
        info.SetChainId(lig_chain)
        info.SetResidueNumber(lig_resid)
        info.SetName(atom.GetSymbol().rjust(4))
        atom.SetPDBResidueInfo(info)

    lig_block = Chem.MolToPDBBlock(lig)
    lig_raw_lines = [
        l.rstrip("\n")
        for l in lig_block.splitlines()
        if l.startswith(("ATOM", "HETATM"))
    ]

    # --- renumber ligand atoms ---
    lig_lines = []
    serial = start_serial
    for line in lig_raw_lines:
        serial += 1
        new_line = f"{line[:6]}{serial:5d}{line[11:]}"
        lig_lines.append(new_line)

    # --- write complex ---
    with open(out_pdb, "w") as f:
        for l in rec_lines:
            if not l.startswith("END"):
                f.write(l + "\n")
        for l in lig_lines:
            f.write(l + "\n")
        f.write("END\n")

    return out_pdb

def build_all_dpocket_complexes(
    crossdocked_root: Path,
    output_root: Path,
    dpocket_input_path: Path
):
    output_root.mkdir(parents=True, exist_ok=True)
    dpocket_lines = []

    for pdb_file in sorted(crossdocked_root.glob("*.pdb")):
        pocket_id = pdb_file.stem
        lig_sdf = crossdocked_root / f"{pocket_id}.sdf"
        if not lig_sdf.exists():
            print(f"[WARN] No ligand SDF for {pocket_id}")
            continue

        # Correct: CALL WITH out_root
        complex_pdb = output_root / f"{pocket_id}_complex_for_dpocket.pdb"

        complex_pdb = build_dpocket_ready_complex(
            crossdocked_root, pocket_id, out_root=output_root
        )
        # relative update for subprocess logic in dpocket later on
        complex_pdb_rel = make_relative_path(complex_pdb, output_root)
    

        # extract ligand name
        parts = pocket_id.split("-")
        lig_resname = parts[4].upper() if len(parts) >= 5 else "LIG"

        dpocket_lines.append(f"{complex_pdb_rel}\t{lig_resname}")

    with open(dpocket_input_path, "w") as f:
        f.write("\n".join(dpocket_lines))

    print(f"[OK] Wrote dpocket master input: {dpocket_input_path}")




import os

def run_dpocket_once(dpocket_input_path: Path, output_root: Path, prefix="dpout_all"):
    """
    Run dpocket only once and ensure the dpocket_input_path
    is correctly referenced relative to output_root.
    """
    output_root = Path(output_root)
    explicit_out = output_root / f"{prefix}_exp.txt"

    # print(dpocket_input_path)
    # print(output_root)
    # print(explicit_out)

    # Skip if dpocket already ran
    if explicit_out.exists():
        print("[OK] dpocket results already exist, skipping.")
        return

    # Compute the path relative to output_root (subprocess cwd)
    dpocket_input_rel = make_relative_path(dpocket_input_path, output_root)
    
    # print(dpocket_input_rel)

    print(f"[INFO] Running dpocket with input: {dpocket_input_rel}")

    subprocess.run(
        ["dpocket", "-f", str(dpocket_input_rel), "-o", prefix],
        cwd=str(output_root),
        check=True
    )

    print("[OK] dpocket executed successfully.")



def load_all_dpocket_descriptors(explicit_path: Path) -> pd.DataFrame:
    """
    Load dpocket explicit pocket descriptors for ALL pocket_ids.
    dpocket writes one row per input pdb, in the same order.
    """
    import os
    # print(os.getcwd())
    df = pd.read_csv(explicit_path, sep="\s+", engine="python")

    # Extract pocket_id from the pdb path column
    df["pdb"] = df["pdb"].astype(str)
    df["pocket_id"] = df["pdb"].apply(lambda x: Path(x).stem.replace("_complex_for_dpocket", ""))

    return df.set_index("pocket_id", drop=False)

def evaluate_all_clashes_and_merge(
    ours_root: Path,
    dpocket_df: pd.DataFrame
) -> pd.DataFrame:

    all_rows = []

    for pocket_dir in tqdm(sorted(ours_root.iterdir())):
        if not pocket_dir.is_dir():
            continue

        pocket_id = pocket_dir.name

        # Load dpocket row (one per pocket_id)
        if pocket_id not in dpocket_df.index:
            print(f"[WARN] No dpocket descriptors for {pocket_id}")
            dp_desc = None
        else:
            dp_desc = dpocket_df.loc[pocket_id].to_dict()

        # Evaluate all sample ligands
        for lig_file in sorted(pocket_dir.glob("*_ligand.sdf")):
            m = re.match(r"(\d+)_ligand\.sdf", lig_file.name)
            if not m:
                continue
            sample_idx = int(m.group(1))
            pocket_pdb = pocket_dir / f"{sample_idx}_pocket.pdb"

            ligand = Chem.SDMolSupplier(str(lig_file), sanitize=False)[0]
            pocket_mol = Chem.MolFromPDBFile(str(pocket_pdb), sanitize=False, removeHs=False)

            if ligand is None or pocket_mol is None:
                continue

            result = check_intermolecular_distance(mol_pred=ligand, mol_cond=pocket_mol)
            details = result["details"]
            clashes = details[details["clash"]].copy()
            if clashes.empty:
                continue

            # annotate with gemmi
            atom_table = build_protein_atom_table_gemmi(pocket_pdb)
            clashes = clashes.merge(atom_table, on="protein_atom_id", how="left")

            # summary info
            for k, v in result["results"].items():
                clashes[f"clash_summary_{k}"] = v

            # unique sample id
            clashes["sample_id"] = f"{pocket_id}_{sample_idx}"
            clashes["pocket_id"] = pocket_id
            clashes["sample_idx"] = sample_idx

            # add dpocket descriptors
            if dp_desc is not None:
                for k, v in dp_desc.items():
                    clashes[f"dpocket_{k}"] = v

            all_rows.append(clashes)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, axis=0, ignore_index=True)

def full_pipeline(
    ours_samples_root: Path,
    crossdocked_root: Path,
    output_root: Path
):
    output_root.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: build complexes + master input ----
    dpocket_input = output_root / "dpocket_input_all.txt"

    build_all_dpocket_complexes(
        crossdocked_root=crossdocked_root,
        output_root=output_root,
        dpocket_input_path=dpocket_input
    )

    # ---- Stage 2: run dpocket once ----
    run_dpocket_once(
        dpocket_input_path=dpocket_input,
        output_root=output_root,
        prefix="dpout_all"
    )

    explicit_path = output_root / "dpout_all_exp.txt"

    # ---- Stage 3: load dpocket descriptor table ----
    dpocket_df = load_all_dpocket_descriptors(explicit_path)

    # ---- Stage 4: evaluate all clashes ----
    final_df = evaluate_all_clashes_and_merge(
        ours_root=ours_samples_root,
        dpocket_df=dpocket_df
    )

    final_df.to_csv(output_root / "full_clash_dpocket_merged.csv", index=False)
    print("[OK] Final dataframe saved.")

    return final_df



if __name__ == "__main__":
    ours_root = Path("../benchmarks/ours/ours_samples/")
    crossdocked_root = Path("../benchmarks/processed_crossdocked/test/")
    out_root = Path("../benchmarks/processed_crossdocked/dpocket_test/")

    final_df = full_pipeline(ours_root, crossdocked_root, out_root)
    # final_df.to_csv("clash_dpocket_annotated.csv", index=False)
    print(final_df.head())

