from posebusters.modules.intermolecular_distance import check_intermolecular_distance
#  posebusters.modules.intermolecular_distance.check_intermolecular_distance(mol_pred: Mol, mol_cond: Mol, radius_type: str = 'vdw', radius_scale: float = 1.0, clash_cutoff: float = 0.75, ignore_types: set[str] = {'hydrogens'}, max_distance: float = 5.0, search_distance: float = 6.0) → dict[str, Any]
from rdkit import Chem, RDLogger
from pathlib import Path

def example_clash_detection():

    sample_idx = 1
    pocket_id = "14gs-A-rec-20gs-cbd-lig-tt-min"
    sample_root = Path("../benchmarks/ours/ours_samples/")
    ligand_path = sample_root / pocket_id / f"{sample_idx}_ligand.sdf"
    pocket_path = sample_root / pocket_id / f"{sample_idx}_pocket.pdb"

    ligand = Chem.SDMolSupplier(str(ligand_path), sanitize=False)[0]
    pocket = Chem.MolFromPDBFile(str(pocket_path), sanitize=False, removeHs=False)

    print(type(ligand), type(pocket))

    results = check_intermolecular_distance(mol_pred=ligand, mol_cond=pocket)

    print(results["results"])
    print(results["details"][results["details"]["clash"]]) # gets us all clashing atoms

from pathlib import Path
from rdkit import Chem

import requests
from pathlib import Path
from rdkit import Chem

def fetch_full_pdb(pdb_id: str) -> str:
    """
    Download full PDB from RCSB and return raw text.
    pdb_id should be 4-character string such as '1H0I'.
    """
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    r = requests.get(url)
    if r.status_code != 200:
        raise RuntimeError(f"Could not download PDB {pdb_id} from RCSB")
    return r.text


def extract_pdb_id(complex_id: str) -> str:
    """
    Extract the 4-character PDB ID from your filename.
    Example: '14gs-A-rec-20gs-cbd-lig-tt-min' → '14gs'
    """
    first = complex_id.split("-")[0]
    if len(first) != 4:
        raise ValueError(f"Cannot interpret '{first}' as PDB ID")
    return first


def build_complex_pdb_from_id(
    complex_id: str,
    lig_sdf: Path,
    out_pdb: Path,
    lig_resname: str = None,
    lig_chain: str = "Z",
    lig_resid: int = 1,
):
    """
    1) Extract true PDB ID from complex_id
    2) Fetch the full receptor from RCSB
    3) Insert the ligand (from SDF) as a new residue
    4) Return full complex PDB path
    """

    # --- 1. extract pdb id ---
    pdb_id = extract_pdb_id(complex_id)
    print(f"[INFO] Using PDB ID: {pdb_id}")

    # --- 2. download full receptor ---
    pdb_text = fetch_full_pdb(pdb_id)
    rec_lines = [
        l for l in pdb_text.splitlines()
        if l.startswith(("ATOM", "HETATM", "TER", "END"))
    ]

    # --- find max atom serial in receptor ---
    rec_serials = [
        int(l[6:11]) for l in rec_lines
        if l.startswith(("ATOM", "HETATM")) and l[6:11].strip().isdigit()
    ]
    start_serial = max(rec_serials) if rec_serials else 0

    # --- 3. load ligand ---
    suppl = Chem.SDMolSupplier(str(lig_sdf), sanitize=False)
    lig = suppl[0]
    if lig is None:
        raise ValueError(f"Could not read ligand from {lig_sdf}")

    # use ligand name from complex_id if not provided
    if lig_resname is None:
        # take e.g. 'cbd' from '14gs-A-rec-20gs-cbd-lig-tt-min'
        parts = complex_id.split("-")
        lig_res = parts[3]  # the ligand name
        lig_resname = lig_res[:3].upper()

    # --- assign residue info to ligand atoms ---
    for atom in lig.GetAtoms():
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName(lig_resname)
        info.SetChainId(lig_chain)
        info.SetResidueNumber(lig_resid)
        info.SetName(atom.GetSymbol().rjust(4))  # crude but dpocket-compatible
        atom.SetPDBResidueInfo(info)

    lig_block = Chem.MolToPDBBlock(lig)
    lig_lines_raw = [
        l for l in lig_block.splitlines()
        if l.startswith(("ATOM", "HETATM"))
    ]

    # --- renumber ligand atoms ---
    lig_lines = []
    serial = start_serial
    for l in lig_lines_raw:
        serial += 1
        new_line = f"{l[:6]}{serial:5d}{l[11:]}"
        lig_lines.append(new_line)

    # --- 4. write full complex PDB ---
    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    with open(out_pdb, "w") as f:
        for l in rec_lines:
            if not l.startswith("END"):
                f.write(l + "\n")
        for l in lig_lines:
            f.write(l + "\n")
        f.write("END\n")

    print(f"[INFO] Complex written to: {out_pdb}")
    return out_pdb


complex_id = "14gs-A-rec-20gs-cbd-lig-tt-min"

lig_sdf = Path(f"../benchmarks/processed_crossdocked/test/{complex_id}.sdf")
out_pdb = Path("tmp/14gs_complex.pdb")

# out= build_complex_pdb_from_id(
#     complex_id,
#     lig_sdf,
#     out_pdb,
# )

import pandas as pd



df = pd.read_csv("dpout_explicitp.txt", sep="\s+")
df2 = pd.read_csv("dpout_fpocketnp.txt", sep="\s+")
df3 = pd.read_csv("dpout_fpocketp.txt", sep="\s+")


df5 = pd.concat([df, df2, df3], axis=0)

print(df.columns)
example_clash_detection()








