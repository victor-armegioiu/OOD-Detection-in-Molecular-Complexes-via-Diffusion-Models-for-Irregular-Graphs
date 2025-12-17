from pathlib import Path
import subprocess
import re
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from typing import List, Dict
from itertools import combinations

import pandas as pd
import numpy as np
from rdkit import Chem
import gemmi

from posebusters.modules.intermolecular_distance import check_intermolecular_distance
from posebusters import PoseBusters

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import pyKVFinder

from scipy.stats import chisquare, kruskal, f_oneway, wasserstein_distance
# https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.stats.chisquare.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
from statsmodels.stats.contingency_tables import cochrans_q
# https://www.statsmodels.org/dev/generated/statsmodels.stats.contingency_tables.cochrans_q.html
from scikit_posthocs import posthoc_dunn
# https://scikit-posthocs.readthedocs.io/en/latest/generated/scikit_posthocs.posthoc_dunn.html


os.environ["KMP_WARNINGS"] = "0"



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
    Build a table mapping protein_atom_id -> residue + atom metadata,
    including distance to the residue CA atom.
    
    Assumes atom order matches what was used for clash detection.
    """
    st = gemmi.read_structure(str(pocket_pdb))
    model = st[0]  # single model

    records = []
    atom_idx = 0

    # --- 1) cache CA coordinates per residue ---
    ca_coords = {}  # key: (chain_id, resid) -> np.array([x,y,z])

    for chain in model:
        for res in chain:
            for atom in res:
                if atom.name.strip() == "CA":
                    ca_coords[(chain.name, res.seqid.num)] = np.array(
                        [atom.pos.x, atom.pos.y, atom.pos.z]
                    )
                    break  # one CA per residue

    # --- 2) build atom table with CA distances ---
    for chain in model:
        chain_id = chain.name
        for res in chain:
            resname = res.name
            resid = res.seqid.num
            ca = ca_coords.get((chain_id, resid), None)

            for atom in res:
                atom_name = atom.name.strip()
                is_backbone = atom_name in {"N", "CA", "C", "O"}

                atom_pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z])

                if ca is None:
                    dist_to_ca = np.nan
                else:
                    dist_to_ca = float(np.linalg.norm(atom_pos - ca))

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
                        "protein_dist_to_ca": dist_to_ca,
                    }
                )
                atom_idx += 1

    return pd.DataFrame.from_records(records)




def build_dpocket_ready_complex(
    rec_pdb: Path, 
    lig_sdf: Path, 
    out_root: Path,
    # crossdocked_root: Path,
    # pocket_id: str,
    lig_resname: str = None,
    lig_chain: str = "Z",
    lig_resid: int = 1
) -> Path:
    """
    Fuse CrossDocked receptor (pocket_id.pdb) and ligand (pocket_id.sdf)
    into a single PDB suitable for dpocket.

    Returns path to the complex PDB.
    """

    sample_id = lig_sdf.name.split("_")[0]
    pocket_id = lig_sdf.parent.name

    if not rec_pdb.exists():
        raise FileNotFoundError(f"Receptor PDB not found: {rec_pdb}")
    if not lig_sdf.exists():
        raise FileNotFoundError(f"Ligand SDF not found: {lig_sdf}")

    out_pdb = out_root / f"{pocket_id}_{sample_id}_complex_for_dpocket.pdb"

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



def run_dpocket(dpocket_input_file: Path, dpocket_processing_dir: Path, dpocket_detection_radius:int = 4):
    """
    Run dpocket only once and ensure the dpocket_input_path
    is correctly referenced relative to output_root.
    """

    # print(f"[INFO] dpocket function received prefix {prefix}")


    # explicit_out = dpocket_processing_dir / f"{prefix}_exp.txt"

    # # print(dpocket_input_path)
    # # print(output_root)
    # # print(explicit_out)

    # # # Skip if dpocket already ran
    # if explicit_out.exists():
    #     print("[OK] dpocket results already exist, skipping.")
    #     return

    # Compute the path relative to output_root (subprocess cwd)
    dpocket_input_rel = make_relative_path(dpocket_input_file, dpocket_processing_dir)
    
    # print(dpocket_input_rel)

    print(f"[INFO] Running dpocket with input: {dpocket_input_rel}")
    # print("[INFO] dpocket ARGV =", ["dpocket", "-f", str(dpocket_input_rel),  "-d", str(dpocket_detection_radius)])
    # assert isinstance(prefix, str) and prefix and not prefix.startswith("-"), f"prefix is {prefix}"

    subprocess.run(
        ["dpocket", "-f", str(dpocket_input_rel), "-E", "-d", str(dpocket_detection_radius)],
        cwd=str(dpocket_processing_dir),
        check=True
    )

    # print(f"[OK] dpocket executed successfully for pocket_id {prefix}.")

def run_pyKVFinder(pocket_path:str, kv_args:Dict, return_df:bool = True):

    results = pyKVFinder.run_workflow(
        input=pocket_path, 
        # ligand="0_ligand.pdb", 
        **kv_args,
        include_depth=True, 
        include_hydropathy=True, 
        hydrophobicity_scale='EisenbergWeiss'
        )
    if results is None:
        return
    
    if return_df:
        residue_class_meaning_mapper = {
            "R1": "alipathic-apolar", 
            "R2": "aromatic", 
            "R3": "polar-uncharged", 
            "R4": "negatively-charged", 
            "R5": "positively-charged", 
            "RX": "Non-standard"
        }
        # Find biggest cavity and construct df
        pocket_id = max(results.volume, key=results.volume.get)
        row = {
            # "sample_id": f"{Path(pocket_path).parent.name}_{Path(pocket_path).stem[0]}" ,
            "pyKV_volume": results.volume.get(pocket_id),
            "pyKV_surface_area": results.area.get(pocket_id),
            "pyKV_avg_depth": results.avg_depth.get(pocket_id),
            "pyKV_max_depth": results.max_depth.get(pocket_id),
            "pyKV_avg_hydropathy": results.avg_hydropathy.get(pocket_id),
        } | {
            f"pyKV_{key}_{residue_class_meaning_mapper.get(subkey, subkey)}": value for key, item in results.frequencies[pocket_id].items() for subkey, value in item.items()
        }

        
        return row
    else:
        return results


def evaluate_class_worker(pocket_dir: Path, dpocket_complex_dir: Path, dpocket_detection_radius:int, pykv_args:Dict = {}) -> pd.DataFrame:
            

        if not pocket_dir.is_dir():
            print(f"[WARNING] returning None for {pocket_dir} with working directory {os.getcwd()}")
            return 

        pocket_id = pocket_dir.name


        lig_rows = []

        # # filtering ligand files that pass all filters prior to the clash filter
        # ligand_files = sorted(pocket_dir.glob("*_ligand.sdf"))
        # pocket_file = sorted(pocket_dir.glob("*_pocket.pdb"))[0]
        # metrics_file = Path(pocket_dir.parents[1]) / (pocket_dir.parent.name.split("_")[0] + "_metrics") / "metrics_detailed.csv"
        # prior_filters = ["sanitization", "all_atoms_connected", "aromatic_ring_flatness", "bond_angles", "bond_lengths", "double_bond_flatness", "internal_steric_clash"]

        # if metrics_file.exists():
        #     bust_result = pd.read_csv(metrics_file)
        #     rel_tail = ["/".join(p.parts[-2:]) for p in ligand_files]
        #     abs_tail = bust_result["sdf_file"].apply(lambda p: "/".join(Path(p).parts[-2: ]))
        #     mask = abs_tail.isin(rel_tail)
        #     # order absolute paths *in the order they appear in list_rel*
        #     order_idx = abs_tail[mask].map(
        #         {k: i for i, k in enumerate(rel_tail)}
        #     ).sort_values().index

        #     # ordered result
        #     updated_prior_filters = [f"posebusters.{s}" for s in prior_filters]
        #     filter_mask = bust_result.loc[order_idx, updated_prior_filters].all(axis=1).values
            
        # else:
        #     print("[INFO] No metrics file found, running PoseBusters")
        #     buster = PoseBusters(config="mol")
        #     bust_result = buster.bust(mol_pred=ligand_files, mol_cond=pocket_file)
        #     filter_mask = bust_result[prior_filters].all(axis=1).values

        # filtered_ligand_files = np.array(ligand_files)[filter_mask]

        filtered_ligand_files = sorted(pocket_dir.glob("*_ligand.sdf"))



        # run the dpocket evaluation of all valid ligand/pocket complexes
        dpocket_lines = []
        dpocket_wd = dpocket_complex_dir / pocket_id # parallel running dpocket should run in seperate folders
        dpocket_wd.mkdir(exist_ok = True)

        # fetch pyKV descriptors
        first_pocket_pdb = pocket_dir / "0_pocket.pdb" # assuming all pockets in the directory are the same
        pyKV_results = run_pyKVFinder(str(first_pocket_pdb), pykv_args)


        # Evaluate all sample ligands
        for lig_file in filtered_ligand_files:
            # print(lig_file)
            m = re.match(r"(\d+)_ligand\.sdf", lig_file.name)
            if not m:
                print(f"[WARN] Skipped {lig_file}: lacking regex match")
                continue
            sample_idx = int(m.group(1))
            pocket_pdb = pocket_dir / f"{sample_idx}_pocket.pdb"

            # run dpocket

            dpocket_pdb_complex_path = build_dpocket_ready_complex(pocket_pdb, lig_file, out_root = dpocket_wd)
            # extract ligand name
            parts = pocket_id.split("-")
            lig_resname =  parts[4].upper() if len(parts) >= 5 else "LIG"
            # Compute the path relative to output_root (subprocess cwd)
            complex_pdb_rel = make_relative_path(dpocket_pdb_complex_path, dpocket_wd)
            dpocket_lines.append(f"{complex_pdb_rel}\t{lig_resname}")


            # get per atom steric clashes
            ligand = Chem.SDMolSupplier(str(lig_file), sanitize=False)[0] # TODO: is this loading only a small fragment?
            pocket_mol = Chem.MolFromPDBFile(str(pocket_pdb), sanitize=False, removeHs=False)

            if ligand is None or pocket_mol is None:
                print(f"[WARN] Skipped {lig_file}: failed ligand or pocket loading")
                continue

            result = check_intermolecular_distance(mol_pred=ligand, mol_cond=pocket_mol)
            details = result["details"]
            clashes = details.copy() # details[details["clash"]].copy()

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

            if pyKV_results is not None:
                for k, v in pyKV_results.items():
                    clashes[k] = v
            
            lig_rows.append(clashes)
        
        lig_df = pd.concat(lig_rows, axis=0, ignore_index=True)
        cols_to_fill = [
            c for c in lig_df.columns
            if c.startswith("RESIDUES") or c.startswith("CLASS")
        ]
        lig_df[cols_to_fill] = lig_df[cols_to_fill].fillna(0) # for residues that where NANs

        dpocket_input_file = dpocket_wd / f"{pocket_id}_dpocket_input.txt"

        with open(dpocket_input_file, "w") as f:
            f.write("\n".join(dpocket_lines))

        run_dpocket(dpocket_input_file, dpocket_wd, dpocket_detection_radius=dpocket_detection_radius)

        # merge the descriptors resulting from the explicit volumen calculated by dpocket
        dpocket_explicit_data = pd.read_csv(f"{dpocket_wd}/dpout_explicitp.txt", sep="\s+")
        dpocket_explicit_data.rename(columns={old_name: f"fpocket_explicit_{old_name}" for old_name in dpocket_explicit_data.columns}, inplace=True)
        dpocket_explicit_data["sample_id"] = dpocket_explicit_data["fpocket_explicit_pdb"].apply(lambda x: x.split("_complex")[0])

        lig_rows = lig_df.merge(dpocket_explicit_data, how="left", on="sample_id")

        # merge the descriptors resulting from the top fpocket detected cavity
        fpocket_data = pd.read_csv(f"{dpocket_wd}/dpout_fpocketp.txt", sep="\s+")
        fpocket_data.rename(columns={old_name: f"fpocket_cavity_{old_name}" for old_name in fpocket_data.columns}, inplace=True)
        fpocket_data["sample_id"] = fpocket_data["fpocket_cavity_pdb"].apply(lambda x: x.split("_complex")[0])

        lig_rows = lig_rows.merge(fpocket_data, how="left", on="sample_id")

        # if available, merge metrics information from drugflow pipeline
        if metrics_file.exists():
            print("[INFO] Merging metrics data")
            metrics_data = pd.read_csv(metrics_file)
            metrics_data["sample_id"] = metrics_data["sdf_file"].apply(lambda x: Path(x).parent.name) + "_" + metrics_data["sample"].astype(str)
            print(metrics_data["sample_id"][0])
            # print(str(metrics_data["sample"]))
            lig_rows = lig_rows.merge(metrics_data, how="left", on="sample_id")


        # print(type(lig_rows))
        
        return lig_rows

def get_allowed_cpus(requested=None):
    # CPUs granted by SLURM
    total_cpus = os.cpu_count()
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", total_cpus))

    if slurm_cpus is None:
        raise RuntimeError("No CPUs available")

    # Allow user to request fewer
    if requested is None:
        return slurm_cpus
    
    if requested > slurm_cpus:
        print(f"[INFO] Number of parallel processes reduced to {slurm_cpus}")

    return min(requested, slurm_cpus)

def full_pipeline(
    ours_samples_root: Path,
    # crossdocked_root: Path,
    # output_root: Path, 
    dpocket_detection_radius: int = 4,
    pykv_args: Dict = {}, 
    overwrite_existing: bool = False,
    requested_workers: int|None = None # None uses max
):
    output_root = ours_samples_root.parent / (ours_samples_root.name.split("_")[0] + "_metrics") / "detailed_clash_evaluation"
    if overwrite_existing:
        try:
            shutil.rmtree(output_root)
            print(f"[INFO] Deleted existing {output_root}")
        except FileNotFoundError:
            print(f"{output_root} doesn't exist and will therefore not be deleted. Creating directory...")
    
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    pdb_paths = sorted(ours_samples_root.iterdir())
    available_workers = get_allowed_cpus(requested_workers)
        

    with ProcessPoolExecutor(max_workers=available_workers) as pool:

        futures = [pool.submit(evaluate_class_worker, p, output_root,  dpocket_detection_radius) for p in pdb_paths]

        for i, fut in tqdm(enumerate(as_completed(futures)), total=len(futures)):

            try:
                clash_result = fut.result()
            except Exception as e:
                print(f"[ERROR] processing path {pdb_paths[i]}:", e)
                traceback.print_exc()
            else:
                all_rows.append(clash_result)

    if not all_rows:
        return pd.DataFrame()

    print(type(all_rows[0]))

    final_df = pd.concat(all_rows, axis=0, ignore_index=True)

    final_df.to_csv(output_root.parent / "detailed_clash_evaluation.csv", index=False)
    print(f"[OK] Final dataframe saved in {output_root}.")

    return final_df


# ============================================================
# Core analysis helpers (largely unchanged)
# ============================================================

def calc_means_classes(df, classes):
    df = df.copy()
    df["clash"] = df["clash"].astype(bool)

    return (
        df
        .groupby(classes)
        .agg(
            n_pairs=("clash", "size"),
            n_clashes=("clash", "sum"),
            clash_rate=("clash", "mean"),
            mean_dist=("distance", "mean"),
            mean_rel_dist=("relative_distance", "mean"),
            n_prot_atoms=("protein_atom_id", "nunique"),
            n_lig_atoms=("ligand_atom_id", "nunique"),
        )
        .sort_values("clash_rate", ascending=False)
    )


def clash_matrix(
    df,
    value="clash",
    aggfunc="mean",
    protein_axis="protein_element",
):
    return (
        df.pivot_table(
            index="ligand_element",
            columns=protein_axis,
            values=value,
            aggfunc=aggfunc,
            fill_value=0.0,
        )
    )


def chi_square_clash_test(df, variable):
    contingency = pd.crosstab(df[variable], df["clash"])
    chi2, p, _, _ = chi2_contingency(contingency)
    return chi2, p


def kruskal_wallis_and_posthoc_dunn(df, cat_var, num_var):
    data = df[[cat_var, num_var]].dropna()

    groups = [g[num_var].values for _, g in data.groupby(cat_var)]
    kw = kruskal(*groups)

    dunn = posthoc_dunn(
        data,
        val_col=num_var,
        group_col=cat_var,
        p_adjust="holm",
    )

    return kw, dunn

def analyze_and_plot_clashes(
    final_df: pd.DataFrame,
    save_path: Path,
):
    save_path.mkdir(parents=True, exist_ok=True)
    out_file = save_path / "detailed_clash_evaluation_analysis.png"

    # =========================================================
    # Filters (exactly as in draft)
    # =========================================================
    prior_posebusters_filters = [
        f"posebusters.{f}"
        for f in [
            "sanitization",
            "all_atoms_connected",
            "aromatic_ring_flatness",
            "bond_angles",
            "bond_lengths",
            "double_bond_flatness",
            "internal_steric_clash",
        ]
    ]

    prior_medchem_filters = [f"medchem.{f}" for f in ["valid", "connected"]]

    clash_df_before_filters = final_df.copy()

    medchem_mask = clash_df_before_filters[prior_medchem_filters].all(axis=1)
    posebusters_mask = clash_df_before_filters[prior_posebusters_filters].all(axis=1)

    clash_df_after_medchem = clash_df_before_filters[medchem_mask]
    clash_df_after_posebusters = clash_df_before_filters[posebusters_mask]
    clash_df_filtered = clash_df_before_filters[medchem_mask & posebusters_mask]

    dataframes_by_filterlevel = {
        "none": clash_df_before_filters,
        "medchem": clash_df_after_medchem,
        "posebusters": clash_df_after_posebusters,
        "both": clash_df_filtered,
    }

    # =========================================================
    # Mask agreement (EXACTLY as requested)
    # =========================================================
    total_mask_agreement = (medchem_mask & posebusters_mask).mean()
    connected_agreement = (
        clash_df_before_filters["posebusters.all_atoms_connected"]
        & clash_df_before_filters["medchem.connected"]
    ).mean()

    # =========================================================
    # Wasserstein distances (clash rate + relative distance)
    # =========================================================
    def per_sample(series, df):
        return df.groupby("sample_id")[series].mean().values

    wass_rows = []
    for (k1, df1), (k2, df2) in combinations(dataframes_by_filterlevel.items(), 2):
        wass_rows.append(
            (k1, k2,
             wasserstein_distance(per_sample("clash", df1), per_sample("clash", df2)),
             wasserstein_distance(per_sample("relative_distance", df1),
                                  per_sample("relative_distance", df2)))
        )

    wasserstein_df = pd.DataFrame(
        wass_rows,
        columns=["filter_a", "filter_b", "W_clash_rate", "W_relative_dist"],
    )

    # =========================================================
    # Protein-axis comparisons
    # =========================================================
    prot_cols = ["protein_element", "protein_atom_name", "protein_resname"]

    # =========================================================
    # Correlation setup (EXACT variables)
    # =========================================================
    numerical_dpocket_descriptors = [
        "pock_vol", "mean_as_solv_acc", "apol_as_prop", "mean_loc_hyd_dens",
        "hydrophobicity_score", "volume_score", "polarity_score",
        "charge_score", "flex", "prop_polar_atm", "as_density",
        "as_max_dst", "convex_hull_volume",
        "surf_pol_vdw14", "surf_pol_vdw22",
        "surf_apol_vdw14", "surf_apol_vdw22",
        "n_abpa",
    ]
    ligand_eval_descriptors  = [
         "medchem.size", "gnina.minimisation_rmsd"
    ]

    per_sample_aggregation = (
        clash_df_filtered
        .groupby("sample_id")
        .agg(
            n_clashes=("clash", "sum"),
            clash_rate=("clash", "mean"),
            relative_distance=("relative_distance", "mean"),
            protein_dist_to_ca=("protein_dist_to_ca", "mean"),
            lig_vol=("lig_vol", "first"),
            **{k: (k, "first") for k in numerical_dpocket_descriptors+ligand_eval_descriptors}
        )
    )

    # =========================================================
    # Figure layout
    # =========================================================
    fig = plt.figure(figsize=(28, 22))
    gs = fig.add_gridspec(5, 3, hspace=0.45, wspace=0.3)

    # ---------------------------------------------------------
    # Row 1: Filter-level histograms
    # ---------------------------------------------------------
    ax_hist = fig.add_subplot(gs[0, :2])
    for name, df in dataframes_by_filterlevel.items():
        ax_hist.hist(
            df.groupby("sample_id")["clash"].mean(),
            bins=40,
            alpha=0.5,
            label=name,
        )
    ax_hist.set_title("Per-sample clash-rate distributions")
    ax_hist.legend()

    # ---------------------------------------------------------
    # Wasserstein table + mask agreement
    # ---------------------------------------------------------
    ax_wass = fig.add_subplot(gs[0, 2])
    ax_wass.axis("off")

    table = ax_wass.table(
        cellText=[
            [a, b, f"{w1:.4f}", f"{w2:.4f}"]
            for a, b, w1, w2 in wasserstein_df.itertuples(index=False)
        ],
        colLabels=wasserstein_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    ax_wass.set_title(
        "Wasserstein distances\n"
        f"Mask agreement: {total_mask_agreement:.2%}\n"
        f"Connected agreement: {connected_agreement:.2%}"
    )

    # ---------------------------------------------------------
    # Rows 2–3: protein-axis heatmaps + tables
    # ---------------------------------------------------------
    clash_df_before_filters["clash_binary"] = clash_df_before_filters["clash"].apply(lambda x: 1 if x else 0)
    for i, prot_col in enumerate(prot_cols):
        ax_hm = fig.add_subplot(gs[1, i])
        mat = clash_matrix(
            clash_df_filtered,
            value="clash",
            aggfunc="mean",
            protein_axis=prot_col,
        )
        sns.heatmap(mat, cmap="viridis", ax=ax_hm)
        ax_hm.set_title(f"P(clash | ligand × {prot_col})")

        ax_tab = fig.add_subplot(gs[2, i])
        ax_tab.axis("off")
        stats = calc_means_classes(clash_df_filtered, [prot_col]).head(10)
        table = ax_tab.table(
            cellText=np.round(stats.values, 3),
            rowLabels=stats.index,
            colLabels=stats.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)
        ax_tab.set_title(f"Top {prot_col} by clash rate")

    # ---------------------------------------------------------
    # Rows 4–5: Targeted Pearson & Kendall correlations
    # ---------------------------------------------------------
    x_vars = numerical_dpocket_descriptors + ligand_eval_descriptors + ["protein_dist_to_ca"]

    for j, method in enumerate(["pearson", "kendall"]):

        # ---- Row 4: clash rate vs descriptors ----
        ax_clash = fig.add_subplot(gs[3, j])

        clash_corr = pd.DataFrame(
            {
                "clash_rate": per_sample_aggregation[x_vars]
                    .corrwith(per_sample_aggregation["clash_rate"], method=method),
                "relative_distance": per_sample_aggregation[x_vars]
                    .corrwith(per_sample_aggregation["relative_distance"], method=method),
            }
        )

        sns.heatmap(
            clash_corr,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            cbar=True,
            ax=ax_clash,
        )

        ax_clash.set_title(f"{method.capitalize()} – per sample clash rate and rel dist")
        ax_clash.set_ylabel("Descriptor")
        ax_clash.set_xlabel("per_ligand_clash_rate")
        ax_clash.set_yticks(np.arange(len(clash_corr.index)) + 0.5)
        ax_clash.set_yticklabels(clash_corr.index, rotation=0)


        # ---- Row 5: relative distance vs descriptors ----
        ax_dist = fig.add_subplot(gs[4, j])

        rel_corr = pd.DataFrame(
            {
                "clash_binary": clash_df_before_filters[x_vars]
                    .corrwith(clash_df_before_filters["clash_binary"], method=method),
                "relative_distance": clash_df_before_filters[x_vars]
                    .corrwith(clash_df_before_filters["relative_distance"], method=method),
            }
        )

        sns.heatmap(
            rel_corr,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            cbar=True,
            ax=ax_dist,
        )

        ax_dist.set_title(f"{method.capitalize()} – per atom clash and rel dist")
        ax_dist.set_ylabel("Descriptor")
        ax_dist.set_xlabel("relative_distance")
        ax_dist.set_yticks(np.arange(len(rel_corr.index)) + 0.5)
        ax_dist.set_yticklabels(rel_corr.index, rotation=0)
        


    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved full analysis figure → {out_file}")


# ============================================================
# __main__ orchestration (exactly one call)
# ============================================================

if __name__ == "__main__":
    # run data extraction
    kv_args = {"step":0.6, "probe_in":0.4, "probe_out":4.0, "removal_distance":2.4}
    input_folder = Path("../benchmarks/ours/ours_samples/")
    final_df = full_pipeline(input_folder, dpocket_detection_radius = 10, pykv_args = kv_args, overwrite_existing=True, requested_workers=None)
    print(final_df[["pyKV_volume", "fpocket_cavity_pock_vol", "fpocket_explicit_pock_vol"]].corr(method="kendall"))
    # run analysis
    # save_path = Path("../benchmarks/ours/ours_metrics/")
    # final_df = pd.read_csv(save_path / "detailed_clash_evaluation.csv")

    # analyze_and_plot_clashes(
    #     final_df=final_df,
    #     save_path=save_path,
    # )









