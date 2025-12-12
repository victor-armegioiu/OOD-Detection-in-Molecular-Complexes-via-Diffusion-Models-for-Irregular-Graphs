from pathlib import Path
import subprocess
import re
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
from rdkit import Chem
import gemmi

from posebusters.modules.intermolecular_distance import check_intermolecular_distance
from posebusters import PoseBusters

from tqdm import tqdm
from typing import List, Dict

from scipy.stats import chisquare, kruskal, f_oneway
# https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.stats.chisquare.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
from statsmodels.stats.contingency_tables import cochrans_q
# https://www.statsmodels.org/dev/generated/statsmodels.stats.contingency_tables.cochrans_q.html
from scikit_posthocs import posthoc_dunn
# https://scikit-posthocs.readthedocs.io/en/latest/generated/scikit_posthocs.posthoc_dunn.html


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
    # if out_root is None:
    #     out_root = crossdocked_root

    # rec_pdb = crossdocked_root / f"{pocket_id}.pdb"
    # lig_sdf = crossdocked_root / f"{pocket_id}.sdf"

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

# def build_all_dpocket_complexes(
#     pocket_root: Path,
#     output_root: Path,
#     dpocket_input_path: Path
# ):
#     output_root.mkdir(parents=True, exist_ok=True)
#     dpocket_lines = []

#     for pdb_file in sorted(pocket_root.glob("*.pdb")):
#         pocket_id = pdb_file.stem
#         lig_sdf = pocket_root / f"{pocket_id}.sdf"
#         if not lig_sdf.exists():
#             print(f"[WARN] No ligand SDF for {pocket_id}")
#             continue

#         # Correct: CALL WITH out_root
#         complex_pdb = output_root / f"{pocket_id}_complex_for_dpocket.pdb"

#         complex_pdb = build_dpocket_ready_complex(
#             pocket_root, pocket_id, out_root=output_root
#         )
#         # relative update for subprocess logic in dpocket later on
#         complex_pdb_rel = make_relative_path(complex_pdb, output_root)
    

#         # extract ligand name
#         parts = pocket_id.split("-")
#         lig_resname = parts[4].upper() if len(parts) >= 5 else "LIG"

#         dpocket_lines.append(f"{complex_pdb_rel}\t{lig_resname}")

#     with open(dpocket_input_path, "w") as f:
#         f.write("\n".join(dpocket_lines))

#     print(f"[OK] Wrote dpocket master input: {dpocket_input_path}")



def run_dpocket(dpocket_input_file: Path, dpocket_processing_dir: Path, prefix: str, dpocket_detection_radius:int = 4):
    """
    Run dpocket only once and ensure the dpocket_input_path
    is correctly referenced relative to output_root.
    """
    explicit_out = dpocket_processing_dir / f"{prefix}_exp.txt"

    # print(dpocket_input_path)
    # print(output_root)
    # print(explicit_out)

    # # Skip if dpocket already ran
    if explicit_out.exists():
        print("[OK] dpocket results already exist, skipping.")
        return

    # Compute the path relative to output_root (subprocess cwd)
    dpocket_input_rel = make_relative_path(dpocket_input_file, dpocket_processing_dir)
    
    # print(dpocket_input_rel)

    print(f"[INFO] Running dpocket with input: {dpocket_input_rel}")

    subprocess.run(
        ["dpocket", "-f", str(dpocket_input_rel), "-o", prefix, "-d", str(dpocket_detection_radius)], # TODO -d pocket distance
        cwd=str(dpocket_processing_dir),
        check=True
    )

    print(f"[OK] dpocket executed successfully for prefix {prefix}.")



# def load_all_dpocket_descriptors(explicit_path: Path) -> pd.DataFrame:
#     """
#     Load dpocket explicit pocket descriptors for ALL pocket_ids.
#     dpocket writes one row per input pdb, in the same order.
#     """
#     # print(os.getcwd())
#     df = pd.read_csv(explicit_path, sep="\s+", engine="python")

#     # Extract pocket_id from the pdb path column
#     df["pdb"] = df["pdb"].astype(str)
#     df["pocket_id"] = df["pdb"].apply(lambda x: Path(x).stem.replace("_complex_for_dpocket", ""))

#     return df.set_index("pocket_id", drop=False)

def evaluate_class_worker(pocket_dir: Path, dpocket_complex_dir: Path, dpocket_detection_radius:int ) -> pd.DataFrame:
            

        if not pocket_dir.is_dir():
            print(f"[WARNING] returning None for {pocket_dir} with working directory {os.getcwd()}")
            return 

        pocket_id = pocket_dir.name

        # # Load dpocket row (one per pocket_id)
        # if pocket_id not in dpocket_df.index:
        #     print(f"[WARN] No dpocket descriptors for {pocket_id}")
        #     dp_desc = None
        # else:
        #     dp_desc = dpocket_df.loc[pocket_id].to_dict()

        dp_desc = ...

        lig_rows = []

        # filtering ligand files that pass all filters prior to the clash filter
        ligand_files = sorted(pocket_dir.glob("*_ligand.sdf"))
        pocket_file = sorted(pocket_dir.glob("*_pocket.pdb"))[0]
        metrics_file = Path(pocket_dir.parents[1]) / (pocket_dir.parent.name.split("_")[0] + "_metrics") / "metrics_detailed.csv"
        prior_filters = ["sanitization", "all_atoms_connected", "aromatic_ring_flatness", "bond_angles", "bond_lengths", "double_bond_flatness", "internal_steric_clash"]

        if metrics_file.exists():
            bust_result = pd.read_csv(metrics_file)
            rel_tail = ["/".join(p.parts[-2:]) for p in ligand_files]
            abs_tail = bust_result["sdf_file"].apply(lambda p: "/".join(Path(p).parts[-2: ]))
            mask = abs_tail.isin(rel_tail)
            # order absolute paths *in the order they appear in list_rel*
            order_idx = abs_tail[mask].map(
                {k: i for i, k in enumerate(rel_tail)}
            ).sort_values().index

            # ordered result
            updated_prior_filters = [f"posebusters.{s}" for s in prior_filters]
            filter_mask = bust_result.loc[order_idx, updated_prior_filters].all(axis=1).values
            
        else:
            print("[INFO] No metrics file found, running PoseBusters")
            buster = PoseBusters(config="mol")
            bust_result = buster.bust(mol_pred=ligand_files, mol_cond=pocket_file)
            filter_mask = bust_result[prior_filters].all(axis=1).values

        filtered_ligand_files = np.array(ligand_files)[filter_mask]



        # run the dpocket evaluation of all valid ligand/pocket complexes
        dpocket_lines = []

        

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
            dpocket_pdb_complex_path = build_dpocket_ready_complex(pocket_pdb, lig_file, out_root = dpocket_complex_dir)
            # extract ligand name
            parts = pocket_id.split("-")
            lig_resname =  parts[4].upper() if len(parts) >= 5 else "LIG"
            # Compute the path relative to output_root (subprocess cwd)
            complex_pdb_rel = make_relative_path(dpocket_pdb_complex_path, dpocket_complex_dir)
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
            # if clashes.empty:
            #     print(f"[INFO] No clashes detected in {lig_file}")
            #     continue
            # else:
            #     print(f"[INFO] Clash detected in {lig_file}")

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
            
            lig_rows.append(clashes)
        
        dpocket_input_file = dpocket_complex_dir / f"{pocket_id}_dpocket_input.txt"

        with open(dpocket_input_file, "w") as f:
            f.write("\n".join(dpocket_lines))

        run_dpocket(dpocket_input_file, dpocket_complex_dir, prefix=pocket_id, dpocket_detection_radius=dpocket_detection_radius)

        dpocket_descriptor_data = pd.read_csv(f"{dpocket_complex_dir}/{pocket_id}_exp.txt", sep="\s+")
        dpocket_descriptor_data["sample_id"] = dpocket_descriptor_data["pdb"].apply(lambda x: x.split("_complex")[0])

        lig_df = pd.concat(lig_rows, axis=0, ignore_index=True)
        lig_rows = lig_df.merge(dpocket_descriptor_data, how="left", on="sample_id")

        # if available, merge metrics information from drugflow pipeline
        if metrics_file.exists():
            metrics_data = pd.read_csv(metrics_file)
            metrics_data["sample_id"] = metrics_data["sdf_file"].apply(lambda x: Path(x).parent.name) + "_" + metrics_data["sample"]
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
    overwrite_existing: bool = False,
    requested_workers: int|None = None # None uses max
):
    output_root = ours_samples_root.parent / (ours_samples_root.name.split("_")[0] + "_metrics") / "detailed_clash_evaluation"
    if overwrite_existing:
        os.rmdir(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # # ---- Stage 1: build complexes + master input ----
    # dpocket_input = output_root / "dpocket_input_all.txt"

    # build_all_dpocket_complexes(
    #     crossdocked_root=crossdocked_root,
    #     output_root=output_root,
    #     dpocket_input_path=dpocket_input
    # )

    # # ---- Stage 2: run dpocket once ----
    # run_dpocket_once(
    #     dpocket_input_path=dpocket_input,
    #     output_root=output_root,
    #     prefix="dpout_all"
    # )

    # explicit_path = output_root / "dpout_all_exp.txt"

    # # ---- Stage 3: load dpocket descriptor table ----
    # dpocket_df = load_all_dpocket_descriptors(explicit_path)

    # ---- Stage 4: evaluate all clashes in parellized ----
    all_rows = []
    pdb_paths = sorted(ours_samples_root.iterdir())
    available_workers = get_allowed_cpus(requested_workers)
        

    with ProcessPoolExecutor(max_workers=available_workers) as pool:

        futures = [pool.submit(evaluate_class_worker, p, output_root, dpocket_detection_radius) for p in pdb_paths]

        for i, fut in tqdm(enumerate(as_completed(futures)), total=len(futures)):

            try:
                clash_result = fut.result()
            except Exception as e:
                print(f"[ERROR] processing path {pdb_paths[i]}:", e)
            else:
                all_rows.append(clash_result)

    if not all_rows:
        return pd.DataFrame()

    print(type(all_rows[0]))

    final_df = pd.concat(all_rows, axis=0, ignore_index=True)

    final_df.to_csv(output_root.parent / "detailed_clash_evaluation.csv", index=False)
    print(f"[OK] Final dataframe saved in {output_root}.")

    return final_df



if __name__ == "__main__":
    # TODO increase dpocket radius so we're sure it detects entire pocket
    input_folder = Path("../benchmarks/ours/ours_samples/")
    # crossdocked_root = Path("../benchmarks/processed_crossdocked/test/")
    # ourt_root = Path("../benchmarks/ours/ours_samples/")

    # final_df = full_pipeline(input_folder, dpocket_detection_radius = 10, overwrite_existing=True, requested_workers=None) #, crossdocked_root, out_root)
    # final_df.to_csv("clash_dpocket_annotated.csv", index=False)
    # print(final_df.head())

    # evaluate_class_worker(pocket_dir=input_folder/"14gs-A-rec-20gs-cbd-lig-tt-min", dpocket_complex_dir = input_folder.parent / (input_folder.name.split("_")[0] + "_metrics") / "detailed_clash_evaluation")

    # develop the pipeline further by loading final df
    final_df = pd.read_csv("../benchmarks/ours/ours_metrics/detailed_clash_evaluation.csv")
    print(final_df.groupby("protein_atom_name").agg(clashes = ("clash", "mean"), count = ("clash", "size"), prot_vdw = ("protein_vdw", "first")).sort_values(["clashes"], ascending=False))

    sys.exit(0)

    # test whether final_df has invalid information

    ## play ground ##


    # print(len(ligand_files), len(filtered_ligand_files))

    #######################################

    # work on a copy if you want to be safe
    df = final_df.copy()

    def calc_means_classes(df, classes = ["protein_resname"]):
        # ensure clash is boolean (it *should* already be)
        df["clash"] = df["clash"].astype(bool)

        # per-residue-class / resname stats
        residue_stats = (
            df
            .groupby(classes)
            .agg(
                n_pairs      = ("clash", "size"),   # number of ligand–protein pairs
                n_clashes    = ("clash", "sum"),    # number of pairs flagged as clash
                clash_rate   = ("clash", "mean"),   # fraction of clashing pairs
                mean_dist    = ("distance", "mean"),
                mean_rel_dist= ("relative_distance", "mean"),
                n_prot_atoms = ("protein_atom_id", "nunique"),
                n_lig_atoms  = ("ligand_atom_id", "nunique"),
            )
            .sort_values(["clash_rate"], ascending=False)
            # .sort_values(["protein_res_class", "clash_rate"], ascending=[True, False])
        )

        print(residue_stats.head(20))

    def clash_matrix(
        df: pd.DataFrame,
        value: str = "clash",              # column to aggregate over
        aggfunc = "mean",                  # e.g. "mean", "sum", "count"
        protein_axis: str = "protein_element",  # or "protein_atom_name"
        normalize: bool = False            # True -> row-normalize
    ) -> pd.DataFrame:
        """
        Build a ligand-type x protein-type matrix for clashes or any metric.
        """
        mat = df.pivot_table(
            index="ligand_element",
            columns=protein_axis,
            values=value,
            aggfunc=aggfunc,
            fill_value=0.0,
        )

        if normalize:
            # e.g. convert counts/sums into row-wise fractions
            mat = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0)

        return mat
    
        # examples:

    # 1) probability of clash for each (lig_elem, prot_elem) pair
    clash_prob_elem = clash_matrix(final_df, value="clash", aggfunc="mean",
                                protein_axis="protein_element")

    # 2) total number of clashes for each (lig_elem, atom_name) pair
    n_clashes_name = clash_matrix(
        final_df[final_df["clash"]],       # restrict to actual clashes if you want
        value="clash",
        aggfunc="sum",
        protein_axis="protein_atom_name"
    )

    print(clash_prob_elem)
    print(n_clashes_name)

    def do_chi_and_cochran(df, variable = "protein_resname"):
        ## chi squared test ##
        overall_clash_p = df["clash"].mean()
        frequencies = (
            df
            .groupby([variable])
            .agg(
                size = ("clash", "size"),
                obs = ("clash", "sum") 
            )
        )
        frequencies["exp"] = frequencies["size"] * overall_clash_p # TODO is this correct expected?
        test = chisquare(f_obs=frequencies["obs"].values, f_exp=frequencies["exp"].values) # TODO do I need to verify chi squ assumptions?


        print("Chi square results:", test)

        ## cochrans q test ##
        frequencies["none"] = frequencies["size"] - frequencies["obs"]

        x = frequencies[["obs", "none"]].values.T
        test2 = cochrans_q(x) # TODO is it okay that the classes are unevenly distirbuted?
        print("Conchrans Q results", test2)

    ## anova and kruskal wallis ##
    def kruskal_wallis_and_posthoc_dunn(cat_var = "protein_resname", num_var = "relative_distance"):

        # ANOVA assumes
        # The samples are independent.
        # Each sample is from a normally distributed population.
        # The population standard deviations of the groups are all equal. This property is known as homoscedasticity.
        # -> this is props not valid so we do KW

        samples = [df[df[cat_var] == name][num_var].values for name in df[cat_var].unique()]
        # test = f_oneway(*samples, equal_var = False)
        test2 = kruskal(*samples)
        print(test2)
        # follow up KW with posthoc dunn

        test = posthoc_dunn(samples)
        print(test)

    ########### correlation ##########################






    # aggregate per sample
    per_sample_aggregation = (
        df
        .groupby("sample_id")
        .agg(
            n_pairs    = ("clash", "size"),
            n_clashes  = ("clash", "sum"),
            per_ligand_clash_rate = ("clash", "mean"),
            lig_vol    = ("dpocket_lig_vol", "first"),
            pock_vol   = ("dpocket_pock_vol", "first"),
            # solvent_accessibility = ("mean_as_solv_acc", "first")

        )
    )

    # sanity check
    # print(per_sample.head())

    num_cols =  [f"dpocket_{s}" for s in ["pock_vol", "mean_as_solv_acc", "apol_as_prop", "mean_loc_hyd_dens", "hydrophobicity_score", "volume_score", "polarity_score", "charge_score", "flex", "prop_polar_atm", "as_density", "as_max_dst", "convex_hull_volume", "surf_pol_vdw14",	"surf_pol_vdw22",	"surf_apol_vdw14",	"surf_apol_vdw22",	"n_abpa"]]

    # Pearson correlations
    pearson_corr = per_sample_aggregation[["per_ligand_clash_rate", "lig_vol", "pock_vol"]].corr(method="pearson")
    print("Pearson correlation:")
    print(pearson_corr)

    # Spearman (rank-based, in case things are non-linear / heavy-tailed)
    kendall_corr = per_sample_aggregation[["per_ligand_clash_rate", "lig_vol", "pock_vol"]].corr(method="kendall")
    print("Kendall correlation:")
    print(kendall_corr)

    print(df[["relative_distance"] + num_cols].corr(method="kendall"))

    # ########### learning to predict clashes ####################

    # # prepare df -> explanatory vars
    # # categorical_columns = ["ligand_element", "protein_element", "protein_atom_name", "protein_resname", "protein_region", "protein_res_class", ]
    # # numerical_columns =  [f"dpocket_{s}" for s in ["pock_vol", "mean_as_solv_acc", "apol_as_prop", "mean_loc_hyd_dens", "hydrophobicity_score", "volume_score", "polarity_score", "charge_score", "flex", "prop_polar_atm", "as_density", "as_max_dst", "convex_hull_volume", "surf_pol_vdw14",	"surf_pol_vdw22",	"surf_apol_vdw14",	"surf_apol_vdw22",	"n_abpa"]]
    # # a subset for faster compute
    # categorical_columns = ["ligand_element", "protein_element", "protein_resname", "protein_region", "protein_res_class"]
    # numerical_columns =  [f"dpocket_{s}" for s in ["pock_vol", "mean_as_solv_acc"]]
    # # vars to be predicted
    # y_binary = "clash" # this is a bool
    # y_continuous = "relative_distance"

    # from sklearn.compose import ColumnTransformer
    # from sklearn.pipeline import Pipeline
    # from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
    # from sklearn.linear_model import LogisticRegression, Lasso
    # from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    # import pandas as pd
    # import numpy as np

    # # Preprocessor
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ("cat", OneHotEncoder(), categorical_columns), # drop possible for logreg
    #         ("num", MinMaxScaler(), numerical_columns),
    #     ],
    #     remainder="drop"
    # )

    # # # ---- BINARY TARGET ----
    # print("********** BINARY TARGET ****************")
    # X = df[categorical_columns + numerical_columns]
    # y_bin = df[y_binary].astype(int)
    # # print("P1")

    # log_reg = Pipeline([
    #     ("prep", preprocessor),
    #     ("clf", LogisticRegression(penalty="l1", solver="saga", max_iter=5000, n_jobs=-1))
    # ])
    # # print("P2")
    # rf_clf = Pipeline([
    #     ("prep", preprocessor),
    #     ("clf", RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1))
    # ])
    # print("Start fitting binary LogReg")
    # log_reg.fit(X, y_bin)
    # print("Start fitting CLF RF")
    # rf_clf.fit(X, y_bin)

    # # Feature names after preprocessing
    # feat_names = (
    #     log_reg.named_steps["prep"]
    #     .get_feature_names_out()
    # )

    # # Coefficients (L1 logistic regression)
    # logreg_coefs = pd.Series(
    #     log_reg.named_steps["clf"].coef_.ravel(),
    #     index=feat_names
    # )

    # # RF feature importances
    # rf_clf_importances = pd.Series(
    #     rf_clf.named_steps["clf"].feature_importances_,
    #     index=feat_names
    # )

    # print("Random Forrest Feature Importances\n", rf_clf_importances)
    # print("===========================================================")
    # print("LogReg Coefficients\n", logreg_coefs)
    
    # print("+++++++++++++++++++++++++++++++++++++++++++++")
    # print("********** CONTINUOUS TARGET ****************")
    # print("+++++++++++++++++++++++++++++++++++++++++++++")


    # # ---- CONTINUOUS TARGET ----
    # y_cont = df[y_continuous]

    # lasso_reg = Pipeline([
    #     ("prep", preprocessor),
    #     ("reg", Lasso(alpha=0.001, max_iter=5000))
    # ])

    # rf_reg = Pipeline([
    #     ("prep", preprocessor),
    #     ("reg", RandomForestRegressor(n_estimators=300, random_state=0))
    # ])

    # "Started fitting Regression"
    # lasso_reg.fit(X, y_cont)
    # "Started fitting RF Regression"
    # rf_reg.fit(X, y_cont)

    # # Coefficients (L1 regression)
    # lasso_coefs = pd.Series(
    #     lasso_reg.named_steps["reg"].coef_,
    #     index=feat_names
    # )

    # # RF regressor importances
    # rf_reg_importances = pd.Series(
    #     rf_reg.named_steps["reg"].feature_importances_,
    #     index=feat_names
    # )

    # print("Random Forrest Feature Importances\n", rf_reg_importances)
    # print("===========================================================")
    # print("LogReg Coefficients\n", lasso_coefs)


    



