import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def extract_pocket_name(path_str: str) -> str:
    """
    Given path like: some/dir/.../<pocket_name>/3_ligand.sdf
    Return pocket_name (the directory above the ligand file).
    """
    p = Path(path_str)
    return p.parent.name


def compute_fail_count(row, pb_cols):
    """
    Given PoseBusters boolean columns (True=pass, False=fail),
    compute fail_count = number of False entries.
    """
    values = row[pb_cols].values
    # ensure boolean
    values = values.astype(bool)
    passed = np.sum(values)
    total = len(values)
    return total - passed


def plot_posebusters_metric_waterfall(df, pb_cols, savepath, title="PoseBusters Breakdown"):
    """
    PoseBusters-style sequential metric waterfall.
    Each ligand is removed at the first metric it fails.
    If a metric eliminates 0% (loss=0), draw a green bar from 0 to remaining%.
    """

    N = len(df)
    if N == 0:
        return

    survivors = np.ones(N, dtype=bool)
    remaining_vals = [100.0]
    labels = ["All predictions"]

    fig, ax = plt.subplots(figsize=(14, 5))

    for col in pb_cols:
        passes = df[col].astype(bool).values

        # compute failures among current survivors
        fail_mask = survivors & (~passes)
        lost_count = fail_mask.sum()
        loss_pct = (lost_count / N) * 100.0

        prev_rem = remaining_vals[-1]
        curr_rem = max(0.0, prev_rem - loss_pct)
        remaining_vals.append(curr_rem)

        # Pretty label
        nice_label = col.replace("posebusters.", "").replace("posebusters_", "").replace("_", " ")
        labels.append(nice_label)

        # Plot bar
        idx = len(remaining_vals) - 1

        if loss_pct > 0:  
            # normal orange drop
            ax.bar(idx, prev_rem - curr_rem, bottom=curr_rem, color="#d95f02")
            ax.text(idx, prev_rem + 1, f"-{loss_pct:.0f}%", ha="center", va="bottom", fontsize=8)

        else:
            # zero-loss → green bar upward from zero to remaining%
            ax.bar(idx, curr_rem, bottom=0, color="#1b9e77")
            ax.text(idx, curr_rem + 1, f"{curr_rem:.0f}%", ha="center", va="bottom", fontsize=8)

        survivors &= passes

    # Initial annotation (100%)
    ax.text(0, remaining_vals[0] + 1, "100%", ha="center", va="bottom", fontsize=8)

    # Axis formatting
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha='right')
    ax.set_ylabel("Remaining %")
    ax.set_ylim(0, 105)
    ax.set_title(title)

    # Remove frame borders
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()




# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df", type=str, required=True,
                        help="Path to dataframe containing PoseBusters columns")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--global_only", action="store_true",
                        help="Only generate global waterfall plot")
    args = parser.parse_args()

    df_path = Path(args.df)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Load dataframe
    df = pd.read_csv(df_path)

    # Hard-code PoseBusters column order here (must match df columns)
    pb_cols_order = [
        # example – replace with the exact columns you want, in the order you want:
        # "posebusters.mol_cond_loaded",
        "posebusters.mol_pred_loaded",
        "posebusters.sanitization",
        "posebusters.all_atoms_connected",
        "posebusters.aromatic_ring_flatness",
        "posebusters.bond_angles",
        "posebusters.bond_lengths",
        "posebusters.double_bond_flatness",
        "posebusters.inchi_convertible",
        "posebusters.internal_energy",
        "posebusters.internal_steric_clash",
        "posebusters.minimum_distance_to_inorganic_cofactors",
        "posebusters.minimum_distance_to_organic_cofactors",
        "posebusters.minimum_distance_to_protein",
        "posebusters.minimum_distance_to_waters",
        "posebusters.protein-ligand_maximum_distance",
        "posebusters.volume_overlap_with_inorganic_cofactors",
        "posebusters.volume_overlap_with_organic_cofactors",
        "posebusters.volume_overlap_with_protein",
        "posebusters.volume_overlap_with_waters",
        "posebusters.all"
    ]

    # Verify columns exist
    for col in pb_cols_order:
        if col not in df.columns:
            raise ValueError(f"PoseBusters column missing in df: {col}")

    # Required file columns
    if "sdf_file" not in df.columns:
        raise ValueError("Missing column 'sdf_file'")
    if "pdb_file" not in df.columns:
        raise ValueError("Missing column 'pdb_file'")

    # Extract pocket names
    print("Extracting pocket names...")
    df["pocket_name"] = df["sdf_file"].apply(extract_pocket_name)

    # ------------------------------------------------------------------
    # Per-pocket waterfalls (unless --global_only is set)
    # ------------------------------------------------------------------
    if not args.global_only:
        print("Generating per-pocket PoseBusters metric waterfalls...")
        for pocket, group in tqdm(df.groupby("pocket_name")):
            savepath = out_dir / f"{pocket}_posebusters_metric_breakdown.png"
            plot_posebusters_metric_waterfall(
                group,
                pb_cols_order,
                savepath,
                title=f"PoseBusters Breakdown – {pocket}"
            )

    # ------------------------------------------------------------------
    # Global waterfall
    # ------------------------------------------------------------------
    print("Generating global PoseBusters metric waterfall...")
    plot_posebusters_metric_waterfall(
        df,
        pb_cols_order,
        out_dir / "global_posebusters_metric_breakdown.png",
        title="PoseBusters Breakdown – Global"
    )

    print("Done.")

    # call like:
    # python -m sbdd_metrics.posebusters_waterfall_plots --df ../benchmarks/ours/ours_metrics/metrics_detailed.csv --out ../benchmarks/ours/posebusters_waterfall_plots
