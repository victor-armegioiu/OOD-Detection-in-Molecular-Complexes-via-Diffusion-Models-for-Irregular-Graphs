import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from moldiff.constants import bond_decoder, atom_decoder
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from collections import Counter

# Optional: silence RDKit spam (we still track failures explicitly)
RDLogger.DisableLog("rdApp.*")


# ----------------------------
# Core: counts + SAFETY GATE
# ----------------------------
def atom_bond_ring_distributions_from_smiles(
    smiles_list,
    include_hydrogens=False,
    max_fail_rate=0.02,
    min_valid=100,
    source_name="(unknown)",
    strict=True,
):
    """
    Count atoms, bonds, rings over RDKit-parseable SMILES.
    Adds a safety gate: if parse failure rate is too high, raise (strict=True) or warn.

    Returns
    -------
    atom_counts : dict
    bond_counts : dict
    ring_counts : dict
    stats : dict with n_total, n_valid, n_failed, fail_rate
    """
    atom_counts = Counter()
    bond_counts = Counter({"NONE": 0, "SINGLE": 0, "DOUBLE": 0, "TRIPLE": 0, "AROMATIC": 0})
    ring_counts = Counter({"3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9+": 0})

    n_total = 0
    n_valid = 0
    n_failed = 0

    for smi in smiles_list:
        n_total += 1
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_failed += 1
            continue

        n_valid += 1

        if include_hydrogens:
            mol = Chem.AddHs(mol)

        # atoms
        for atom in mol.GetAtoms():
            atom_counts[atom.GetSymbol()] += 1

        # bonds
        for bond in mol.GetBonds():
            if bond.GetIsAromatic():
                bond_counts["AROMATIC"] += 1
            else:
                bt = bond.GetBondType()
                if bt == Chem.BondType.SINGLE:
                    bond_counts["SINGLE"] += 1
                elif bt == Chem.BondType.DOUBLE:
                    bond_counts["DOUBLE"] += 1
                elif bt == Chem.BondType.TRIPLE:
                    bond_counts["TRIPLE"] += 1
                else:
                    bond_counts["NONE"] += 1

        # rings (SSSR)
        for ring_atoms in mol.GetRingInfo().AtomRings():
            r = len(ring_atoms)
            if 3 <= r <= 8:
                ring_counts[str(r)] += 1
            elif r >= 9:
                ring_counts["9+"] += 1

    fail_rate = (n_failed / n_total) if n_total else 0.0
    stats = {
        "source": source_name,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_failed": n_failed,
        "fail_rate": fail_rate,
    }

    # Safety gate
    msg = (
        f"[{source_name}] RDKit SMILES parse failures: {n_failed}/{n_total} = {fail_rate:.2%} "
        f"(valid={n_valid}). Gate: max_fail_rate={max_fail_rate:.2%}, min_valid={min_valid}."
    )
    if n_valid < min_valid:
        if strict:
            raise RuntimeError(msg + " Too few valid molecules; distribution would be noisy/biased.")
        else:
            print("WARNING:", msg + " Proceeding anyway (strict=False).")

    if fail_rate > max_fail_rate:
        if strict:
            raise RuntimeError(msg + " Failure rate too high; distributions likely biased.")
        else:
            print("WARNING:", msg + " Proceeding anyway (strict=False).")

    return dict(atom_counts), dict(bond_counts), dict(ring_counts), stats


# ----------------------------
# Helpers
# ----------------------------
def counts_to_series(counts_dict, categories=None, name=None):
    s = pd.Series(counts_dict, dtype=float)
    if categories is not None:
        s = s.reindex(categories, fill_value=0.0)
    if name is not None:
        s.name = name
    return s


def normalize_series(s: pd.Series) -> pd.Series:
    total = float(s.sum())
    return (s * 0.0) if total <= 0 else (s / total)


def make_long_df(source_to_series):
    rows = []
    for source, s in source_to_series.items():
        s = s.astype(float)
        frac = normalize_series(s)
        for cat in s.index:
            rows.append(
                {"source": source, "category": cat, "count": float(s.loc[cat]), "fraction": float(frac.loc[cat])}
            )
    return pd.DataFrame(rows)


def pivot_for_plot(df_long, metric="fraction"):
    assert metric in ("count", "fraction")
    return df_long.pivot(index="category", columns="source", values=metric).fillna(0.0)


def compare_distributions(ref: pd.Series, test: pd.Series, title="Distribution diff report"):
    ref = ref.astype(float)
    test = test.astype(float)

    idx = ref.index.union(test.index)
    ref = ref.reindex(idx, fill_value=0.0)
    test = test.reindex(idx, fill_value=0.0)

    abs_diff = (test - ref).abs()
    rel_diff = abs_diff / ref.replace(0.0, np.nan)

    report = (
        pd.DataFrame({"ref": ref, "test": test, "abs_diff": abs_diff, "rel_diff": rel_diff})
        .sort_values("abs_diff", ascending=False)
    )

    l1 = float(abs_diff.sum())
    l2 = float(np.sqrt((abs_diff**2).sum()))
    max_abs = float(abs_diff.max())

    print("\n" + "=" * 80)
    print(title)
    print(f"L1(abs)={l1:.6g} | L2(abs)={l2:.6g} | max(abs)={max_abs:.6g}")
    print("-" * 80)
    with pd.option_context("display.max_rows", 50, "display.max_columns", 10, "display.width", 200):
        print(report.head(20))
    print("=" * 80 + "\n")


def grouped_barplot(ax, wide_df, colors, title=None, ylabel=None, label_letter=None):
    categories = list(wide_df.index)
    sources = list(wide_df.columns)

    x = np.arange(len(categories))
    n = len(sources)
    width = 0.8 / max(n, 1)

    for i, src in enumerate(sources):
        y = wide_df[src].values
        ax.bar(
            x + (i - (n - 1) / 2) * width,
            y,
            width=width,
            label=src,
            color=colors.get(src, None),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)

    if label_letter is not None:
        ax.text(
            -0.05, 1.05, label_letter,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=14, fontweight="bold",
        )
        
    plt.tight_layout()


if __name__ == "__main__":
    # ----------------------------
    # Config
    # ----------------------------
    folders_of_interest = ["ours_guidanceBL", "targetdiff", "diffsbdd", "pocket2mol", "drugflow"]

    BOND_CATS = ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
    RING_CATS = ["3", "4", "5", "6", "7", "8", "9+"]

    # Safety thresholds
    MAX_FAIL_RATE = 0.15
    MIN_VALID = 100

    # Cache + output
    out_dir = "../benchmarks/distribution_graphics"
    os.makedirs(out_dir, exist_ok=True)

    atoms_csv = os.path.join(out_dir, "atoms_distribution_long.csv")
    bonds_csv = os.path.join(out_dir, "bonds_distribution_long.csv")
    rings_csv = os.path.join(out_dir, "rings_distribution_long.csv")
    stats_csv = os.path.join(out_dir, "smiles_parse_failure_stats.csv")

    fig_path = os.path.join(out_dir, "atom_bond_ring_distributions.png")

    # ----------------------------
    # 1) Load PDBbind discrete distribution pickle ONLY for TEST (do not plot)
    # ----------------------------
    with open("../PDBbind_train/trainingPDB_discrete_distributions.pkl", "rb") as f:
        PDB_distribution_data = pickle.load(f)

    pdb_atoms_ref = counts_to_series(
        dict(zip(atom_decoder, PDB_distribution_data["atom_types"])),
        categories=atom_decoder,
    )
    pdb_bonds_ref = counts_to_series(
        dict(zip(bond_decoder, PDB_distribution_data["bond_types"])),
        categories=BOND_CATS,
    )

    # ----------------------------
    # 2) Cache gate: if CSVs exist, reuse; else compute + save
    # ----------------------------
    use_cache = all(os.path.exists(p) for p in [atoms_csv, bonds_csv, rings_csv, stats_csv])

    if use_cache:
        print("✓ Found cached distribution CSVs — skipping recomputation.")
        df_atoms_long = pd.read_csv(atoms_csv)
        df_bonds_long = pd.read_csv(bonds_csv)
        df_rings_long = pd.read_csv(rings_csv)
        df_stats = pd.read_csv(stats_csv)
        print(df_stats.sort_values("fail_rate", ascending=False))

    else:
        print("⟳ Cached CSVs not found — computing distributions.")

        atoms_by_source = {}
        bonds_by_source = {}
        rings_by_source = {}
        stats_rows = []

        # ---- Benchmarks ----
        for folder in tqdm(folders_of_interest, desc="Benchmark SMILES → distributions"):
            path = f"../benchmarks/{folder}/{folder}_metrics/metrics_detailed.csv"
            folder_data = pd.read_csv(path)
            smiles_list = folder_data["representation.smiles"].dropna().astype(str).tolist()

            atom_counts, bond_counts, ring_counts, stats = atom_bond_ring_distributions_from_smiles(
                smiles_list,
                include_hydrogens=False,
                max_fail_rate=MAX_FAIL_RATE,
                min_valid=MIN_VALID,
                source_name=folder,
                strict=True,
            )
            stats_rows.append(stats)

            atoms_by_source[folder] = counts_to_series(atom_counts, categories=atom_decoder)
            bonds_by_source[folder] = counts_to_series(bond_counts, categories=BOND_CATS)
            rings_by_source[folder] = counts_to_series(ring_counts, categories=RING_CATS)

        # ---- CrossDocked train ----
        cross_smiles = np.load("../benchmarks/processed_crossdocked/train_smiles.npy", allow_pickle=True)
        cross_smiles = [s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s) for s in cross_smiles]

        atom_counts, bond_counts, ring_counts, stats = atom_bond_ring_distributions_from_smiles(
            cross_smiles,
            include_hydrogens=False,
            max_fail_rate=MAX_FAIL_RATE,
            min_valid=MIN_VALID,
            source_name="crossdocked_train",
            strict=True,
        )
        stats_rows.append(stats)

        atoms_by_source["crossdocked_train"] = counts_to_series(atom_counts, categories=atom_decoder)
        bonds_by_source["crossdocked_train"] = counts_to_series(bond_counts, categories=BOND_CATS)
        rings_by_source["crossdocked_train"] = counts_to_series(ring_counts, categories=RING_CATS)

        # ---- PDBbind train SMILES (PDBbind in BLUE for plots) ----
        pdbbind_smiles = np.load("../PDBbind_train/train_smiles.npy", allow_pickle=True)
        pdbbind_smiles = [s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s) for s in pdbbind_smiles]

        atom_counts, bond_counts, ring_counts, stats = atom_bond_ring_distributions_from_smiles(
            pdbbind_smiles,
            include_hydrogens=False,
            max_fail_rate=MAX_FAIL_RATE,
            min_valid=MIN_VALID,
            source_name="PDBbind_train_smiles",
            strict=True,
        )
        stats_rows.append(stats)

        atoms_by_source["PDBbind_train_smiles"] = counts_to_series(atom_counts, categories=atom_decoder)
        bonds_by_source["PDBbind_train_smiles"] = counts_to_series(bond_counts, categories=BOND_CATS)
        rings_by_source["PDBbind_train_smiles"] = counts_to_series(ring_counts, categories=RING_CATS)

        # ---- Build long DataFrames + cache ----
        df_atoms_long = make_long_df(atoms_by_source)
        df_bonds_long = make_long_df(bonds_by_source)
        df_rings_long = make_long_df(rings_by_source)
        df_stats = pd.DataFrame(stats_rows).sort_values("fail_rate", ascending=False)

        df_atoms_long.to_csv(atoms_csv, index=False)
        df_bonds_long.to_csv(bonds_csv, index=False)
        df_rings_long.to_csv(rings_csv, index=False)
        df_stats.to_csv(stats_csv, index=False)

        print("✓ Distributions computed and cached.")
        print(df_stats)

    # ----------------------------
    # 3) TEST: PDBbind_pickle(ref) vs PDBbind_train_smiles(test)
    #    - Note: the test needs the TEST series; build it from df_*_long if cached.
    # ----------------------------
    # Build wide from long (fractions) and also reconstruct counts (summing over categories) if needed.
    # For the diff test, we compare COUNTS, so reconstruct counts from df_long.
    def counts_series_from_long(df_long, source, categories):
        s = (
            df_long[df_long["source"] == source]
            .set_index("category")["count"]
            .reindex(categories, fill_value=0.0)
            .astype(float)
        )
        return s

    if "PDBbind_train_smiles" not in df_atoms_long["source"].unique():
        raise RuntimeError("Cached CSVs missing PDBbind_train_smiles. Delete cache or recompute.")

    pdbbind_atoms_test = counts_series_from_long(df_atoms_long, "PDBbind_train_smiles", atom_decoder)
    pdbbind_bonds_test = counts_series_from_long(df_bonds_long, "PDBbind_train_smiles", BOND_CATS)

    compare_distributions(
        ref=pdb_atoms_ref,
        test=pdbbind_atoms_test,
        title="ATOM COUNTS TEST: PDBbind_pickle(ref) vs PDBbind_train_smiles(test)",
    )
    compare_distributions(
        ref=pdb_bonds_ref,
        test=pdbbind_bonds_test,
        title="BOND COUNTS TEST: PDBbind_pickle(ref) vs PDBbind_train_smiles(test)",
    )

    # ----------------------------
    # 4) Plot (updated per your spec)
    # ----------------------------
    atoms_wide = pivot_for_plot(df_atoms_long, metric="fraction")
    bonds_wide = pivot_for_plot(df_bonds_long, metric="fraction")
    rings_wide = pivot_for_plot(df_rings_long, metric="fraction")

    # Enforced plotting order (must exist in columns)
    plot_order = [
        "PDBbind_train_smiles",
        "ours_guidanceBL",
        "crossdocked_train",
        "targetdiff",
        "diffsbdd",
        "pocket2mol",
        "drugflow",
    ]
    plot_order = [c for c in plot_order if c in atoms_wide.columns]

    # Reorder columns consistently
    atoms_wide = atoms_wide[plot_order]
    bonds_wide = bonds_wide[[c for c in plot_order if c in bonds_wide.columns]]
    rings_wide = rings_wide[[c for c in plot_order if c in rings_wide.columns]]

    # Drop bond type NONE from plotting (keep in CSV/test if you want, just don't plot)
    if "NONE" in bonds_wide.index:
        bonds_wide = bonds_wide.drop(index="NONE")

    # Colors (exact mapping)
    darkblue  = "#1F4E79"
    lightblue = "#9DC3E6"

    darkred   = "#8B0000"
    lightred1 = "#cf2359"
    lightred2 = "#cc5278"
    lightred3 = "#d16f8e"
    lightred4 = "#cc8fa2"

    colors = {
        "PDBbind_train_smiles": darkblue,
        "ours_guidanceBL":      lightblue,
        "crossdocked_train":    darkred,
        "targetdiff":           lightred1,
        "diffsbdd":             lightred2,
        "pocket2mol":           lightred3,
        "drugflow":             lightred4,
    }

    # Make sure all plotted sources have a color (fallback to gray if not)
    for src in plot_order:
        colors.setdefault(src, "#999999")

    # Shared y-axis across subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True, sharey=True)

    grouped_barplot(
        axes[0],
        atoms_wide,
        colors=colors,
        # title= "",#"Atom type distribution",
        ylabel="Fraction",
        label_letter="A",
    )
    grouped_barplot(
        axes[1],
        bonds_wide,
        colors=colors,
        # title=""#Bond type distribution",
        ylabel=None,  # shared y label on A
        label_letter="B",
    )
    grouped_barplot(
        axes[2],
        rings_wide,
        colors=colors,
        # title="",#"Ring size distribution (SSSR)",
        ylabel=None,
        label_letter="C",
    )

    # Horizontal dashed gridlines, alpha=0.5, shared y scaling
    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    # Put legend only in subplot C, single column
    handles, labels = axes[2].get_legend_handles_labels()
    axes[2].legend(
        handles, 
        [
        "MISATO",
        "Ours",
        "CrossDocked2020",
        "TargetDiff",
        "DiffSBDD",
        "Pocket2Mol",
        "DrugFlow",
        ], 
        loc="upper left", 
        frameon=False, 
        ncol=1)

    # Save
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved figure: {fig_path}")

    print(f"✓ Cached CSVs:\n  - {atoms_csv}\n  - {bonds_csv}\n  - {rings_csv}\n  - {stats_csv}")
