#!/usr/bin/env python3
import argparse
import sys
import pickle
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Minimal eval wrapper for sbdd_metrics.")
    # ap.add_argument("--sbdd-metrics-dir", required=True,
    #                 help="Path to the sbdd_metrics folder (added to PYTHONPATH).")
    ap.add_argument("--samples-dir", required=True,
                    help="Root with per-pocket subdirs containing *_ligand.sdf / *_pocket.pdb.")
    ap.add_argument("--datadir", required=False, default=None,
                    help="Directory containing train_smiles.npy.")
    ap.add_argument("--gnina", default="gnina",
                    help="Path to gnina executable or 'gnina' if on PATH.")
    ap.add_argument("--reduce", default="", help="Path to MolProbity 'reduce'; leave empty to disable.")
    ap.add_argument("--n-samples", default="all",
                    help="'all' for all pairs, or an integer cap per pocket.")
    ap.add_argument("--exclude", default="",
                    help="Comma/space-separated evaluators to skip (e.g. 'gnina_docking,fcd').")
    args = ap.parse_args()

    # # Make sbdd_metrics importable
    # sbdd_metrics_dir = str(Path(args.sbdd_metrics_dir).resolve())
    # if sbdd_metrics_dir not in sys.path:
    #     sys.path.insert(0, sbdd_metrics_dir)

    # Import here so the path above is honored.
    # Adjust the import path if your repo structure differs:
    from sbdd_metrics.evaluation import compute_all_metrics_drugflow

    # Parse simple args
    samples_dir = Path(args.samples_dir).resolve()
    reference_smiles_path = Path(args.datadir).resolve() / "train_smiles.npy" if args.datadir else None
    reduce_path = args.reduce if args.reduce.strip() else None

    # n_samples: None means "all"
    if str(args.n_samples).strip().lower() in {"all", "none", "null"}:
        n_samples = None
    else:
        n_samples = max(int(args.n_samples), 0)

    # exclude evaluators
    exclude = [t for t in args.exclude.replace(",", " ").split() if t]
    if reduce_path is None and "gnina_docking" not in exclude:
        exclude.append("gnina_docking")  # auto-skip docking if reduce not provided

    print("Evaluation...")
    data, table_detailed, table_aggregated = compute_all_metrics_drugflow(
        in_dir=samples_dir,
        gnina_path=args.gnina,
        reduce_path=reduce_path,
        reference_smiles_path=reference_smiles_path,
        n_samples=n_samples,
        exclude_evaluators=exclude or None,
    )

    # Save outputs
    with open(samples_dir / "metrics_data.pkl", "wb") as f:
        pickle.dump(data, f)
    table_detailed.to_csv(samples_dir / "metrics_detailed.csv", index=False)
    table_aggregated.to_csv(samples_dir / "metrics_aggregated.csv", index=False)
    print("Wrote metrics_data.pkl, metrics_detailed.csv, metrics_aggregated.csv to:", samples_dir)

if __name__ == "__main__":
    main()

    # information provided by evlauation obj is pkl and data is detailed table:
    # >>> set(data.columns) & set(obj[0].keys())
    # {'chembl_ring_systems.min_ring_freq_gt100_', 'gnina.cnn_score', 'posebusters.internal_steric_clash', 'posebusters.all', 'posebusters.all_atoms_connected', 'chembl_ring_systems.min_ring_freq_gt10_', 'clashes.clash_score_between', 'clashes.passed_clash_score_between', 'medchem.lipinski', 'medchem.size', 'posebusters.bond_angles', 'gnina.gnina_efficiency', 'representation.smiles', 'sample', 'clashes.clash_score_pockets', 'posebusters.protein-ligand_maximum_distance', 'medchem.valid', 'reos.all', 'posebusters.inchi_convertible', 'gnina.minimisation_rmsd', 'pdb_file', 'gnina.vina_score', 'posebusters.sanitization', 'reos.BMS', 'medchem.connected', 'gnina.vina_efficiency', 'reos.Inpharmatica', 'reos.SureChEMBL', 'sdf_file', 'posebusters.minimum_distance_to_inorganic_cofactors', 'clashes.passed_clash_score_ligands', 'reos.Glaxo', 'chembl_ring_systems.min_ring_smi', 'posebusters.mol_pred_loaded', 'posebusters.minimum_distance_to_organic_cofactors', 'reos.PAINS', 'posebusters.volume_overlap_with_protein', 'posebusters.internal_energy', 'posebusters.volume_overlap_with_organic_cofactors', 'chembl_ring_systems.min_ring_freq_gt0_', 'subdir', 'posebusters.volume_overlap_with_inorganic_cofactors', 'medchem.sa', 'clashes.passed_clash_score_pockets', 'posebusters.volume_overlap_with_waters', 'reos.Dundee', 'posebusters.bond_lengths', 'posebusters.minimum_distance_to_waters', 'posebusters.mol_cond_loaded', 'posebusters.double_bond_flatness', 'medchem.logp', 'medchem.n_rotatable_bonds', 'reos.MLSMR', 'energy.energy', 'posebusters.aromatic_ring_flatness', 'posebusters.minimum_distance_to_protein', 'reos.LINT', 'gnina.gnina_score', 'medchem.qed', 'clashes.clash_score_ligands'}
    # >>> len(set(data.columns) & set(obj[0].keys()))
    # 60
    # >>> len(set(data.columns) not in set(obj[0].keys()))
    # Traceback (most recent call last):
    # File "<stdin>", line 1, in <module>
    # TypeError: object of type 'bool' has no len()
    # >>> len(set(data.columns) | set(obj[0].keys()))
    # 89
    # >>> len(set(data.columns))
    # 71
    # >>> len(set(obj[0].keys()))
    # 78
    # >>> set(data.columns) - set(obj[0].keys())
    # {'ring_count.num_20_rings', 'ring_count.num_3_rings', 'ring_count.num_13_rings', 'ring_count.num_4_rings', 'ring_count.num_5_rings', 'ring_count.num_9_rings', 'ring_count.num_6_rings', 'ring_count.num_7_rings', 'ring_count.num_12_rings', 'ring_count.num_8_rings', 'ring_count.num_11_rings'}
    # >>> set(obj[0].keys()) - set(data.columns)
    # {'geometry.N-C-O', 'geometry.C-C-C', 'geometry.C-N-N', 'geometry.C-C', 'geometry.C=O', 'geometry.C-C-N', 'geometry.N=C-N', 'geometry.C-N-C', 'geometry.C-C-O', 'geometry.N-N-C', 'geometry.C-N', 'geometry.C=N', 'geometry.C-C=N', 'geometry.C-O', 'geometry.C-C=O', 'geometry.N-C=O', 'geometry.N-N', 'geometry.N-C=N'}
    # >>> obj[0][geometry.N-C-O]
    # Traceback (most recent call last):
    # File "<stdin>", line 1, in <module>
    # NameError: name 'geometry' is not defined
    # >>> obj[0]["geometry.N-C-O"]
    # [156.9606505519542]
    # >>> obj[1]["geometry.N-C-O"]
    # Traceback (most recent call last):
    # File "<stdin>", line 1, in <module>
    # KeyError: 'geometry.N-C-O'
    # >>> obj[2]["geometry.N-C-O"]
    # Traceback (most recent call last):
    # File "<stdin>", line 1, in <module>
    # KeyError: 'geometry.N-C-O'
    # >>> obj[0]["geometry.N-C-O"]
    # [156.9606505519542]
    # >>> obj[10]["geometry.N-C-O"]
    # [95.37373313934752]
    # >>> obj[20]["geometry.N-C-O"]
    # [123.13746843510931, 107.00363152474165, 108.73562102662363, 111.41766812798024, 105.57386347161642, 112.13168242228429, 146.21587171999124, 88.23241334367911]
