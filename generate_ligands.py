import torch
import numpy as np
import os
import moldiff.utils as utils
from rdkit.Chem import SDWriter
import argparse
from moldiff.metrics import (
    load_checkpoint,
    sample_molecules_conditionally, 
    build_molecule, 
    process_molecule
)
from moldiff.molecular_diffusion import ConditionalMolecularDenoisingModel
from moldiff.protlig_encoder import find_protein_ligand_pairs,  encode_protein_ligand_pairs, ProteinGraphEncoder
from datetime import datetime

import traceback


def sample_molecule_given_graph(graph_path: str,
                                model: ConditionalMolecularDenoisingModel,
                                output_dir: str,
                                pocket_source: str,
                                n_samples: int = 2,
                                num_steps: int = 600,
                                schedule_type: str = "exponential",
                                guidance_scale: float = 0
                                ):
    """Generate new molecules conditioned on a given pocket graph file and save them in individual files"""

    pocket_code = graph_path.split("/")[-1].replace(".pt", "")
    
    # Create directory for this pocket's samples
    pocket_output_dir = os.path.join(output_dir, pocket_code)
    os.makedirs(pocket_output_dir, exist_ok=True)
    # try: 
    #     os.makedirs(pocket_output_dir, exist_ok=False)
    # except FileExistsError:
    #     print(f"Output directory {pocket_output_dir} already exists. Skipping sampling for pocket {pocket_code}.")
    #     return None

    # Copy the original pocket PDB file
    pocket_pdb_path = os.path.join(pocket_source, f"{pocket_code}.pdb")
    if os.path.exists(pocket_pdb_path):
        for i in range(n_samples):
            target_pdb_path = os.path.join(pocket_output_dir, f"{i}_pocket.pdb")
            os.system(f"cp {pocket_pdb_path} {target_pdb_path}")
    
    graph = torch.load(graph_path)
    prot_coords = graph.pos[graph.prot_mask]
    prot_feats = graph.x[graph.prot_mask][:, model.atom_nf:]

    assert prot_coords.shape[0] == prot_feats.shape[0], f"Shape mismatch at dim one: Coords {prot_coords.shape} and Feats: {prot_feats.shape}"

    # expand the coordinates and features to the desired n_samples
    pocket_coords_expanded = prot_coords.tile((n_samples, 1))
    pocket_feats_expanded = prot_feats.tile((n_samples, 1))
    pocket_mask = torch.arange(n_samples).repeat_interleave(prot_coords.shape[0])

    sample_batch = {
        'pocket_mask': pocket_mask,
        'pocket_coords': pocket_coords_expanded,
        'pocket_features': pocket_feats_expanded
    }

    # Sample molecules conditionally
    samples = sample_molecules_conditionally(
        model=model,
        sample_batch=sample_batch,
        num_steps=num_steps,
        schedule_type=schedule_type,
        guidance_scale=guidance_scale
    )

    # Save each molecule in its own file
    for i, (atom_coords, atom_types) in enumerate(zip(
        utils.batch_to_list(samples["ligand_coords"], samples["ligand_mask"]),
        utils.batch_to_list(samples["ligand_features"], samples["ligand_mask"])
        )):
        index_atom_types = torch.argmax(atom_types, dim=-1)
        mol = build_molecule(atom_coords, index_atom_types)
        mol = process_molecule(mol,
                                add_hydrogens=False,
                                sanitize=False,          # bool
                                relax_iter=0,       # int
                                largest_frag=False          # bool
                                )
        if mol is not None:
            # Create individual SDF file for this molecule
            save_path = os.path.join(pocket_output_dir, f"{i}_ligand.sdf")
            w = SDWriter(save_path)
            w.SetKekulize(False)
            mol.SetProp("_Name", f"ligand_{i}")
            w.write(mol)
            w.close()
        
        # w = Chem.SDWriter(str(save_path))
        # w.SetKekulize(False)
        # for m in molecules:
        #     if m is not None:
        #         w.write(m)

    return samples

def sample_molecule_given_all_atom_graph(
        graph_path: str,
        model: ConditionalMolecularDenoisingModel,
        output_dir: str,
        pocket_source: str,
        n_samples: int = 2,
        num_steps: int = 600,
        schedule_type: str = "exponential",
        guidance_scale: float = 0
    ):
    """Generate new molecules conditioned on a given pocket graph file and save them in individual files"""

    pocket_code = graph_path.split("/")[-1].replace(".pt", "")
    
    # Create directory for this pocket's samples
    pocket_output_dir = os.path.join(output_dir, pocket_code)
    os.makedirs(pocket_output_dir, exist_ok=True)
    # try: 
    #     os.makedirs(pocket_output_dir, exist_ok=False)
    # except FileExistsError:
    #     print(f"Output directory {pocket_output_dir} already exists. Skipping sampling for pocket {pocket_code}.")
    #     return None

    # Copy the original pocket PDB file
    pocket_pdb_path = os.path.join(pocket_source, f"{pocket_code}.pdb")
    if os.path.exists(pocket_pdb_path):
        for i in range(n_samples):
            target_pdb_path = os.path.join(pocket_output_dir, f"{i}_pocket.pdb")
            os.system(f"cp {pocket_pdb_path} {target_pdb_path}")
    
    graph = torch.load(graph_path)
    prot_coords = graph.pos[graph.prot_mask & graph.ca_mask]
    prot_feats =  graph.x[graph.prot_mask & graph.ca_mask][:, 2*model.atom_nf:]               
    prot_sidechain_coords = graph.pos[graph.prot_mask]
    prot_sidechain_features = graph.x[graph.prot_mask][:, model.atom_nf:2*model.atom_nf] # here we only want atom features

    assert prot_coords.shape[0] == prot_feats.shape[0], f"Shape mismatch at dim one: Coords {prot_coords.shape} and Feats: {prot_feats.shape}"

    # expand the coordinates and features to the desired n_samples
    pocket_coords_expanded = prot_coords.tile((n_samples, 1))
    pocket_feats_expanded = prot_feats.tile((n_samples, 1))
    # prot_sidechain_coords_expanded = prot_sidechain_coords.tile((n_samples, 1))
    # prot_sidechain_features_expanded = prot_sidechain_features.tile((n_samples, 1))
    pocket_mask = torch.arange(n_samples).repeat_interleave(prot_coords.shape[0])


    sample_batch = {
        'pocket_mask': pocket_mask,
        'pocket_coords': pocket_coords_expanded,
        'sidechain_coords': prot_sidechain_coords,
        'pocket_features': pocket_feats_expanded, # pocket_feats_expanded, -> we reuse the same sidechain coords in all of molecular sampler
        'sidechain_features': prot_sidechain_features, # prot_sidechain_features_expanded
    }

    # Sample molecules conditionally
    samples = sample_molecules_conditionally(
        model=model,
        sample_batch=sample_batch,
        num_steps=num_steps,
        schedule_type=schedule_type,
        guidance_scale=guidance_scale
    )

    # Save each molecule in its own file
    for i, (atom_coords, atom_types) in enumerate(zip(
        utils.batch_to_list(samples["ligand_coords"], samples["ligand_mask"]),
        utils.batch_to_list(samples["ligand_features"], samples["ligand_mask"])
        )):
        index_atom_types = torch.argmax(atom_types, dim=-1)
        mol = build_molecule(atom_coords, index_atom_types)
        mol = process_molecule(mol,
                                add_hydrogens=False,
                                sanitize=False,          # bool
                                relax_iter=0,       # int
                                largest_frag=False          # bool
                                )
        if mol is not None:
            # Create individual SDF file for this molecule
            save_path = os.path.join(pocket_output_dir, f"{i}_ligand.sdf")
            w = SDWriter(save_path)
            w.SetKekulize(False)
            mol.SetProp("_Name", f"ligand_{i}")
            w.write(mol)
            w.close()
        
        # w = Chem.SDWriter(str(save_path))
        # w.SetKekulize(False)
        # for m in molecules:
        #     if m is not None:
        #         w.write(m)

    return samples


def test_sample_molecule_given_graph(pdb_path: str = "data_extracted_graphs/1a1e.pt"):
    model = load_checkpoint("/cluster/work/math/pbaertschi/molecular-diffusion/training_runs/RETRAIN_norm1_1128_084449_datasets/datasetmd_pdbbind_train/checkpoint_epoch_210.pt")
    samples = sample_molecule_given_graph(
        pdb_path, 
        model, 
        output_dir="test_output",
        pocket_source="data_extracted_graphs",
        n_samples=20
    )

def test_sample_molecule_given_all_atom_graph(pdb_path: str = "../bioinformatics/data_extracted_graphs_atom_level/1a1e.pt"):
    model = load_checkpoint("/cluster/work/math/pbaertschi/molecular-diffusion/training_runs/RETRAIN_norm1_1128_084449_datasets/datasetmd_pdbbind_train/checkpoint_epoch_210.pt")
    samples = sample_molecule_given_all_atom_graph(
        pdb_path, 
        model, 
        output_dir="test_output",
        pocket_source="data_extracted_graphs",
        n_samples=20
    )
    

def parse_args():
    parser = argparse.ArgumentParser(description="Protein-Ligand Graph Encoder")
    parser.add_argument("--protein_source", required=True, help="Protein file or directory with input PDB files")
    parser.add_argument("--ligand_source", required=True, help="Ligand file or directory of ligand files")
    parser.add_argument("--graph_dir", required=True, help="Optional output directory for saving graphs")
    parser.add_argument("--output_dir", required=True, help="Output directory for saving generated molecules and copied pockets")
    parser.add_argument("--ckpt_path",  default="/cluster/work/math/pbaertschi/molecular-diffusion/training_runs/RM_COM_DiffSBDD_config_1029_091048_datasets/dataset_pdbbind_train/checkpoint_epoch_580.pt")
    parser.add_argument("--n_samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--guidance_scale", type=float, default=0., help="Guidance scale for classifier guidance")
    parser.add_argument("--job_id", type=int, default=0, help="Job ID for parallel processing")
    parser.add_argument("--n_jobs", type=int, default=1, help="Total number of parallel jobs")
    # parser.add_argument("--detect_lig_bonds_by_distance", type=lambda x: x.lower() == 'true', default=False, help="Detect ligand bonds by distance")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # _ = test_sample_molecule_given_all_atom_graph()

    # --------------------------
    # Parse arguments
    # --------------------------
    args = parse_args()
    print(f"Using args: {args}")

    # --------------------------
    # Instantiate encoder
    # --------------------------
    granularity = "atom-level"#"residue-level-fully-connected"
    encoder = ProteinGraphEncoder(
        granularity=granularity
    )

    # --------------------------
    # List existing encoded graphs
    # --------------------------
    graph_dir = args.graph_dir
    manifest_path = os.path.join(graph_dir, "fixed_processing_order_manifest.txt")

    # Warning: All shards MUST see same folder structure
    existing_pts = [
        f.replace(".pt", "")
        for f in os.listdir(graph_dir)
        if f.endswith(".pt")
    ]
    existing_pts = sorted(np.unique(existing_pts))

    # --------------------------
    # Determine which protein-ligand pairs still need encoding
    # --------------------------
    protein_ligand_pairs = find_protein_ligand_pairs(
        args.protein_source, args.ligand_source
    )

    missing_pairs = [
        pair
        for pair in protein_ligand_pairs
        if pair[0].split("/")[-1].replace(".pdb", "") not in existing_pts
    ]

    # --------------------------
    # Encode missing graphs (if any)
    # --------------------------
    if len(missing_pairs) > 0:
        print(f"[INFO] Encoding {len(missing_pairs)} missing graph files...")
        encode_protein_ligand_pairs(
            missing_pairs,
            encoder,
            graph_dir,
            # detect_ligand_bonds=True,
            # remove_hydrogens=False,
        )
        print("[INFO] Finished encoding.")
    else:
        print("[INFO] All PDB files already encoded.")

    # --------------------------
    # Create manifest file (AFTER encoding)
    # --------------------------
    if not os.path.exists(manifest_path):
        print(f"[INFO] Creating manifest at: {manifest_path}")

        pt_files = sorted([
            os.path.join(graph_dir, f)
            for f in os.listdir(graph_dir)
            if f.endswith(".pt")
        ])

        with open(manifest_path, "w") as f:
            for p in pt_files:
                f.write(p + "\n")

        print("[INFO] Manifest written.")
    else:
        print("[INFO] Found existing manifest:", manifest_path)

    # --------------------------
    # Load manifest consistently
    # --------------------------
    with open(manifest_path, "r") as f:
        manifest_graph_paths = [line.strip() for line in f]

    # Convert to graph codes
    encoded_graphs = [
        os.path.basename(p).replace(".pt", "")
        for p in manifest_graph_paths
    ]

    print(f"[INFO] Total graphs found: {len(encoded_graphs)}")

    # --------------------------
    # Deterministic sharding
    # --------------------------
    chunks = np.array_split(encoded_graphs, args.n_jobs)
    my_graphs = chunks[args.job_id]

    print(f"[INFO][SHARD {args.job_id}] Assigned graphs: {len(my_graphs)}")

    # --------------------------
    # Load diffusion model
    # --------------------------
    print(f"[INFO] Loading checkpoint: {args.ckpt_path}")
    for attempt in range(10):
        try:
            model = load_checkpoint(args.ckpt_path)
            break
        except Exception as e:
            print(f"[WARN] CUDA init failed ({attempt+1}): {e}")
    else:
        raise RuntimeError("CUDA initialization failed after 10 attempts.")

    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------------
    # MAIN SAMPLING LOOP
    # --------------------------
    sample_molecule_given_graph_ = sample_molecule_given_graph if granularity != "atom-level" else sample_molecule_given_all_atom_graph
    for graph_code in my_graphs:
        graph_path = os.path.join(graph_dir, f"{graph_code}.pt")

        print(f"[SHARD {args.job_id}] Sampling for pocket: {graph_code}")

        try:
            _ = sample_molecule_given_graph_(
                graph_path=graph_path,
                model=model,
                output_dir=args.output_dir,
                pocket_source=args.protein_source,
                n_samples=args.n_samples,
                num_steps=600,
                schedule_type="exponential",
                guidance_scale=args.guidance_scale,
            )

        except Exception as e:
            print(f"[ERROR][{graph_code}] {e}")
            tb = traceback.format_exc()
            print(tb)
            continue

    print(f"[SHARD {args.job_id}] All assigned pockets processed.")

