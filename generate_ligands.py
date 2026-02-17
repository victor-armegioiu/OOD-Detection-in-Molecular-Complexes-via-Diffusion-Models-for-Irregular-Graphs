import torch
import numpy as np
import os
import moldiff.utils as utils
from rdkit import Chem
from rdkit.Chem import SDWriter
import argparse
from moldiff.metrics import (
    load_checkpoint,
    sample_molecules_conditionally, 
    build_molecule, 
    process_molecule
)
from moldiff.constants import atom_decoder
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

def save_diffusion_trajectories_as_sdf(
    samples: dict,
    pocket_output_dir: str
):
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

def _rdkit_atom_only_mol_from_coords_and_types(coords_tA3, types_tA):
    """
    Build an RDKit Mol with atoms + a conformer, but NO bonds.
    coords_tA3: torch.Tensor [A, 3] (float)
    types_tA:   torch.Tensor [A]    (long), values index into atom_decoder
    """
    # Safety: detach -> cpu -> python types
    coords = coords_tA3.detach().cpu().float().numpy()
    types = types_tA.detach().cpu().long().numpy()

    rw = Chem.RWMol()
    for idx in types:
        sym = atom_decoder[int(idx)]
        rw.AddAtom(Chem.Atom(sym))

    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x, y, z = coords[i]
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    return mol

def save_diffusion_trajectories_as_multiframe_sdf(
    samples: dict,
    pocket_output_dir: str, 
    path_save_step: int = 1
):
    """
    Assumes samples contain (batched):
      - samples["ligand_coords"]   : [B, T, A, 3] OR compatible with utils.batch_to_list -> [T, A, 3]
      - samples["ligand_features"] : [B, T, A, 10] OR compatible -> [T, A, 10]
      - samples["ligand_mask"]     : mask for utils.batch_to_list

    Writes per-sample SDF:
      frames 0..T-2  : atom-only mol (no bonds)
      frame  T-1     : atom-only mol (no bonds)
      frame  T-1     : full mol with bonds via build_molecule/process_molecule
    """

    os.makedirs(pocket_output_dir, exist_ok=True)

    # You said you handle dim==2 elsewhere; we assume per-sample tensors are [T, A, ...]
    coords_list = utils.batch_to_list(samples["ligand_coords"], samples["ligand_mask"], atom_dim=1)
    feats_list  = utils.batch_to_list(samples["ligand_features"], samples["ligand_mask"], atom_dim=1)

    for i, (traj_coords, traj_feats) in enumerate(zip(coords_list, feats_list)):
        # traj_coords: [T, A, 3]
        # traj_feats : [T, A, 10]
        if traj_coords.dim() != 3 or traj_feats.dim() != 3:
            raise ValueError(
                f"Expected trajectory tensors with dim==3. Got coords dim {traj_coords.dim()}, feats dim {traj_feats.dim()}."
            )
        T = traj_coords.shape[0]

        # Decode atom types per frame: [T, A]
        traj_types = torch.argmax(traj_feats, dim=-1)

        save_path = os.path.join(pocket_output_dir, f"{i}_ligand.sdf")
        w = SDWriter(save_path)
        w.SetKekulize(False)

        # ---- 1) Write frames 0..T-2 as "atom-only" ----
        for t in range(0, max(T - 1, 0), path_save_step):
            mol_t = _rdkit_atom_only_mol_from_coords_and_types(
                coords_tA3=traj_coords[t],
                types_tA=traj_types[t],
            )
            mol_t.SetProp("_Name", f"ligand_{i}_t{t:04d}_atoms_only")
            w.write(mol_t)

        # ---- 2) Write last frame once as "atom-only" ----
        t_last = T - 1
        mol_last_atoms = _rdkit_atom_only_mol_from_coords_and_types(
            coords_tA3=traj_coords[t_last],
            types_tA=traj_types[t_last],
        )
        mol_last_atoms.SetProp("_Name", f"ligand_{i}_t{t_last:04d}_atoms_only")
        w.write(mol_last_atoms)

        # ---- 3) Write last frame again as "full" molecule with bonds ----
        # Your build_molecule expects coords [A,3] and index_atom_types [A]
        mol_full = build_molecule(traj_coords[t_last], traj_types[t_last])
        mol_full = process_molecule(
            mol_full,
            add_hydrogens=False,
            sanitize=False,
            relax_iter=0,
            largest_frag=False,
        )
        if mol_full is not None:
            mol_full.SetProp("_Name", f"ligand_{i}_t{t_last:04d}_with_bonds")
            w.write(mol_full)

        w.close()

def sample_molecule_given_all_atom_graph(
        graph_path: str,
        model: ConditionalMolecularDenoisingModel,
        output_dir: str,
        pocket_source: str,
        n_samples: int = 2,
        num_steps: int = 600,
        schedule_type: str = "exponential",
        guidance_scale: float = 0, 
        return_full_paths: bool = False, 
        path_save_step: int = 1
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
        guidance_scale=guidance_scale, 
        return_full_paths=return_full_paths
    )

    if not return_full_paths:
        save_diffusion_trajectories_as_sdf(samples, pocket_output_dir)
    else:
        save_diffusion_trajectories_as_multiframe_sdf(samples, pocket_output_dir, path_save_step=path_save_step)

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

