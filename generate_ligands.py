import torch
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

def test_sample_molecule_given_graph(pdb_path: str = "data_extracted_graphs/1a1e.pt"):
    model = load_checkpoint("/cluster/work/math/pbaertschi/molecular-diffusion/training_runs/RM_COM_DiffSBDD_config_1029_091048_datasets/dataset_pdbbind_train/checkpoint_epoch_580.pt")
    samples = sample_molecule_given_graph(
        pdb_path, 
        model, 
        output_dir="test_output",
        pocket_source="data_extracted_graphs",
        n_samples=20
    )
    # print(samples)

if __name__ == "__main__":
    # _ = test_sample_molecule_given_graph()
    def parse_args():
        parser = argparse.ArgumentParser(description="Protein-Ligand Graph Encoder")
        parser.add_argument("--protein_source", required=True, help="Protein file or directory with input PDB files")
        parser.add_argument("--ligand_source", required=True, help="Ligand file or directory of ligand files")
        parser.add_argument("--graph_dir", required=True, help="Optional output directory for saving graphs")
        parser.add_argument("--output_dir", required=True, help="Output directory for saving generated molecules and copied pockets")
        parser.add_argument("--ckpt_path",  default="/cluster/work/math/pbaertschi/molecular-diffusion/training_runs/RM_COM_DiffSBDD_config_1029_091048_datasets/dataset_pdbbind_train/checkpoint_epoch_580.pt")
        parser.add_argument("--n_samples", type=int, default=20, help="Number of samples")
        # parser.add_argument("--detect_lig_bonds_by_distance", type=lambda x: x.lower() == 'true', default=False, help="Detect ligand bonds by distance")
        args = parser.parse_args()
        return args

    # instantiate parser
    args = parse_args()
    print(f"Using args: {args}")

    # instantiate encoder
    encoder = ProteinGraphEncoder(granularity="residue-level-fully-connected")

    # get protein ligand pairs and check if already encoded
    protein_ligand_pairs = find_protein_ligand_pairs(args.protein_source, args.ligand_source)
    # find pt files in graph dir
    encoded_graphs = set([f.replace(".pt", "") for f in os.listdir(args.graph_dir) if f.endswith(".pt")])
    protein_ligand_pairs = [pair for pair in protein_ligand_pairs if pair[0].split("/")[-1].replace(".pdb", "") not in encoded_graphs]
    if len(protein_ligand_pairs) > 0:
        encode_protein_ligand_pairs(
            protein_ligand_pairs, 
            encoder, 
            args.graph_dir, 
            True, 
            False
        )
        print("Encoded pdb files. Processing with sampling...")
    else:
        print("All pbd files already encoded as graphs. Proceeding with sampling...")

    model = load_checkpoint(args.ckpt_path)
    os.makedirs(args.output_dir, exist_ok=True)
    print(encoded_graphs)
    
    for graph_code in encoded_graphs: 
        print(f"Sampling {args.n_samples} Ligands for Pocket {graph_code}")
        graph_path = f"{args.graph_dir}/{graph_code}.pt"
        _ = sample_molecule_given_graph(
            graph_path, 
            model, 
            output_dir=args.output_dir,
            pocket_source=args.protein_source,
            n_samples=args.n_samples,
            num_steps=600,
            schedule_type="exponential",
            guidance_scale=0
        )
