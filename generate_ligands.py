import torch
import os
import moldiff.utils as utils
from rdkit.Chem import SDWriter
from moldiff.metrics import (
    load_checkpoint,
    sample_molecules_conditionally, 
    build_molecule, 
    process_molecule
)
from moldiff.molecular_diffusion import ConditionalMolecularDenoisingModel

def sample_molecule_given_graph(graph_path: str,
                                model: ConditionalMolecularDenoisingModel,
                                save_path: str|None = None, 
                                n_samples: int = 2,
                                num_steps: int = 600,
                                schedule_type: str = "exponential",
                                guidance_scale: float = 0
                                ):
    """Generate new molecules conditioned on a given pocket graph file"""

    pocket_code = graph_path.split("/")[-1].replace(".pt", "")
    
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

    if save_path is not None:
        # os.makedirs(save_path, exist_ok=True)
        w = SDWriter(save_path)
        w.SetKekulize(False)
        # make sure the folder exists
        
        # molecules = []
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
                # give molecule an internal name
                mol.SetProp("_Name", f"ligand_{i+1}")
                # write molecule to its own SDF file
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
    samples = sample_molecule_given_graph(pdb_path, model, save_path = "1a1e_our_model.sdf", n_samples=20)
    # print(samples)

if __name__ == "__main__":
    # _ = test_sample_molecule_given_graph()
    model = load_checkpoint(ckpt_path)
    sample_molecule_given_graph(graph_path, model, save_path, n_samples,
                                num_steps: int = 600,
                                schedule_type: str = "exponential",
                                guidance_scale: float = 0
