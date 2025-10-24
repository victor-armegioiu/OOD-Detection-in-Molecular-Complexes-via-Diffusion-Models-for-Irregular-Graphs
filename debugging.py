from moldiff.Dataset import PDBbind_Dataset
from moldiff.constants import atom_decoder
import torch

from moldiff.metrics import (
    load_checkpoint,
    sample_molecules,
    sample_molecules_conditionally, 
    evaluate_atom_aa_distributions, 
    build_mol_objects
)

from main_optimize import (
    CONFIG,  
    create_batches_from_dataset
)
from torch_scatter import scatter_mean

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def write_ligand_to_sdf(coords, elements, filename="ligand.sdf", charge_threshold=0.4):
    """
    Build and write a ligand to SDF given atomic coordinates and element types.
    Bonds are inferred based on covalent radii and distance heuristics.

    Args:
        coords (np.ndarray): (N, 3) array of Cartesian coordinates in Å.
        elements (list[str]): list of element symbols (e.g., ['C','O','N',...]).
        filename (str): path to output SDF file.
        charge_threshold (float): tolerance factor for bond detection distance.

    Returns:
        mol (rdkit.Chem.Mol): RDKit molecule object.
    """
    from rdkit.Chem import rdchem

    # --- 1. Create editable molecule ---
    mol = Chem.RWMol()
    atom_indices = []
    for elem in elements:
        a = Chem.Atom(elem)
        atom_indices.append(mol.AddAtom(a))

    # --- 2. Infer bonds using covalent radii ---
    cov_radii = Chem.GetPeriodicTable().GetRcovalent
    n_atoms = len(elements)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            rij = np.linalg.norm(coords[i] - coords[j])
            cutoff = cov_radii(elements[i]) + cov_radii(elements[j]) + charge_threshold
            if rij < cutoff:  # likely bonded
                bond_order = Chem.rdchem.BondType.SINGLE
                mol.AddBond(i, j, bond_order)

    # --- 3. Embed coordinates ---
    conf = Chem.Conformer(n_atoms)
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)

    # --- 4. Sanitize and adjust ---
    mol = mol.GetMol()
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ADJUSTHS)

    # --- 5. Write SDF ---
    # writer = Chem.SDWriter(filename)
    # writer.write(mol)
    # writer.close()

    return mol

n_proteins = 1
n_samples_of_protein = 100
# try to reconstruct sample error
config = CONFIG.copy()
loaded_model = load_checkpoint(f"training_runs/1022_113052_datasets/dataset_pdbbind_train/checkpoint_final.pt")
eval_data = create_batches_from_dataset("datasets/dataset_pdbbind_validation.pt", {"batch_size": n_proteins})
sample_loader = eval_data[0] # TODO should be replaced by a test dataset
# print(scatter_mean(sample_loader["pocket_coords"], sample_loader["pocket_mask"], dim=0))  # print center of masses
for p in range(n_samples_of_protein):
    samples = sample_molecules_conditionally(
        loaded_model, 
        sample_loader,
        num_steps = config['num_sampling_steps'], 
        schedule_type=config['schedule_type'], 
        guidance_scale = config["cfg_guidance_scale"] # attach guidance scale
    )
    # print(sample_loader["ids"])
    for s in range(n_proteins):
        ligand_coords = samples['ligand_coords'][samples['ligand_mask']==s].cpu().detach().numpy()
        ligand_features = samples['ligand_features'][samples['ligand_mask']==s].cpu()
        ligand_elements = [atom_decoder[idx] for idx in torch.argmax(ligand_features, dim=-1)]
        mol = write_ligand_to_sdf(
            ligand_coords, 
            ligand_elements, 
            filename=f"{sample_loader['ids'][s]}_ligand{p}.sdf", 
            charge_threshold=0.2
            )
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            writer = Chem.SDWriter(f"{sample_loader['ids'][s]}_ligand{p}.sdf")
            writer.write(mol)
            writer.close()
        except Exception as e: 
            print(e)
    
    # graph = Data(
    #     ligand_coords=s,
    #     ,
    #     ,
    #     pocket_features=samples['pocket_features'][samples['pocket_mask']==s].cpu()
    # )