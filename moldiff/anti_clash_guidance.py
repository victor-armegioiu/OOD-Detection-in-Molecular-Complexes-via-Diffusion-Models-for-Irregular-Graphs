# inspired by https://github.com/maabuu/posebusters/blob/main/posebusters/modules/intermolecular_distance.py#L16
# idea: for every ligand atom, get a gradient pulling them away from the neighbouring protein atoms

import torch
from main_optimize import create_batches_from_dataset
from moldiff.Dataset import PDBbind_Dataset
from moldiff.constants import atom_decoder
from torch_geometric.loader import DataLoader
from typing import List, Dict, Tuple, Optional

from rdkit.Chem.rdchem import GetPeriodicTable


def create_batches_from_dataset(dataset_path: str, config: Dict) -> List[Dict]:
    """Create dataset from PDBbind dataset"""
    data = []
    dataset = torch.load(dataset_path)
    dataloader = DataLoader(dataset, 
                            batch_size=config['batch_size'], 
                            shuffle=True, 
                            follow_batch=['lig_coords', 'prot_coords']
                            )

    for batch in dataloader:
        # track COM for later addition in conditional sampling
        # com = batch.prot_coords.mean(dim=0, keepdim=True) # torch.cat([batch.lig_coords, batch.prot_coords], dim=0).mean(dim=0, keepdim=True)
        # assert len(com) == config["batch_size"]

        batch = {
            'ligand_coords': batch.lig_coords,
            'ligand_features': batch.lig_features,
            'ligand_mask': batch.lig_coords_batch,

            'pocket_coords': batch.prot_coords,
            'pocket_features': batch.prot_features,
            'pocket_mask': batch.prot_coords_batch,

            'sidechain_coords': batch.prot_sidechain_coords, 
            'sidechain_features': batch.prot_sidechain_features,

            'batch_size': config['batch_size'],
            'ids': batch.id
            # track COM for later addition in conditional sampling
            # 'pocket_com': com #torch.cat([batch.lig_coords, batch.prot_coords], dim=0).mean(dim=0, keepdim=True)
        }
        data.append(batch)


    return data

PERIODIC_TABLE = GetPeriodicTable()


# load a random graph
batch_size = 1 # TODO make sure it works across masks (scatter funtctionof gemometric?)
graphs = create_batches_from_dataset("../bioinformatics/example_dataset_graph_level.pt", {"batch_size": batch_size})

batch = next(iter(graphs))
device = "cpu"

lig_coords = batch["ligand_coords"].to(device)
lig_features = batch["ligand_features"].to(device)  # One-hot features
pocket_coords = batch["pocket_coords"].to(device)
pocket_features = batch["pocket_features"].to(device)  # One-hot features
lig_mask = batch["ligand_mask"].to(device)
pocket_mask = batch["pocket_mask"].to(device)
batch_size = batch["batch_size"]
# new
sidechain_coords =  batch["sidechain_coords"].to(device)
sidechain_features = batch["sidechain_features"].to(device)

# assert
assert sidechain_coords.shape[1] == pocket_coords.shape[1] == lig_coords.shape[1] == 3
assert sidechain_features.shape[1] == lig_features.shape[1] == 10
assert pocket_features.shape[1] == 21

## clash guiding function
# TODO ignore types or mask H like posebusters needed? i don't think so, parellilzation across batch follow masks
# ask victor: why do we have to do gradient
radius_type: str = "vdw" # covalent
radius_scale: float = 1.0
clash_cutoff: float = 0.75
ignore_types: set[str] = {"hydrogens"}
max_distance: float = 5.0
search_distance: float = 4

# clash_cutoff

def _get_radius(a, radius_type="vdw"):
    if radius_type == "vdw":
        return PERIODIC_TABLE.GetRvdw(a)
    elif radius_type == "covalent":
        return PERIODIC_TABLE.GetRcovalent(a)
    else:
        raise ValueError(f"Unknown radius type {radius_type}. Valid values are 'vdw' and 'covalent'.")


# def _pairwise_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#     return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)

lig_coords_grad = lig_coords.clone().detach().requires_grad_(True)

# get radii
lig_radii = torch.tensor([
    _get_radius(atom_decoder[key], radius_type) for key in torch.argmax(lig_features, dim=1)
    ])
sidechain_radii = torch.tensor([
    _get_radius(atom_decoder[key], radius_type) for key in torch.argmax(sidechain_features, dim=1)
    ])

# select atoms that are close to ligand to check for clash
complete_distance_matrix = torch.cdist(lig_coords_grad, sidechain_coords) 
mask_protein = complete_distance_matrix.min(dim=0).values <= search_distance # search distance of interacting
in_search_distance_matrix = complete_distance_matrix[:, mask_protein]
sidechain_radii = sidechain_radii[mask_protein]
# mask_protein_idxs = mask_protein_idxs[mask_protein]

radii_sum = lig_radii[:, None] + sidechain_radii[None, :]
relative_distance = in_search_distance_matrix / radii_sum
distance_score_per_atom = relative_distance.topk(k=3, dim=1, largest=False).values.mean(dim=1).sum() # TODO or just the min -> or function that is easier to differentiate like logsumexp (higher values become smaller relative to smaller ones)

distance_score_per_atom.backward()
print(lig_coords_grad.grad)


# TODO iterative test to confirm that score improves over amplying grad 


#######



# violations = relative_distance < 1 / radius_scale

# if distances.size > 0:
#     violations[np.unravel_index(distances.argmin(), distances.shape)] = True  # add smallest distances as info
#     violations[np.unravel_index(relative_distance.argmin(), relative_distance.shape)] = True
# violation_ligand, violation_protein = np.where(violations)
# reverse_ligand_idxs = mask_ligand_idxs[violation_ligand]
# reverse_protein_idxs = mask_protein_idxs[violation_protein]

# # collect details around those violations in a dataframe
# details = pd.DataFrame()
# details["ligand_atom_id"] = reverse_ligand_idxs
# details["protein_atom_id"] = reverse_protein_idxs
# details["ligand_element"] = [lig_atoms[i] for i in violation_ligand]
# details["protein_element"] = [atoms_protein[i] for i in violation_protein]
# details["ligand_vdw"] = [radius_ligand[i] for i in violation_ligand]
# details["protein_vdw"] = [radius_protein[i] for i in violation_protein]
# details["sum_radii"] = details["ligand_vdw"] + details["protein_vdw"]
# details["distance"] = distances[violation_ligand, violation_protein]
# details["sum_radii_scaled"] = details["sum_radii"] * radius_scale
# details["relative_distance"] = details["distance"] / details["sum_radii_scaled"]
# details["clash"] = details["relative_distance"] < clash_cutoff

# results = {
#     "smallest_distance": details["distance"].min(),
#     "not_too_far_away": details["distance"].min() <= max_distance,
#     "num_pairwise_clashes": details["clash"].sum(),
#     "no_clashes": not details["clash"].any(),
# }

# # add most extreme values to results table
# i = np.argmin(details["relative_distance"]) if len(details) > 0 else None
# most_extreme = {"most_extreme_" + c: details.loc[i][str(c)] if i is not None else pd.NA for c in details.columns}
# results = {**results, **most_extreme}

# return {"results": results, "details": details}

