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
    create_batches_from_dataset, 
    create_molecular_model, 
    save_checkpoint
)
from moldiff.metrics import load_checkpoint
from torch_scatter import scatter_mean

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# CONFIG["update_pocket_coords"] = False
# print(CONFIG)
# model = create_molecular_model(CONFIG)
# p = "/cluster/scratch/pbaertschi/tmp/debug_ckpt.pt"
# save_checkpoint(model, CONFIG, save_path=p)
# model = load_checkpoint(p)
# print(model.__dict__)

path ="training_runs/TEST_1127_140716_datasets/dataset_pdbbind_train/checkpoint_final.pt"
print(load_checkpoint(path).__dict__)
checkpoint = torch.load(path, map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(checkpoint["config"])


