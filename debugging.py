from moldiff.Dataset import PDBbind_Dataset

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

# try to reconstruct sample error
config = CONFIG.copy()
loaded_model = load_checkpoint(f"training_runs/1010_153623_datasets/dataset_pdbbind_train/checkpoint_epoch_30.pt")
eval_data = create_batches_from_dataset("datasets/dataset_pdbbind_validation.pt", {"batch_size": 16})
sample_loader = eval_data[0] # TODO should be replaced by a test dataset
print(scatter_mean(sample_loader["pocket_coords"], sample_loader["pocket_mask"], dim=0))  # print center of masses
samples = sample_molecules_conditionally(
    loaded_model, 
    sample_loader,
    num_steps = config['num_sampling_steps'], 
    schedule_type=config['schedule_type'], 
    guidance_scale = config["cfg_guidance_scale"] # attach guidance scale
)
# error should occur here
molecules = build_mol_objects(samples)
print("Molecules built without error")