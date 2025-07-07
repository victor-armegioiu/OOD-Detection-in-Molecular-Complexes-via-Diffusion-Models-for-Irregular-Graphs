"""
Main Molecular Diffusion Pipeline

Complete demonstration of training, saving, loading, and sampling with molecular diffusion models.
All operations are CUDA-optimized.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import wandb
from typing import List, Dict
from Dataset import PDBbind_Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from datetime import datetime

# Import our molecular modules
from molecular_diffusion import (
    MolecularDenoisingModel, 
    exponential_noise_schedule,
    log_uniform_sampling,
    edm_weighting,
    MolecularDiffusion
)
from metrics import (
    load_checkpoint,
    sample_molecules, 
    evaluate_atom_aa_distributions
)
from egnn_dynamics import EGNNDynamics
import torch.nn.functional as F


# Configuration following framework patterns
CONFIG = {
    'train_dataset_path': 'dataset_pdbbind.pt',
    'eval_dataset_path': 'dataset_casf2016.pt',

    # Model parameters
    'atom_nf': 10,            # Number of atom types
    'residue_nf': 21,         # Number of residue types  
    'n_dims': 3,              # 3D coordinates
    'n_layers': 4,            # Number of EGNN layers
    'joint_nf': 256,          # Even richer shared space
    'hidden_nf': 128,         # More EGNN capacity
    'edge_embedding_dim': 32, # Edge embeddings
    
    # Training parameters
    'num_epochs': 1000,
    'learning_rate': 1e-4,
    'batch_size': 16,
    'log_interval': 20,
    'eval_interval': 2,
    'num_eval_samples': 100,
    
    # Diffusion parameters
    'sigma_max': 100.0,      # Maximum noise level
    'sigma_min': 1e-4,      # Minimum noise level
    'update_pocket_coords': True,  # Joint modeling
    
    # Sampling parameters
    'num_sampling_steps': 16,
    'schedule_type': "exponential",
    
    # Random data generation parameters
    'num_train_batches': 100,
    'num_eval_batches': 20,
    'ligand_size_range': (4, 12),     # Min/max ligand atoms
    'pocket_size_range': (15, 30),   # Min/max pocket residues
    
    # I/O
    'device': 'cuda',
    
    # Weights & Biases configuration
    'wandb': {
        'project': 'molecular-diffusion',
        'entity': 'dagraber',
        'tags': ['molecular-diffusion', 'training'],
        'notes': 'Training run for molecular diffusion model',
        'log_model': False,
        'log_gradients': False,  # Set to True to log gradient distributions
    }
}

# Create a run id with a timestamp
run_id = f"{datetime.now().strftime('%m%d_%H%M')}_{CONFIG['train_dataset_path'].split('.')[0]}"
CONFIG['wandb']['name'] = run_id
CONFIG['run_dir'] = f'training_runs/{run_id}'
if not os.path.exists(CONFIG['run_dir']):
    os.makedirs(CONFIG['run_dir'])
CONFIG['checkpoint_path'] = f'{CONFIG["run_dir"]}/checkpoint.pt'



def create_fake_molecular_batch(batch_size: int, ligand_size_range: tuple, 
                               pocket_size_range: tuple, atom_nf: int, 
                               residue_nf: int) -> Dict:
    """Create realistic fake molecular data for training (all CUDA)"""
    
    # Sample molecule sizes
    ligand_sizes = [
        np.random.randint(ligand_size_range[0], ligand_size_range[1] + 1) 
        for _ in range(batch_size)
    ]
    pocket_sizes = [
        np.random.randint(pocket_size_range[0], pocket_size_range[1] + 1)
        for _ in range(batch_size)
    ]
    
    total_lig_atoms = sum(ligand_sizes)
    total_pocket_atoms = sum(pocket_sizes)
    
    # Create masks (CUDA)
    ligand_mask = torch.cat([
        torch.full((size,), i, dtype=torch.long, device='cuda')
        for i, size in enumerate(ligand_sizes)
    ])
    pocket_mask = torch.cat([
        torch.full((size,), i, dtype=torch.long, device='cuda')
        for i, size in enumerate(pocket_sizes)
    ])
    
    # Generate coordinates (CUDA) - somewhat realistic molecular geometry
    # Ligands: smaller, more compact
    ligand_coords = torch.randn(total_lig_atoms, 3, device='cuda') * 2.0
    
    # Pockets: larger, more spread out
    pocket_coords = torch.randn(total_pocket_atoms, 3, device='cuda') * 5.0
    
    # Generate categorical features (one-hot encoded, CUDA)
    ligand_atom_types = torch.randint(0, atom_nf, (total_lig_atoms,), device='cuda')
    ligand_features = F.one_hot(ligand_atom_types, atom_nf).float()
    
    pocket_residue_types = torch.randint(0, residue_nf, (total_pocket_atoms,), device='cuda')
    pocket_features = F.one_hot(pocket_residue_types, residue_nf).float()
    
    # Make ligands roughly centered around pocket regions (more realistic)
    for i in range(batch_size):
        lig_mask_i = (ligand_mask == i)
        pocket_mask_i = (pocket_mask == i)
        
        if lig_mask_i.any() and pocket_mask_i.any():
            pocket_center = pocket_coords[pocket_mask_i].mean(dim=0)
            ligand_coords[lig_mask_i] += pocket_center + torch.randn(3, device='cuda') * 1.0
    
    return {
        'ligand_coords': ligand_coords,
        'ligand_features': ligand_features,
        'pocket_coords': pocket_coords,
        'pocket_features': pocket_features,
        'ligand_mask': ligand_mask,
        'pocket_mask': pocket_mask,
        'batch_size': batch_size,
        'ligand_sizes': ligand_sizes,
        'pocket_sizes': pocket_sizes
    }


def create_random_training_data(config: Dict) -> List[Dict]:
    """Generate training dataset"""
    
    print(f"Generating {config['num_train_batches']} training batches...")
    
    train_data = []
    for _ in range(config['num_train_batches']):
        batch = create_fake_molecular_batch(
            batch_size=config['batch_size'],
            ligand_size_range=config['ligand_size_range'],
            pocket_size_range=config['pocket_size_range'],
            atom_nf=config['atom_nf'],
            residue_nf=config['residue_nf']
        )
        train_data.append(batch)
    
    print(f"✅ Generated {len(train_data)} training batches")
    return train_data


def create_random_eval_data(config: Dict) -> List[Dict]:
    """Generate evaluation dataset"""
    
    print(f"Generating {config['num_eval_batches']} evaluation batches...")
    
    eval_data = []
    for _ in range(config['num_eval_batches']):
        batch = create_fake_molecular_batch(
            batch_size=config['batch_size'],
            ligand_size_range=config['ligand_size_range'],
            pocket_size_range=config['pocket_size_range'],
            atom_nf=config['atom_nf'],
            residue_nf=config['residue_nf']
        )
        eval_data.append(batch)
    
    print(f"✅ Generated {len(eval_data)} evaluation batches")
    return eval_data


def create_batches_from_dataset(dataset_path: str, config: Dict) -> List[Dict]:
    """Create dataset from PDBbind dataset"""
    data = []

    dataset = torch.load(dataset_path)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, follow_batch=['lig_coords', 'prot_coords'])

    for batch in dataloader:
        batch = {
            'ligand_coords': batch.lig_coords,
            'ligand_features': batch.lig_features,
            'ligand_mask': batch.lig_coords_batch,

            'pocket_coords': batch.prot_coords,
            'pocket_features': batch.prot_features,
            'pocket_mask': batch.prot_coords_batch,
            'batch_size': config['batch_size'],
        }
        data.append(batch)

    print(f"✅ Created {len(data)} batches from PDBbind dataset")
    return data
    
    

def create_molecular_model(config: Dict) -> MolecularDenoisingModel:
    """Create molecular model following the GenCFD setup"""
    
    print("Creating molecular diffusion model...")
    
    # Create noise schedule
    sigma_schedule = exponential_noise_schedule(
        clip_max=config['sigma_max'],
        base=np.e**0.5,
        start=0.0,
        end=5.0
    )
    
    # Create diffusion scheme 
    scheme = MolecularDiffusion.create_variance_exploding(
        sigma=sigma_schedule,
        coord_norm=10.0, # TODO: check if okay.
        feature_norm=1.0,
        feature_bias=0.0
    )
    
    # Create noise sampling and weighting
    noise_sampling = log_uniform_sampling(
        scheme=scheme,
        clip_min=config['sigma_min'],
        uniform_grid=True,
    )
    
    noise_weighting = edm_weighting(data_std=1.0)
    
    # Create model
    model = MolecularDenoisingModel(
        atom_nf=config['atom_nf'],
        residue_nf=config['residue_nf'],
        n_dims=config['n_dims'],
        joint_nf=config['joint_nf'],
        hidden_nf=config['hidden_nf'],
        n_layers=config['n_layers'],
        edge_embedding_dim=config['edge_embedding_dim'],
        update_pocket_coords=config['update_pocket_coords'],
        scheme=scheme,
        noise_sampling=noise_sampling,
        noise_weighting=noise_weighting
    )
    
    model.initialize()
    return model


def train_model(model: MolecularDenoisingModel, train_data: List[Dict], 
                eval_data: List[Dict], config: Dict):
    """Train the molecular diffusion model"""
    
    print(f"\n{'='*60}")
    print("TRAINING MOLECULAR DIFFUSION MODEL")
    print(f"{'='*60}")

    # samples = sample_molecules(model,
    #                                 num_steps=CONFIG['num_sampling_steps'],
    #                                 schedule_type=CONFIG['schedule_type'],
    #                                 num_samples=CONFIG['num_eval_samples']
    #                                 )
    # sampling_losses = evaluate_atom_aa_distributions(samples)
    # print(sampling_losses)
    # import sys
    # sys.exit(0)
    
    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        name=config['wandb']['name'],
        tags=config['wandb']['tags'],
        notes=config['wandb']['notes'],
        config=config
    )

    # Setup optimizer
    optimizer = optim.AdamW(
        model.denoiser.parameters(),
        lr=config['learning_rate'],
    )
    
    # Watch model gradients if configured
    if config['wandb']['log_gradients']:
        wandb.watch(model.denoiser, log='all', log_freq=config['log_interval'])
    
    # Training loop
    model.denoiser.train()
    
    for epoch in range(config['num_epochs']):
        epoch_losses = []
        epoch_metrics = {
            'coord_loss': [],
            'categorical_loss': [],
            'coord_loss_ligand': [],
            'coord_loss_pocket': [],
            'categorical_loss_ligand': [],
            'categorical_loss_pocket': [],
            'avg_sigma': []
        }
        
        for batch_idx, batch in enumerate(train_data):
            optimizer.zero_grad()
            
            # Forward pass 
            loss, metrics = model.loss_fn(batch)
            
            # Backward pass 
            loss.backward()
            
            # Clip gradients 
            #torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_losses.append(loss.item())
            
            # Collect metrics
            for key in epoch_metrics.keys():
                epoch_metrics[key].append(metrics[key])
            
            # Logging 
            if (batch_idx + 1) % config['log_interval'] == 0:
                print(f"Epoch {epoch+1:3d}/{config['num_epochs']}, "
                      f"Batch {batch_idx+1:3d}/{len(train_data)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Coord: {metrics['coord_loss']:.4f} "
                      f"(L:{metrics['coord_loss_ligand']:.4f}, P:{metrics['coord_loss_pocket']:.4f}), "
                      f"Cat: {metrics['categorical_loss']:.4f} "
                      f"(L:{metrics['categorical_loss_ligand']:.4f}, P:{metrics['categorical_loss_pocket']:.4f}), "
                      f"σ: {metrics['avg_sigma']:.3f}, "
                      # f"s: {metrics['avg_scale']:.3f}"
                      )
                
                # Log batch metrics to wandb
                wandb.log({
                    'batch/loss': loss.item(),
                    'batch/coord_loss': metrics['coord_loss'],
                    'batch/categorical_loss': metrics['categorical_loss'],
                    'batch/coord_loss_ligand': metrics['coord_loss_ligand'],
                    'batch/coord_loss_pocket': metrics['coord_loss_pocket'],
                    'batch/categorical_loss_ligand': metrics['categorical_loss_ligand'],
                    'batch/categorical_loss_pocket': metrics['categorical_loss_pocket'],
                    'batch/avg_sigma': metrics['avg_sigma'],
                    'batch/learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch': epoch + 1,
                    'batch': batch_idx + 1
                })
        
        # Calculate epoch metrics
        avg_epoch_loss = np.mean(epoch_losses)
        avg_epoch_metrics = {
            key: np.mean(values) for key, values in epoch_metrics.items()
        }
        
        # Log epoch metrics
        wandb.log({
            'epoch/loss': avg_epoch_loss,
            'epoch/coord_loss': avg_epoch_metrics['coord_loss'],
            'epoch/categorical_loss': avg_epoch_metrics['categorical_loss'],
            'epoch/coord_loss_ligand': avg_epoch_metrics['coord_loss_ligand'],
            'epoch/coord_loss_pocket': avg_epoch_metrics['coord_loss_pocket'],
            'epoch/categorical_loss_ligand': avg_epoch_metrics['categorical_loss_ligand'],
            'epoch/categorical_loss_pocket': avg_epoch_metrics['categorical_loss_pocket'],
            'epoch/avg_sigma': avg_epoch_metrics['avg_sigma'],
            #'epoch/avg_scale': avg_epoch_metrics['avg_scale'],
            'epoch/learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': epoch + 1
        })
        
        # Evaluation 
        if (epoch + 1) % config['eval_interval'] == 0 or epoch == 0:
            print(f"\n--- Evaluation at epoch {epoch+1} ---")
            eval_losses = evaluate_model(model, eval_data)

            # Sampling-based losses
            save_path = CONFIG['checkpoint_path'].replace(".pt", f"_epoch_{epoch}.pt")
            save_checkpoint(model, CONFIG, save_path)
            loaded_model = load_checkpoint(save_path)
            samples = sample_molecules(loaded_model,
                                    num_steps=CONFIG['num_sampling_steps'],
                                    schedule_type=CONFIG['schedule_type'],
                                    num_samples=CONFIG['num_eval_samples']
                                    )
            distribution_losses = evaluate_atom_aa_distributions(samples)
            eval_losses.update(distribution_losses)
            print(f"Eval losses: {eval_losses}", flush=True)
            
            # Log evaluation metrics
            wandb.log({
                f'eval/{key}': value 
                for key, value in eval_losses.items()
            })
            
        print(f"Epoch {epoch+1:3d} - Average Loss: {avg_epoch_loss:.4f}")
    
    # Save model to wandb if configured
    if config['wandb']['log_model']:
        model_artifact = wandb.Artifact(
            'molecular_diffusion_model', 
            type='model',
            description='Trained molecular diffusion model'
        )
        model_artifact.add_file(config['checkpoint_path'])
        wandb.log_artifact(model_artifact)
    
    # Close wandb run
    wandb.finish()
    
    print(f"\n✅ Training completed!")


def evaluate_model(model: MolecularDenoisingModel, eval_data: List[Dict]) -> Dict:
    """Evaluate the model following """
    
    model.denoiser.eval()
    all_eval_metrics = {}
    
    with torch.no_grad():
        for batch in eval_data[:5]:  # Evaluate on subset
            eval_metrics = model.eval_fn(batch)
            
            for key, value in eval_metrics.items():
                if key not in all_eval_metrics:
                    all_eval_metrics[key] = []
                all_eval_metrics[key].append(value)
    
    # Average metrics
    averaged_metrics = {
        key: np.mean(values) for key, values in all_eval_metrics.items()
    }
    
    model.denoiser.train()
    return averaged_metrics


def save_checkpoint(model: MolecularDenoisingModel, config: Dict, save_path: None):
    """Save model checkpoint"""
    
    checkpoint = {
        'model_state_dict': model.denoiser.state_dict(),
        'config': config,
        'model_params': {
            'atom_nf': model.atom_nf,
            'residue_nf': model.residue_nf,
            'n_dims': model.n_dims,
            'joint_nf': model.joint_nf,
            'hidden_nf': model.hidden_nf,
            'n_layers': model.n_layers,
            'edge_embedding_dim': model.edge_embedding_dim,
            'update_pocket_coords': model.update_pocket_coords,
            # Save scheme parameters instead of the scheme itself
            'scheme_params': {
                'coord_norm': model.scheme.coord_norm,
                'feature_norm': model.scheme.feature_norm,
                'feature_bias': model.scheme.feature_bias,
                'sigma_max': config['sigma_max'],
                'sigma_min': config['sigma_min']
            }
        }
    }
    if save_path is None:
        torch.save(checkpoint, config['checkpoint_path'])
        print(f"✅ Checkpoint saved to {config['checkpoint_path']}")
    else:
        torch.save(checkpoint, save_path)
        print(f"✅ Checkpoint saved to {save_path}")



def analyze_samples(samples: Dict, config: Dict):
    """Analyze the generated molecular samples"""
    
    print(f"\n{'='*60}")
    print("SAMPLE ANALYSIS")
    print(f"{'='*60}")
    
    # Basic statistics 
    print("Coordinate Statistics:")
    print(f"  Ligand coords - Mean: {samples['ligand_coords'].mean(dim=0)}")
    print(f"  Ligand coords - Std:  {samples['ligand_coords'].std(dim=0)}")
    print(f"  Pocket coords - Mean: {samples['pocket_coords'].mean(dim=0)}")
    print(f"  Pocket coords - Std:  {samples['pocket_coords'].std(dim=0)}")
    
    # Check center of mass (should be near zero)
    total_coords = torch.cat([samples['ligand_coords'], samples['pocket_coords']], dim=0)
    total_mask = torch.cat([samples['ligand_mask'], samples['pocket_mask']], dim=0)
    
    # Calculate COM for each molecule
    batch_size = samples['batch_size']
    for i in range(batch_size):
        mol_coords = total_coords[total_mask == i]
        com = mol_coords.mean(dim=0)
        print(f"  Molecule {i} center of mass: {com}")
    
    # Check molecular geometry reasonableness
    print(f"\nMolecular Geometry Analysis:")
    for i in range(batch_size):
        lig_coords = samples['ligand_coords'][samples['ligand_mask'] == i]
        pocket_coords = samples['pocket_coords'][samples['pocket_mask'] == i]
        
        if len(lig_coords) > 1:
            lig_distances = torch.cdist(lig_coords, lig_coords)
            lig_distances = lig_distances[lig_distances > 0]  # Remove self-distances
            if len(lig_distances) > 0:  # Only calculate stats if we have distances
                print(f"  Molecule {i} ligand bond lengths - Mean: {lig_distances.mean():.3f}, Min: {lig_distances.min():.3f}")
            else:
                print(f"  Molecule {i} ligand bond lengths - No valid distances found")
        
        if len(pocket_coords) > 1:
            pocket_distances = torch.cdist(pocket_coords, pocket_coords)
            pocket_distances = pocket_distances[pocket_distances > 0]
            if len(pocket_distances) > 0:  # Only calculate stats if we have distances
                print(f"  Molecule {i} pocket distances - Mean: {pocket_distances.mean():.3f}, Min: {pocket_distances.min():.3f}")
            else:
                print(f"  Molecule {i} pocket distances - No valid distances found")
    
    print(f"\n✅ Sample analysis completed!")


def main():
    """Main pipeline demonstrating the complete molecular diffusion workflow"""
    
    print(f"🧬 MOLECULAR DIFFUSION PIPELINE")
    print(f"Device: {CONFIG['device']}")
    print(f"Configuration: {CONFIG}")
    
    try:
        # 1. Generate training and evaluation data
        print(f"\n{'='*60}")
        print("GENERATING TRAINING DATA")
        print(f"{'='*60}")
        
        train_data = create_batches_from_dataset(CONFIG['train_dataset_path'], CONFIG)
        eval_data = create_batches_from_dataset(CONFIG['eval_dataset_path'], CONFIG)
        
        # Show example batch 
        example_batch = train_data[0]
        print(f"\nExample batch:")
        print(f"  Ligand atoms: {example_batch['ligand_coords'].shape[0]}")
        print(f"  Pocket residues: {example_batch['pocket_coords'].shape[0]}")
        
        # 2. Create and initialize model
        print(f"\n{'='*60}")
        print("CREATING MODEL")
        print(f"{'='*60}")
        
        model = create_molecular_model(CONFIG)
        
        # 3. Train the model
        train_model(model, train_data, eval_data, CONFIG)
        
        # 4. Save checkpoint
        save_checkpoint(model, CONFIG, save_path=CONFIG['checkpoint_path'].replace(".pt", "_final.pt"))
        
        # 5. Load checkpoint to demonstrate persistence
        print(f"\n{'='*60}")
        print("LOADING MODEL FROM CHECKPOINT")
        print(f"{'='*60}")
        
        loaded_model = load_checkpoint(CONFIG['checkpoint_path'])
        
        # 6. Sample new molecules
        samples = sample_molecules(
            loaded_model,
            num_steps=CONFIG['num_sampling_steps'],
            schedule_type=CONFIG['schedule_type'],
            num_samples=CONFIG['num_eval_samples'],
        )

        for s in range(CONFIG['num_eval_samples']):
            graph = Data(
                ligand_coords=samples['ligand_coords'][samples['ligand_mask']==s].cpu(),
                ligand_features=samples['ligand_features'][samples['ligand_mask']==s].cpu(),
                pocket_coords=samples['pocket_coords'][samples['pocket_mask']==s].cpu(),
                pocket_features=samples['pocket_features'][samples['pocket_mask']==s].cpu()
            )
            torch.save(graph, CONFIG['run_dir'] + f"/sample_{s}.pt")
    
        
        # 7. Analyze samples
        analyze_samples(samples, CONFIG)
        
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Checkpoint saved at: {CONFIG['checkpoint_path']}")
        print(f"Used {torch.cuda.get_device_name()} for computation")
        
    except Exception as e:
        # Log error to wandb if it's initialized
        if wandb.run is not None:
            wandb.run.finish(exit_code=1)
        raise e


if __name__ == "__main__":
    main()
