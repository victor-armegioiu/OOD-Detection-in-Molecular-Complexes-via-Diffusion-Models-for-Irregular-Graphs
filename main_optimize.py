"""
Main Molecular Diffusion Pipeline

Complete demonstration of training, saving, loading, and sampling with molecular diffusion models.
All operations are CUDA-optimized.
"""

import torch
import time
import warnings
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import wandb
import argparse
import itertools
import json
import csv
import random
from typing import List, Dict, Tuple, Optional
from moldiff.Dataset import PDBbind_Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from datetime import datetime
import logging
import sys

# sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path

# Import our molecular modules
from moldiff.molecular_diffusion import (
    MolecularDenoisingModel, 
    ConditionalMolecularDenoisingModel,
    exponential_noise_schedule,
    log_uniform_sampling,
    edm_weighting,
    MolecularDiffusion
)
from moldiff.metrics import (
    load_checkpoint,
    sample_molecules, 
    sample_molecules_conditionally,
    evaluate_atom_aa_distributions,
    evaluate_mols,
    build_mol_objects
)

# Try to import optuna for Bayesian optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not available. Bayesian optimization will be disabled.")

DATA_PATH = Path("datasets")


CONFIG = {
    'train_dataset_path': DATA_PATH / 'dataset_cleansplit_train.pt',
    'eval_dataset_path': DATA_PATH / 'dataset_cleansplit_validation.pt',

    # Model parameters
    'atom_nf': 10,           
    'residue_nf': 21,         
    'n_dims': 3,             
    'n_layers': 4, # 4           
    'joint_nf': 256,          
    'hidden_nf': 128, # 256         
    'edge_embedding_dim': 32, # 64 
    
    # Training parameters
    'num_epochs': 500,
    'learning_rate': 1e-4,
    'batch_size': 16,
    'log_interval': 20,
    'eval_interval': 10,
    'num_eval_samples': 50,
    'early_stopping_patience': 50,
    'cfg_training': False,
    'cfg_p_uncond': 0.2,
    'cfg_guidance_scale': 3,
    
    # Diffusion parameters
    'sigma_max': 100.0,      # Maximum noise level
    'sigma_min': 1e-4,      # Minimum noise level
    'update_pocket_coords': True,  # Joint modeling
    'freeze_pocket_embeddings': False, # No CE loss on residue classes
    'geometric_regularization': False,
    'geom_loss_weight': 0.0,
    
    # Sampling parameters
    'num_sampling_steps': 400,
    'schedule_type': "exponential",
    
    # I/O
    'device': torch.device("cuda"), # if torch.cuda.is_available() else "cpu"),
    
    # Scheduler, scaler, and seed
    'use_scheduler': False,
    'use_amp': False,  # Automatic Mixed Precision training
    'seed': 42,  # Random seed for reproducibility

    # Weights & Biases configuration
    'wandb': {
        'project': 'equivariant-diffusion',
        'entity': 'paertschi-eth',
        'tags': ['molecular-diffusion', 'training'],
        'notes': 'Training run for molecular diffusion model',
        'log_model': False,
        'log_gradients': False,  # Set to True to log gradient distributions
    }
}

# Hyperparameter search spaces
HYPERPARAM_SPACES = {
    'num_sampling_steps': [200, 400], # [50, 100, 200, 300, 400],
    'joint_nf': [128, 256], #[32, 64, 128, 256],
    'hidden_nf': [64, 128], 
    'n_layers': [4, 5],
    'edge_embedding_dim': [32, 64],
    'learning_rate': [1e-4, 5e-4],
    'batch_size': [16, 32]
}

def set_random_seeds(seed: int):
    """Set random seeds for all random number generators for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"✅ Random seeds set to {seed}")

# helpers to measure time and potentially log as warning if using flag --use_warnings from bash
def log(msg: str, use_warnings: bool = False, flush: bool = True):
    """Log message to stdout or warnings depending on flag"""
    if use_warnings:
        warnings.warn(msg, RuntimeWarning)
    else:
        print(msg, flush=flush)

def log_time(msg: str, start: float, end: float, use_warnings: bool = False):
    """Log timing info with chosen method"""
    log(f"{msg} completed in {end - start:.2f} seconds", use_warnings)

def parse_arguments():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(description="Molecular Diffusion Pipeline with Hyperparameter Optimization")
    
    # Basic run options
    parser.add_argument('--mode', choices=['single', 'grid', 'random', 'bayesian'], default='single',
        help='Run mode: single training run or hyperparameter optimization')
    
    # Dataset options
    parser.add_argument('--train_dataset', default=DATA_PATH / 'dataset_cleansplit_train.pt', help='Path to training dataset')
    parser.add_argument('--eval_dataset', default=DATA_PATH / 'dataset_cleansplit_validation.pt', help='Path to evaluation dataset')
    
    # Model parameters (for single mode or to override defaults)
    parser.add_argument('--freeze_pocket_coords', action='store_false', help='Enables conditional model')
    parser.add_argument('--freeze_pocket_embeddings', action='store_true', help='Freezes pocket encoder weights')
    parser.add_argument('--joint_nf', type=int, help='Joint embedding dimension')
    parser.add_argument('--hidden_nf', type=int, help='Hidden layer size')
    parser.add_argument('--n_layers', type=int, help='Number of EGNN layers')
    parser.add_argument('--edge_embedding_dim', type=int, help='Edge feature dimension')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--eval_interval', type=int, help='Interval for evaluation')
    parser.add_argument('--early_stopping_patience', type=int, help='Patience for early stopping')
    parser.add_argument('--use_scheduler', action='store_true', help='If the predefined learning rate scheduler should be used')
    parser.add_argument('--cfg_training', action='store_true', help='Enables classifier-free training')
    parser.add_argument('--cfg_p_uncond', type = float, help='Fraction of null residue training rounds')
    parser.add_argument('--cfg_guidance_scale', type = float, help='Guidance scale in CFG sampling procedure')
    
    # Sampling parameters
    parser.add_argument('--num_sampling_steps', type=int, help='Number of sampling steps')
    
    # Optimization parameters
    parser.add_argument('--max_combinations', type=int, default=100, help='Maximum number of combinations for grid search')
    parser.add_argument('--num_trials', type=int, default=20, help='Number of trials for random/bayesian search')
    parser.add_argument('--study_name', default=None, help='Optuna study name (for resuming Bayesian optimization)')
    parser.add_argument('--storage', default=None, help='Optuna storage URI (e.g. sqlite:///example-study.db)')
    
    # Output options
    parser.add_argument('--output_dir', default='optimization_results',help='Directory to save optimization results')
    parser.add_argument('--results_file', default='optimization_results.csv', help='Filename for optimization results (CSV format)')
    
    # Wandb options
    parser.add_argument('--wandb_project', default='equivariant-diffusion', help='Wandb project name')
    parser.add_argument('--wandb_entity', default='paertschi-eth', help='Wandb entity name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    
    # Reproducibility options
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Checkpoint options
    parser.add_argument('--resume_checkpoint_path', type=str, default=None, help='Path to checkpoint file to resume training from')
    parser.add_argument('--resume_epoch', type=int, default=None, help='Epoch to resume training from (if not specified, will be inferred from checkpoint)')

    # Debugging
    parser.add_argument('--use_warnings', action='store_true', help='Enable warnings for debugging')
    
    return parser.parse_args()


def update_config_from_args(config: Dict, args) -> Dict:
    """Update configuration with command line arguments"""
    
    # Update dataset paths
    if args.train_dataset is not None:
        config['train_dataset_path'] = args.train_dataset
    if args.eval_dataset is not None:
        config['eval_dataset_path'] = args.eval_dataset
    
    # Update model parameters if provided
    # config['update_pocket_coords'] = args.update_pocket_coords or config['update_pocket_coords'] # replaces all falsy values, e.g. also zero
    if args.freeze_pocket_coords is not None: # just replaces if not None
        config['update_pocket_coords'] = args.freeze_pocket_coords # is stored as false if flag is present
    if args.freeze_pocket_embeddings is not None:
        config['freeze_pocket_embeddings'] = args.freeze_pocket_embeddings
    if args.joint_nf is not None:
        config['joint_nf'] = args.joint_nf
    if args.hidden_nf is not None:
        config['hidden_nf'] = args.hidden_nf
    if args.n_layers is not None:
        config['n_layers'] = args.n_layers
    if args.edge_embedding_dim is not None:
        config['edge_embedding_dim'] = args.edge_embedding_dim
    
    # Update training parameters
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    if args.eval_interval is not None:
        config['eval_interval'] = args.eval_interval
    if args.early_stopping_patience is not None:
        config['early_stopping_patience'] = args.early_stopping_patience
    if args.use_scheduler is not None:
        config['use_scheduler'] = args.use_scheduler
    if args.cfg_training is not None:
        config['cfg_training'] = args.cfg_training
    if args.cfg_p_uncond is not None:
        config['cfg_p_uncond'] = args.cfg_p_uncond
    if args.cfg_guidance_scale is not None:
        config['cfg_guidance_scale'] = args.cfg_guidance_scale
    
    # Update sampling parameters
    if args.num_sampling_steps is not None:
        config['num_sampling_steps'] = args.num_sampling_steps
    
    # Update wandb configuration
    config['wandb']['project'] = args.wandb_project
    config['wandb']['entity'] = args.wandb_entity
    if args.no_wandb:
        config['wandb']['enabled'] = False
    
    # Add seed to config
    if args.seed is not None:
        config['seed'] = args.seed
    
    # Add checkpoint options to config
    if args.resume_checkpoint_path is not None:
        config['resume_checkpoint_path'] = args.resume_checkpoint_path
    if args.resume_epoch is not None:
        config['resume_epoch'] = args.resume_epoch
    
    return config


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

            'batch_size': config['batch_size']
            # track COM for later addition in conditional sampling
            # 'pocket_com': com #torch.cat([batch.lig_coords, batch.prot_coords], dim=0).mean(dim=0, keepdim=True)
        }
        data.append(batch)

    log(f"✅ Created {len(data)} batches from PDBbind dataset")
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
        uniform_grid=True
    )
    noise_weighting = edm_weighting(data_std=1.0)
    
    # Create model (joint if config['update_pocket_coords'] is enabled else conditional)
    model = MolecularDenoisingModel(
        atom_nf=config['atom_nf'],
        residue_nf=config['residue_nf'],
        n_dims=config['n_dims'],
        joint_nf=config['joint_nf'],
        hidden_nf=config['hidden_nf'],
        n_layers=config['n_layers'],
        edge_embedding_dim=config['edge_embedding_dim'],
        update_pocket_coords=config['update_pocket_coords'],
        geometric_regularization=config['geometric_regularization'],
        geom_loss_weight=config['geom_loss_weight'],
        scheme=scheme,
        noise_sampling=noise_sampling,
        noise_weighting=noise_weighting, 
        device=config["device"]
    ) if config['update_pocket_coords'] else ConditionalMolecularDenoisingModel(
        atom_nf=config['atom_nf'],
        residue_nf=config['residue_nf'],
        n_dims=config['n_dims'],
        joint_nf=config['joint_nf'],
        hidden_nf=config['hidden_nf'],
        n_layers=config['n_layers'],
        edge_embedding_dim=config['edge_embedding_dim'],
        update_pocket_coords=config['update_pocket_coords'],
        geometric_regularization=config['geometric_regularization'],
        geom_loss_weight=config['geom_loss_weight'],
        scheme=scheme,
        noise_sampling=noise_sampling,
        noise_weighting=noise_weighting, 
        device=config["device"]
    )
    
    model.initialize()
    return model


def train_model(model: MolecularDenoisingModel | ConditionalMolecularDenoisingModel, train_data: List[Dict], 
                eval_data: List[Dict], config: Dict):
    """Train the molecular diffusion model"""

    # check for CFG training in joint mode (not permitted)
    if config["cfg_training"] and config["update_pocket_coords"]:
        raise RuntimeError("Classifier-free guidance training not possible for joint ligand-pocket modelling: " \
        "\n config.update_pocket_coords cannot be True if config.cfg_training is True")
    
    log(f"\n{'='*60}")
    log("TRAINING MOLECULAR DIFFUSION MODEL")
    log(f"{'='*60}")
    
    log(config['wandb']['project'], config['wandb']['entity'])
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
        lr=config['learning_rate'])

    # Setup scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8) if config.get('use_scheduler', False) else None
    
    # Setup scaler for mixed precision training (optional)
    scaler = torch.cuda.amp.GradScaler() if config.get('use_amp', False) else None
    

    # Load from checkpoint if specified
    # --------------------------------------------------------------------------------------
    start_epoch = 0
    best_metrics = None
    if 'resume_checkpoint_path' in config and config['resume_checkpoint_path'] is not None:
        model, checkpoint_config, optimizer_state, scheduler_state, scaler_state, checkpoint_epoch, checkpoint_best_metrics = load_checkpoint_for_resume(config['resume_checkpoint_path'], config)
        
        # Load optimizer state if available
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            print("✅ Optimizer state loaded")
        
        # Load scheduler state if available
        if scheduler_state is not None and scheduler is not None:
            scheduler.load_state_dict(scheduler_state)
            print("✅ Scheduler state loaded")
        
        # Load scaler state if available
        if scaler_state is not None and scaler is not None:
            scaler.load_state_dict(scaler_state)
            print("✅ Scaler state loaded")

        # Load best metrics if available
        if checkpoint_best_metrics is not None:
            best_metrics = checkpoint_best_metrics
            print("✅ Best metrics loaded from checkpoint")

        # Determine starting epoch
        if 'resume_epoch' in config and config['resume_epoch'] is not None:
            start_epoch = config['resume_epoch']
            print(f"Resuming from specified epoch: {start_epoch}")
        elif checkpoint_epoch is not None:
            start_epoch = checkpoint_epoch + 1  # Resume from next epoch
            print(f"Resuming from checkpoint epoch: {checkpoint_epoch}, starting at epoch: {start_epoch}")
        else:
            print("No epoch information found in checkpoint, starting from epoch 0")
    # --------------------------------------------------------------------------------------
    

    # Early stopping setup
    # ----------------------------------------------------------------------------------------------
    # Early stopping metrics (k) with worst possible value (w) and direction (d) of improvement
    early_stop_metrics = [
        ('percent_fragmented', 1.0, '-'),
        ('mean_num_fragments', float('inf'), '-'),
        ('mean_ring_size', 0.0, '+')
        ]

    # Initialize best metrics - use loaded metrics if available, otherwise use worst possible values
    if best_metrics is None:
        best_metrics = {k: w for k, w, _ in early_stop_metrics}
    else:
        # Merge loaded best metrics with early stopping metrics
        for k, w, _ in early_stop_metrics:
            if k not in best_metrics:
                best_metrics[k] = w
    
    best_epoch = {k: -1 for k, _, _ in early_stop_metrics}
    patience_counter = 0
    early_stopped = False
    eval_history = []  # For debugging/logging
    epoch_times = []

    best_epoch_metrics = None
    best_epoch_idx = -1
    # ----------------------------------------------------------------------------------------------

    # Watch model gradients if configured
    if config['wandb']['log_gradients']:
        wandb.watch(model.denoiser, log='all', log_freq=config['log_interval'])

    # ----------------------------------------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------------------------------------
    model.denoiser.train()
    for epoch in range(start_epoch, config['num_epochs']):
        epoch_losses = []
        epoch_metrics = {
            'coord_loss': [],
            'categorical_loss': [],
            'coord_loss_ligand': [],
            'coord_loss_pocket': [],
            'categorical_loss_ligand': [],
            'categorical_loss_pocket': [],
            'geometric_loss_total': [],
            'avg_sigma': []
        }
        start = time.perf_counter()
        for batch_idx, batch in enumerate(train_data):

            optimizer.zero_grad()

            # Forward pass
            if config["cfg_training"] and torch.rand(()) < config["cfg_p_uncond"]:
                # unconditional loss
                loss, metrics = model.null_residue_loss_fn(batch)
            else:
                # traditional loss
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
                log(f"Epoch {epoch+1:3d}/{config['num_epochs']}, "
                      f"Batch {batch_idx+1:3d}/{len(train_data)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Coord: {metrics['coord_loss']:.4f} "
                      f"(L:{metrics['coord_loss_ligand']:.4f}, P:{metrics['coord_loss_pocket']:.4f}), "
                      f"Cat: {metrics['categorical_loss']:.4f} "
                      f"(L:{metrics['categorical_loss_ligand']:.4f}, P:{metrics['categorical_loss_pocket']:.4f}), "
                      f"σ: {metrics['avg_sigma']:.3f}, "
                      f"Geom: {metrics['geometric_loss_total']:.4f}"
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
                    'batch/geometric_loss_total': metrics['geometric_loss_total'],
                    'batch/avg_sigma': metrics['avg_sigma'],
                    'batch/learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch': epoch + 1,
                    'batch': batch_idx + 1
                })
        
        if scheduler is not None:
            scheduler.step()
        
        # Calculate epoch metrics
        avg_epoch_loss = np.mean(epoch_losses)
        avg_epoch_metrics = {
            key: np.mean(values) for key, values in epoch_metrics.items()
        }

        # Log epoch metrics
        epoch_metrics = {
            'epoch/loss': avg_epoch_loss,
            'epoch/coord_loss': avg_epoch_metrics['coord_loss'],
            'epoch/categorical_loss': avg_epoch_metrics['categorical_loss'],
            'epoch/coord_loss_ligand': avg_epoch_metrics['coord_loss_ligand'],
            'epoch/coord_loss_pocket': avg_epoch_metrics['coord_loss_pocket'],
            'epoch/categorical_loss_ligand': avg_epoch_metrics['categorical_loss_ligand'],
            'epoch/categorical_loss_pocket': avg_epoch_metrics['categorical_loss_pocket'],
            'epoch/geometric_loss_total': avg_epoch_metrics['geometric_loss_total'],
            'epoch/avg_sigma': avg_epoch_metrics['avg_sigma'],
            'epoch/learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': epoch + 1
        }
        wandb.log(epoch_metrics)


        
        # Evaluation 
        if (epoch + 1) % config['eval_interval'] == 0 or epoch == 0:
            log(f"\n--- Evaluation at epoch {epoch+1} ---")
            eval_losses = evaluate_model(model, eval_data)

            # Sampling-based losses
            save_path = config['checkpoint_path'].replace(".pt", f"_epoch_{epoch + 1}.pt")
            save_checkpoint(model, config, save_path, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, best_metrics=best_metrics)
            loaded_model = load_checkpoint(save_path)

            if not config["update_pocket_coords"]:
                sample_loader = eval_data[0] # TODO should be replaced by a test dataset
                samples = sample_molecules_conditionally(
                    loaded_model, 
                    sample_loader,
                    num_steps = config['num_sampling_steps'], 
                    schedule_type=config['schedule_type'], 
                    guidance_scale = config["cfg_guidance_scale"] # attach guidance scale
                )
                print("Conditional sampling used in training!")
            else:
                samples = sample_molecules(
                    loaded_model,
                    num_steps=config['num_sampling_steps'],
                    schedule_type=config['schedule_type'],
                    num_samples=config['num_eval_samples']
                    )
            distribution_losses = evaluate_atom_aa_distributions(samples)
            eval_losses.update(distribution_losses)

            # Evaluate molecule integrity
            molecules = build_mol_objects(samples)
            integrity_data = evaluate_mols(molecules)

            if integrity_data['sum_num_rings'] > 0:
                mean_ring_size = sum(ring_size * count for ring_size, count in 
                                    integrity_data['ring_size_counts'].items()) / integrity_data['sum_num_rings']
            else:
                mean_ring_size = 0

            integrity_losses = {
                'percent_valid': integrity_data['valid_molecules'] / integrity_data['num_molecules'],
                'percent_fragmented': integrity_data['num_multifragment'] / integrity_data['num_molecules'],
                'percent_disconnected': integrity_data['num_disconnected_atoms'] / integrity_data['num_molecules'],
                'percent_valence_issues': integrity_data['sum_num_valence_issues'] / integrity_data['num_molecules'],
                'n_rings_per_mol': integrity_data['sum_num_rings'] / integrity_data['num_molecules'],
                'mean_num_rings': integrity_data['mean_num_rings'],
                'mean_ring_size': mean_ring_size,
                'mean_num_fragments': integrity_data['mean_num_fragments'],
                'mean_num_valence_issues': integrity_data['mean_num_valence_issues'],
                'ring_size_dist_kl_divergence': integrity_data['ring_size_dist_kl_divergence'],
                'ring_size_dist_js_divergence': integrity_data['ring_size_dist_js_divergence'],
            }

            eval_losses.update(integrity_losses)
            log("Eval losses:")
            for key, value in eval_losses.items(): log(f'{key}: {value}')
            
            # Log evaluation metrics
            eval_log = {f'eval/{key}': value for key, value in eval_losses.items()}
            eval_log['eval_epoch'] = epoch + 1
            wandb.log(eval_log)


            # Early stopping logic
            # ------------------------------------------------------------------------------------
            print(f"\n ----- Epoch {epoch+1}: Early Stopping Logic: -----")
            print(f"Early stopping metrics: {[k for k, _, _ in early_stop_metrics]}")
            eval_history.append({k: eval_losses.get(k, w) for k, w, _ in early_stop_metrics})

            improved = False
            for k, w, d in early_stop_metrics:
                v = eval_losses.get(k, w) # Fallback to worst possible value if metric not found
                if v < best_metrics[k] and d == '-':
                    print(f"    Improved {k} from {best_metrics[k]} to {v}")
                    best_metrics[k] = v
                    best_epoch[k] = epoch
                    improved = True
                elif v > best_metrics[k] and d == '+':
                    print(f"    Improved {k} from {best_metrics[k]} to {v}")
                    best_metrics[k] = v
                    best_epoch[k] = epoch
                    improved = True
                else:
                    print(f"    No improvement in {k} from {best_metrics[k]} to {v}")

            if improved:
                patience_counter = 0
                best_epoch_metrics = {**eval_losses, **epoch_metrics}
                best_epoch_idx = epoch
                # log("[Early Stopping] Early Stopping Patience Reset. Model improved on metrics:")
                # for k in early_stop_metrics:
                #     log(f"    {k}: {best_metrics[k]}")
            else:
                patience_counter += 1
                log(f"[Early Stopping] Patience Counter: {patience_counter}")

            if patience_counter >= 10:
                log(f"\n[EARLY STOPPING] No improvement in {early_stop_metrics} for two evaluation intervals ({config['eval_interval']} epochs). Stopping at epoch {epoch+1}.")
                early_stopped = True
                break
            # ------------------------------------------------------------------------------------
        
        epoch_times.append(time.perf_counter() - start)
        log(f"Epoch {epoch+1:3d} - Average Loss: {avg_epoch_loss:.4f} - Time: {epoch_times[-1]:.2f} seconds (At this speed, {3600 / epoch_times[-1]:.2f} epochs/hour)", flush=True)

    # Save model to wandb if configured
    if config['wandb']['log_model']:
        model_artifact = wandb.Artifact(
            'molecular_diffusion_model', 
            type='model',
            description='Trained molecular diffusion model'
        )
        model_artifact.add_file(config['checkpoint_path'])
        wandb.log_artifact(model_artifact)
    
    print("\n" + 60*"*")
    print(f" Avg Epoch Time: {np.mean(epoch_times):.2f} seconds (At this speed, {3600 / np.mean(epoch_times):.2f} epochs/hour)")
    print(60*"*")
    # Close wandb run
    wandb.finish()
    return best_epoch_metrics, early_stopped, optimizer, scheduler, scaler, epoch


def evaluate_model(model: MolecularDenoisingModel, eval_data: List[Dict]) -> Dict:
    
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


def save_checkpoint(model: MolecularDenoisingModel, config: Dict, save_path: None, optimizer=None, scheduler=None, scaler=None, epoch=None, best_metrics=None):
    """Save model checkpoint with full reproducibility state"""
    
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
            'scheme_params': {
                'coord_norm': model.scheme.coord_norm,
                'feature_norm': model.scheme.feature_norm,
                'feature_bias': model.scheme.feature_bias,
                'sigma_max': config['sigma_max'],
                'sigma_min': config['sigma_min']
            }
        }
    }
    
    # Add epoch information if provided
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    # Add optimizer state if provided
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Add scheduler state if provided
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Add scaler state if provided (for mixed precision training)
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # Add random number generator states for perfect reproducibility
    checkpoint['torch_rng_state'] = torch.get_rng_state()
    checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()
    
    # Add best metrics if provided
    if best_metrics is not None:
        checkpoint['best_metrics'] = best_metrics
    
    if save_path is None:
        torch.save(checkpoint, config['checkpoint_path'])
        log(f"✅ Checkpoint saved to {config['checkpoint_path']}")
    else:
        torch.save(checkpoint, save_path)
        log(f"✅ Checkpoint saved to {save_path}")


def load_checkpoint_for_resume(checkpoint_path: str, current_config: Dict) -> Tuple[MolecularDenoisingModel, Dict, Optional[Dict], Optional[Dict], Optional[Dict], Optional[Dict], Optional[int], Optional[Dict]]:
    """Load checkpoint for resuming training with full reproducibility state"""
    
    print(f"\nLoading checkpoint for resume from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # Initialize model using current config, load model state dict from checkpoint,
    model = create_molecular_model(current_config)
    model.denoiser.load_state_dict(checkpoint['model_state_dict'])
    
    # Get config from checkpoint (for reference, but we use current_config for the model)
    checkpoint_config = checkpoint.get('config', {})
    if checkpoint_config:
        print("⚠️  Note: Model will use current configuration parameters, not checkpoint configuration")
        print("    This ensures your command line arguments take precedence over saved checkpoint settings")
    
    # Get optimizer, scheduler, scaler, epoch, random number generator states, and best metrics if available
    optimizer_state = checkpoint.get('optimizer_state_dict', None)
    scheduler_state = checkpoint.get('scheduler_state_dict', None)
    scaler_state = checkpoint.get('scaler_state_dict', None)
    epoch = checkpoint.get('epoch', None)
    torch_rng_state = checkpoint.get('torch_rng_state', None)
    cuda_rng_state = checkpoint.get('cuda_rng_state', None)
    best_metrics = checkpoint.get('best_metrics', None)
    
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state.cpu())
        print("   PyTorch RNG state restored")
    
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state])
        print("   CUDA RNG state restored")
    
    if torch_rng_state is None and cuda_rng_state is None and 'seed' in current_config:
        print("   No RNG states found in checkpoint, setting seeds from config")    

    if optimizer_state is not None:
        print(f"   Optimizer state found")
    if scheduler_state is not None:
        print(f"   Scheduler state found")
    if scaler_state is not None:
        print(f"   Scaler state found")
    if best_metrics is not None:
        print(f"   Best metrics found")
    print(f"✅ Checkpoint loaded successfully\n", flush=True)

    return model, checkpoint_config, optimizer_state, scheduler_state, scaler_state, epoch, best_metrics


def load_checkpoint_for_resume(checkpoint_path: str, current_config: Dict) -> Tuple[MolecularDenoisingModel, Dict, Optional[Dict], Optional[Dict], Optional[Dict], Optional[Dict], Optional[int], Optional[Dict]]:
    """Load checkpoint for resuming training with full reproducibility state"""
    
    print(f"\nLoading checkpoint for resume from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # Initialize model using current config, load model state dict from checkpoint,
    model = create_molecular_model(current_config)
    model.denoiser.load_state_dict(checkpoint['model_state_dict'])
    
    # Get config from checkpoint (for reference, but we use current_config for the model)
    checkpoint_config = checkpoint.get('config', {})
    if checkpoint_config:
        print("⚠️  Note: Model will use current configuration parameters, not checkpoint configuration")
        print("    This ensures your command line arguments take precedence over saved checkpoint settings")
    
    # Get optimizer, scheduler, scaler, epoch, random number generator states, and best metrics if available
    optimizer_state = checkpoint.get('optimizer_state_dict', None)
    scheduler_state = checkpoint.get('scheduler_state_dict', None)
    scaler_state = checkpoint.get('scaler_state_dict', None)
    epoch = checkpoint.get('epoch', None)
    torch_rng_state = checkpoint.get('torch_rng_state', None)
    cuda_rng_state = checkpoint.get('cuda_rng_state', None)
    best_metrics = checkpoint.get('best_metrics', None)
    
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state.cpu())
        print("   PyTorch RNG state restored")
    
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state])
        print("   CUDA RNG state restored")
    
    if torch_rng_state is None and cuda_rng_state is None and 'seed' in current_config:
        print("   No RNG states found in checkpoint, setting seeds from config")    

    if optimizer_state is not None:
        print(f"   Optimizer state found")
    if scheduler_state is not None:
        print(f"   Scheduler state found")
    if scaler_state is not None:
        print(f"   Scaler state found")
    if best_metrics is not None:
        print(f"   Best metrics found")
    print(f"✅ Checkpoint loaded successfully\n", flush=True)

    return model, checkpoint_config, optimizer_state, scheduler_state, scaler_state, epoch, best_metrics


def analyze_samples(samples: Dict, config: Dict):
    """Analyze the generated molecular samples"""
    
    log(f"\n{'='*60}")
    log("SAMPLE ANALYSIS")
    log(f"{'='*60}")
    
    # Basic statistics 
    log("Coordinate Statistics:")
    log(f"  Ligand coords - Mean: {samples['ligand_coords'].mean(dim=0)}")
    log(f"  Ligand coords - Std:  {samples['ligand_coords'].std(dim=0)}")
    log(f"  Pocket coords - Mean: {samples['pocket_coords'].mean(dim=0)}")
    log(f"  Pocket coords - Std:  {samples['pocket_coords'].std(dim=0)}")
    
    # Check center of mass (should be near zero)
    total_coords = torch.cat([samples['ligand_coords'], samples['pocket_coords']], dim=0)
    total_mask = torch.cat([samples['ligand_mask'], samples['pocket_mask']], dim=0)
    
    # Calculate COM for each molecule
    batch_size = samples['batch_size']
    for i in range(batch_size):
        mol_coords = total_coords[total_mask == i]
        com = mol_coords.mean(dim=0)
        log(f"  Molecule {i} center of mass: {com}")
    
    # Check molecular geometry reasonableness
    log(f"\nMolecular Geometry Analysis:")
    for i in range(batch_size):
        lig_coords = samples['ligand_coords'][samples['ligand_mask'] == i]
        pocket_coords = samples['pocket_coords'][samples['pocket_mask'] == i]
        
        if len(lig_coords) > 1:
            lig_distances = torch.cdist(lig_coords, lig_coords)
            lig_distances = lig_distances[lig_distances > 0]  # Remove self-distances
            if len(lig_distances) > 0:  # Only calculate stats if we have distances
                log(f"  Molecule {i} ligand bond lengths - Mean: {lig_distances.mean():.3f}, Min: {lig_distances.min():.3f}")
            else:
                log(f"  Molecule {i} ligand bond lengths - No valid distances found")
        
        if len(pocket_coords) > 1:
            pocket_distances = torch.cdist(pocket_coords, pocket_coords)
            pocket_distances = pocket_distances[pocket_distances > 0]
            if len(pocket_distances) > 0:  # Only calculate stats if we have distances
                log(f"  Molecule {i} pocket distances - Mean: {pocket_distances.mean():.3f}, Min: {pocket_distances.min():.3f}")
            else:
                log(f"  Molecule {i} pocket distances - No valid distances found")
    
    log(f"\n✅ Sample analysis completed!")


def generate_grid_search_configs(param_spaces: Dict, max_combinations: int = 100) -> List[Dict]:
    """Generate all combinations for grid search"""
    
    # Get all parameter combinations
    param_names = list(param_spaces.keys())
    param_values = list(param_spaces.values())
    all_combinations = list(itertools.product(*param_values))
    
    # Limit number of combinations if too many
    if len(all_combinations) > max_combinations:
        log(f"Warning: {len(all_combinations)} combinations found, limiting to {max_combinations}")
        all_combinations = random.sample(all_combinations, max_combinations)
    
    configs = []
    for combination in all_combinations:
        config = CONFIG.copy()
        for param_name, param_value in zip(param_names, combination):
            config[param_name] = param_value
        configs.append(config)
    
    return configs


def generate_random_search_configs(param_spaces: Dict, num_trials: int) -> List[Dict]:
    """Generate random configurations for random search"""
    
    configs = []
    for _ in range(num_trials):
        config = CONFIG.copy()
        for param_name, param_values in param_spaces.items():
            config[param_name] = random.choice(param_values)
        configs.append(config)
    
    return configs


def run_optimization_configs(configs: List[Dict], train_data: List[Dict], eval_data: List[Dict], 
                             csv_path: str, json_path: str, optimization_id: str, optimization_metrics: List[str]):

    for i, config in enumerate(configs):
        log(f"\n\n\n--- Configuration {i+1}/{len(configs)} ---")

        # Create name for the individual training run
        run_id = f"{optimization_id}_{i}"
        config['wandb']['name'] = run_id
        config['run_dir'] = f'{config["run_dir"]}/{run_id}'
        if not os.path.exists(config['run_dir']):
            os.makedirs(config['run_dir'])
        config['checkpoint_path'] = f'{config["run_dir"]}/checkpoint.pt'
        
        log("Main Hyperparameters:")
        log(f"N-Layers: {config['n_layers']}")
        log(f"Joint NF: {config['joint_nf']}")
        log(f"Hidden NF: {config['hidden_nf']}")
        log(f"Edge Embedding Dim: {config['edge_embedding_dim']}")
        log(f"Learning Rate: {config['learning_rate']}")
        log(f"Batch Size: {config['batch_size']}")
        log(f"Num Sampling Steps: {config['num_sampling_steps']}")

        try:
            model = create_molecular_model(config)
            best_metrics, early_stopped, _, _, _, _ = train_model(model, train_data, eval_data, config)
        
        except Exception as e:
            log(f"Configuration failed: {e}")
            for key in optimization_metrics:
                config.update({key: float('inf')})
            if csv_path: save_result_to_csv(csv_path, config, run_id, 'failed')
            if json_path: save_result_to_json(json_path, config, run_id, 'failed')
            continue


        # Save result of this configuration to CSV and JSON   
        for key in optimization_metrics:
            config.update({key: best_metrics.get(key, float('inf'))})
        status = 'completed' if not early_stopped else 'early_stopped'
        if csv_path: save_result_to_csv(csv_path, config, run_id, status)
        if json_path: save_result_to_json(json_path, config, run_id, status)

        log(f"Configuration {i+1}/{len(configs)} completed")


def objective_function(trial, param_spaces: Dict, train_data: List[Dict], eval_data: List[Dict],
                       base_config: Dict, csv_path: str = None, json_path: str = None, 
                       optimization_id: str = None, optimization_metrics: List[str] = None, 
                       trial_number: int = None) -> float:
    
    config = base_config.copy()
    
    # Create name for the individual training run
    run_id = f"{optimization_id}_trial_{trial_number}"
    config['wandb']['name'] = run_id
    config['run_dir'] = f'{config["run_dir"]}/{run_id}'
    if not os.path.exists(config['run_dir']):
        os.makedirs(config['run_dir'])
    config['checkpoint_path'] = f'{config["run_dir"]}/checkpoint.pt'

    log(f"\n\n\n[Optuna] Starting trial {trial_number} (run_id: {run_id})")
    log("[Optuna] Suggesting hyperparameters:")
    
    # HYPERPARAMETER SUGGESTIONS
    for param_name, param_values in param_spaces.items():
        if isinstance(param_values[0], int):
            config[param_name] = trial.suggest_categorical(param_name, param_values)
        elif isinstance(param_values[0], float):
            config[param_name] = trial.suggest_categorical(param_name, param_values)
        log(f"    {param_name}: {config[param_name]}")
    
    try:
        # Create and train model
        model = create_molecular_model(config)
        best_metrics, early_stopped, _, _, _, _ = train_model(model, train_data, eval_data, config)
        
        final_loss = (best_metrics.get('atoms_dist_js_divergence', float('inf')) +
                      best_metrics.get('aa_dist_js_divergence', float('inf')) +
                      best_metrics.get('percent_fragmented', float('inf')) + 
                      best_metrics.get('ring_size_dist_js_divergence', float('inf')))
        
    except Exception as e:
        log(f"[Optuna] Trial {trial_number} failed with error: {e}")
        for key in optimization_metrics:
                config.update({key: float('inf')})
        if csv_path: save_result_to_csv(csv_path, config, run_id, 'failed')
        if json_path: save_result_to_json(json_path, config, run_id, 'failed')
        return float('inf')

    # Save result of this configuration to CSV and JSON
    for key in optimization_metrics:
        config.update({key: best_metrics.get(key, float('inf'))})
    status = 'completed' if not early_stopped else 'early_stopped'
    if csv_path: save_result_to_csv(csv_path, config, run_id, status)
    if json_path: save_result_to_json(json_path, config, run_id, status)

    log(f"[Optuna] Trial {trial_number} completed. Final loss: {final_loss}")
    return final_loss


def run_bayesian_optimization(param_spaces: Dict, train_data: List[Dict], 
                            eval_data: List[Dict], num_trials: int = 50, 
                            csv_path: str = None, json_path: str = None, 
                            optimization_id: str = None, optimization_metrics: List[str] = None,
                            study_name: str = None, storage: str = None) -> Tuple[Dict, float]:
    
    """Run Bayesian optimization using Optuna with persistent storage"""
    
    if not OPTUNA_AVAILABLE: raise ImportError("Optuna is required for Bayesian optimization")

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    log(f"\n[Optuna] Starting Bayesian optimization with {num_trials} trials...")
    trial_counter = 0
    
    def objective(trial):
        nonlocal trial_counter
        trial_counter += 1
        return objective_function(trial, param_spaces, train_data, eval_data, CONFIG, 
                                 csv_path, json_path, optimization_id, optimization_metrics, trial_counter)

    # Set up persistent storage if requested
    if study_name is None:
        study_name = optimization_id or f"optuna_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if storage is None:
        storage = f"sqlite:///{study_name}.db"
    log(f"[Optuna] Using study_name: {study_name}")
    log(f"[Optuna] Using storage: {storage}")

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        log(f"[Optuna] Loaded existing study '{study_name}' from storage.")
    except Exception:
        study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage)
        log(f"[Optuna] Created new study '{study_name}' with storage.")

    study.optimize(objective, n_trials=num_trials)
    log(f"[Optuna] Optimization completed. Best trial:")
    log(f"    Value: {study.best_value}")
    log(f"    Params: {study.best_params}")
    
    best_config = CONFIG.copy()
    for param_name in param_spaces.keys():
        best_config[param_name] = study.best_params[param_name]
    
    return best_config, study.best_value
            

def run_hyperparameter_optimization(optimization_type: str, param_spaces: Dict, 
                                  train_data: List[Dict], eval_data: List[Dict], 
                                  **kwargs) -> List[Tuple[Dict, float]]:
    
    log(f"\n{'='*20}" + f"HYPERPARAMETER OPTIMIZATION: {optimization_type.upper()}" + f"{'='*20}\n")

    optimization_metrics = [
        # Training losses
        'epoch/loss',
        'epoch/coord_loss',
        'epoch/categorical_loss',
        'epoch/coord_loss_ligand',
        'epoch/coord_loss_pocket',
        'epoch/categorical_loss_ligand',
        'epoch/categorical_loss_pocket',
        'epoch/avg_sigma',

        # Evaluation losses
        'coord_loss_lvl0',
        'coord_loss_lvl1',
        'coord_loss_lvl2',
        'coord_loss_lvl3',
        'coord_loss_lvl4',
        'categorical_loss_lvl0',
        'categorical_loss_lvl1',
        'categorical_loss_lvl2',
        'categorical_loss_lvl3',
        'categorical_loss_lvl4',
        'atom_accuracy_lvl0',
        'atom_accuracy_lvl1',
        'atom_accuracy_lvl2',
        'atom_accuracy_lvl3',
        'atom_accuracy_lvl4',
        'residue_accuracy_lvl0',
        'residue_accuracy_lvl1',
        'residue_accuracy_lvl2',
        'residue_accuracy_lvl3',
        'residue_accuracy_lvl4',

        # Integrity metrics
        'percent_valid',
        'percent_fragmented',
        'percent_disconnected',
        'percent_valence_issues',
        'n_rings_per_mol',
        'mean_num_rings',
        'mean_ring_size',
        'ring_size_dist_kl_divergence',
        'ring_size_dist_js_divergence',
        'mean_num_fragments',
        'mean_num_valence_issues',

        # Distribution losses
        'atoms_dist_kl_divergence',
        'atoms_dist_js_divergence',
        'aa_dist_kl_divergence',
        'aa_dist_js_divergence'
    ]

    # Create name for the optimization
    optimization_id = f"{datetime.now().strftime('%m%d_%H%M%S')}_{optimization_type}"
    CONFIG['run_dir'] = f'optimization_runs/{optimization_id}'
    if not os.path.exists(CONFIG['run_dir']):
        os.makedirs(CONFIG['run_dir'])
    
    # Initialize CSV results file in the optimization run directory
    csv_path = os.path.join(CONFIG['run_dir'], 'optimization_results.csv')
    json_path = os.path.join(CONFIG['run_dir'], 'optimization_results.json')
    initialize_csv_results(csv_path, CONFIG, ['run_id', 'status'], ['checkpoint_path'] + optimization_metrics)
    
    results = []
    
    if optimization_type == 'grid':
        configs = generate_grid_search_configs(param_spaces, kwargs.get('max_combinations', 100))
        log(f"Running grid search with {len(configs)} configurations...")
        run_optimization_configs(configs, train_data, eval_data, csv_path, json_path, optimization_id, optimization_metrics)

    elif optimization_type == 'random':
        configs = generate_random_search_configs(param_spaces, kwargs.get('num_trials', 20))
        log(f"Running random search with {len(configs)} trials...")
        run_optimization_configs(configs, train_data, eval_data, csv_path, json_path, optimization_id, optimization_metrics)
    
    elif optimization_type == 'bayesian':
        num_trials = kwargs.get('num_trials', 20)
        study_name = kwargs.get('study_name', None)
        storage = kwargs.get('storage', None)
        log(f"Running Bayesian optimization with {num_trials} trials...")
        
        best_config, best_loss = run_bayesian_optimization(
            param_spaces, 
            train_data, 
            eval_data, 
            num_trials, 
            csv_path, 
            json_path, 
            optimization_id, 
            optimization_metrics,
            study_name=study_name,
            storage=storage)

        results.append((best_config, best_loss)) 

        # Save best config to JSON at the run directory
        with open(os.path.join(CONFIG['run_dir'], 'best_config.json'), 'w') as f:
            json.dump(best_config, f, indent=2)
    
    else: 
        raise ValueError(f"Unknown optimization type: {optimization_type}")
    
    # SUMMARY
    results.sort(key=lambda x: x[1]) # Sort results by loss
    log(f"\n{'='*60}")
    log("OPTIMIZATION RESULTS SUMMARY")
    log(f"{'='*60}")
    for i, (config, loss) in enumerate(results[:5]):  # Top 5 results
        log(f"Rank {i+1}: Loss = {loss:.4f}")
        log(f"  Parameters: {', '.join([f'{k}={v}' for k, v in config.items() if k in param_spaces])}")
    
    return results



def main():
    """Main pipeline demonstrating the complete molecular diffusion workflow with timing"""

    args = parse_arguments()

    # Add toggle to CONFIG for convenience
    use_warnings = getattr(args, "use_warnings", False)

    global CONFIG
    CONFIG = update_config_from_args(CONFIG, args)

    log("🧬 MOLECULAR DIFFUSION PIPELINE", use_warnings)
    log(f"Mode: {args.mode}", use_warnings)
    log(f"Device: {CONFIG['device']}", use_warnings)

    if args.mode != 'single':
        os.makedirs(args.output_dir, exist_ok=True)

    # Set random seed for reproducible data creation
    if 'seed' in CONFIG:
        set_random_seeds(CONFIG['seed'])

    # 1. Generate training and evaluation data
    # --------------------------------------------------------------------------------------
    try:
        log(f"\n{'='*60}")
        log("GENERATING TRAINING DATA")
        log(f"{'='*60}")

        t0 = time.perf_counter()
        train_data = create_batches_from_dataset(CONFIG['train_dataset_path'], CONFIG)
        eval_data = create_batches_from_dataset(CONFIG['eval_dataset_path'], CONFIG)
        t1 = time.perf_counter()
        log_time("Training/Eval data creation", t0, t1, use_warnings)

        example_batch = train_data[0]
        log(f"Example batch: Ligand atoms={example_batch['ligand_coords'].shape[0]}, "
            f"Pocket residues={example_batch['pocket_coords'].shape[0]}",
            use_warnings)

    except Exception as e:
        log(f"Error generating training data: {e}", use_warnings)
        return
    # --------------------------------------------------------------------------------------


    try:
        # Single training run
        # --------------------------------------------------------------------------------------
        if args.mode == 'single':

            log("="*60 + "\nSINGLE TRAINING RUN\n" + "="*60, use_warnings)

            # --- Setup ---
            t_setup_start = time.perf_counter()
            if isinstance(CONFIG['train_dataset_path'], str):
                run_id = f"{datetime.now().strftime('%m%d_%H%M%S')}_{CONFIG['train_dataset_path'].split('.')[0]}"
            elif isinstance(CONFIG['train_dataset_path'], Path): 
                run_id = f"{datetime.now().strftime('%m%d_%H%M%S')}_{CONFIG['train_dataset_path'].stem}"
            else:
                raise TypeError("Unknown train_dataset_path type")

            CONFIG['wandb']['name'] = run_id
            CONFIG['run_dir'] = f'training_runs/{run_id}'
            os.makedirs(CONFIG['run_dir'], exist_ok=True)
            if "checkpoint_path" not in CONFIG:
                CONFIG['checkpoint_path'] = f'{CONFIG["run_dir"]}/checkpoint.pt'
            t_setup_end = time.perf_counter()
            log_time("Setup", t_setup_start, t_setup_end, use_warnings)

            # --- Model creation ---
            log("="*60 + "\nCREATING MODEL\n" + "="*60, use_warnings)
            t_model_start = time.perf_counter()
            model = create_molecular_model(CONFIG)
            t_model_end = time.perf_counter()
            log_time("Model creation", t_model_start, t_model_end, use_warnings)
            log(f"-> Model is on {next(model.denoiser.parameters()).device}", use_warnings)
            if next(model.denoiser.parameters()).device == "cpu": raise RuntimeWarning("Model is on CPU, training will be very slow!")

            # --- Training ---
            t_train_start = time.perf_counter()
            best_metrics, early_stopped, _, _, _, _  = train_model(model, train_data, eval_data, CONFIG)
            t_train_end = time.perf_counter()
            log_time("Training", t_train_start, t_train_end, use_warnings)

            # --- Save checkpoint ---
            t_ckpt_start = time.perf_counter()
            ckpt_path = CONFIG['checkpoint_path'].replace(".pt", "_final.pt")
            save_checkpoint(model, CONFIG, save_path=ckpt_path)
            t_ckpt_end = time.perf_counter()
            log_time("Checkpoint saving", t_ckpt_start, t_ckpt_end, use_warnings)

            # --- Load checkpoint ---
            log("="*60 + "\nLOADING MODEL FROM CHECKPOINT\n" + "="*60, use_warnings)
            t_load_start = time.perf_counter()
            loaded_model = load_checkpoint(ckpt_path)
            t_load_end = time.perf_counter()
            log_time("Model loading", t_load_start, t_load_end, use_warnings)

            # --- Sampling ---
            t_sample_start = time.perf_counter()
            if not CONFIG["update_pocket_coords"]:
                sample_loader = create_batches_from_dataset(CONFIG['eval_dataset_path'], {"batch_size": CONFIG["num_eval_samples"]})[0] # TODO should be replaced by a test dataset
                samples = sample_molecules_conditionally(
                    loaded_model, 
                    sample_loader,
                    num_steps = CONFIG['num_sampling_steps'], 
                    schedule_type=CONFIG['schedule_type'], 
                    guidance_scale = CONFIG["cfg_guidance_scale"] # attach guidance scale
                )
                print("Conditional Sampling enabled!")
            else:
                samples = sample_molecules(
                    loaded_model,
                    num_steps=CONFIG['num_sampling_steps'],
                    schedule_type=CONFIG['schedule_type'],
                    num_samples=CONFIG['num_eval_samples']
                    )
            t_sample_end = time.perf_counter()
            log_time("Sampling", t_sample_start, t_sample_end, use_warnings)

            for s in range(CONFIG['num_eval_samples']):
                graph = Data(
                    ligand_coords=samples['ligand_coords'][samples['ligand_mask']==s].cpu(),
                    ligand_features=samples['ligand_features'][samples['ligand_mask']==s].cpu(),
                    pocket_coords=samples['pocket_coords'][samples['pocket_mask']==s].cpu(),
                    pocket_features=samples['pocket_features'][samples['pocket_mask']==s].cpu()
                )
                torch.save(graph, CONFIG['run_dir'] + f"/sample_{s}.pt")

            # --- Analysis ---
            t_analysis_start = time.perf_counter()
            analyze_samples(samples, CONFIG)
            t_analysis_end = time.perf_counter()
            log_time("Analysis", t_analysis_start, t_analysis_end, use_warnings)

            log("🎉 PIPELINE COMPLETED SUCCESSFULLY!", use_warnings)
            log(f"Checkpoint saved at: {CONFIG['checkpoint_path']}", use_warnings)
            log(f"Used {torch.cuda.get_device_name()} for computation", use_warnings)

        else:
            log("="*60 + "\nHYPERPARAMETER OPTIMIZATION\n" + "="*60, use_warnings)
            t_opt_start = time.perf_counter()

            param_spaces = {}
            for param_name in ['num_sampling_steps', 'joint_nf', 'hidden_nf', 'n_layers', 
                               'edge_embedding_dim', 'learning_rate', 'batch_size']:
                if hasattr(args, param_name) and getattr(args, param_name) is None:
                    param_spaces[param_name] = HYPERPARAM_SPACES[param_name]

            if not param_spaces:
                log("No parameters to optimize (all specified via CLI)", use_warnings)
                return

            log(f"Optimizing parameters: {list(param_spaces.keys())}", use_warnings)

            # Run optimization
            # --------------------------------------------------------------------------------------
            optimization_kwargs = {
                'max_combinations': args.max_combinations,
                'num_trials': args.num_trials,
                'study_name': args.study_name,
                'storage': args.storage,
            }
            results = run_hyperparameter_optimization(args.mode, param_spaces, train_data, eval_data, 
                                                    **optimization_kwargs)
            
            t_opt_end = time.perf_counter()
            log_time("Hyperparameter optimization", t_opt_start, t_opt_end, use_warnings)
            # --------------------------------------------------------------------------------------
            
    except Exception as e:
        if wandb.run is not None:
            wandb.run.finish(exit_code=1)
        raise e


# ------------------------------------------------------------------------------------------------
# Functions for saving optimization results
# ------------------------------------------------------------------------------------------------

def initialize_csv_results(csv_path: str, config_template: Dict, add_columns_start: List[str] = [], add_columns_end: List[str] = []):
    """Initialize CSV file with headers for all CONFIG parameters"""
    
    # Get all possible parameter names from CONFIG
    all_params = list(config_template.keys())
    
    # Add result columns
    columns = add_columns_start + all_params + add_columns_end
    
    # Create CSV file with headers
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
    
    log(f"CSV results file initialized: {csv_path}")



def save_result_to_csv(csv_path: str, config: Dict, run_id: str, status: str = 'completed'):
    """Save a single optimization result to CSV file"""
    
    # Get all possible parameter names from CONFIG
    all_params = list(config.keys())
    
    # Prepare row data
    row_data = [run_id, status]
    for param in all_params:
        value = config[param]
        # Convert complex types to strings
        if isinstance(value, (list, tuple, dict)):
            value = str(value)
        row_data.append(value)
    
    # Append to CSV file
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)
    
    log(f"Result saved to CSV: {run_id} - Status: {status}")


def save_result_to_json(output_path: str, config: Dict, run_id: str, status: str = 'completed'):
    """Save optimization results to JSON file"""
    
    # Convert results to serializable format
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool)):
            serializable_config[key] = value
        elif isinstance(value, tuple):
            serializable_config[key] = list(value)
        elif isinstance(value, dict):
            serializable_config[key] = value

        serializable_config['status'] = status

    
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing_results = json.load(f)
        existing_results.update({run_id: serializable_config})
        with open(output_path, 'w') as f:
            json.dump(existing_results, f, indent=2)
    else:
        with open(output_path, 'w') as f:
            json.dump({run_id: serializable_config}, f, indent=2)
    
    log(f"Configuration results saved to {output_path}")



if __name__ == "__main__":
    main()