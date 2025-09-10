"""
Modified Molecular Likelihood Evaluator for Per-Sample Evaluation

Key changes to produce likelihood for each complex in the batch:
1. Loop over each sample in the batch
2. Extract individual molecular states
3. Compute likelihood for each sample separately
4. Return tensor of shape (batch_size,)
"""

import torch
import json
import argparse
import numpy as np
from typing import Tuple, Optional, Callable, Mapping, Any, NamedTuple, Protocol, Dict, List, Union
from torch.autograd import grad
from torch_scatter import scatter_mean
import torch.nn.functional as F
import os
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# Import from the existing molecular sampler module
from molecular_samplers import (
    MolecularState, MolecularDenoiseFn, MolecularSdeCoefficientFn, 
    MolecularSdeDynamics, MolecularEulerMaruyamaStep, remove_mean_batch,
    create_molecular_denoiser_wrapper, edm_noise_decay, dlog_dt, dsquare_dt
)
from metrics import load_checkpoint
from Dataset import PDBbind_Dataset


Tensor = torch.Tensor
TensorMapping = Mapping[str, Tensor]
MolecularParams = Mapping[str, Any]


class MolecularLikelihoodEvaluator:
    """
    Molecular likelihood evaluator for OOD detection - Per Sample Version.
    
    Returns likelihood for each complex in the batch separately.
    """
    
    def __init__(
        self,
        scheme,  # MolecularDiffusion
        denoise_fn: MolecularDenoiseFn,
        tspan: Tensor,  # Forward time schedule: 0.0 → 1.0
        num_hutchinson_samples: int = 1,
        n_dims: int = 3,
        atom_nf: int = 10,
        residue_nf: int = 20,
        joint_nf: int = 16,
    ):
        self.scheme = scheme
        self.denoise_fn = denoise_fn
        self.tspan = tspan.cuda()
        self.num_hutchinson_samples = num_hutchinson_samples
        self.n_dims = n_dims
        self.atom_nf = atom_nf
        self.residue_nf = residue_nf
        self.joint_nf = joint_nf
        
        # Verify tspan goes forward in time
        assert self.tspan[0] < self.tspan[-1], "tspan should be forward: t=0 → t=1"
    
    def _extract_single_sample(self, molecular_state: MolecularState, sample_idx: int) -> MolecularState:
        """Extract a single sample from the batch"""
        
        # Get masks for this sample
        ligand_mask = (molecular_state.ligand_mask == sample_idx)
        pocket_mask = (molecular_state.pocket_mask == sample_idx)
        
        # Extract data for this sample
        ligand_data = molecular_state.ligand[ligand_mask]
        pocket_data = molecular_state.pocket[pocket_mask]
        
        # Create new masks (all atoms belong to sample 0 now)
        new_ligand_mask = torch.zeros(len(ligand_data), dtype=torch.long, device=ligand_data.device)
        new_pocket_mask = torch.zeros(len(pocket_data), dtype=torch.long, device=pocket_data.device)
        
        return MolecularState(
            ligand=ligand_data,
            pocket=pocket_data,
            ligand_mask=new_ligand_mask,
            pocket_mask=new_pocket_mask,
            batch_size=1
        )
    
    def _create_molecular_dynamics(self, cond: TensorMapping | None) -> MolecularSdeDynamics:
        """Create molecular probability flow ODE dynamics for likelihood evaluation."""
        
        def _molecular_drift(state: MolecularState, t: Tensor, params: dict) -> MolecularState:
            # Make t require grad for autograd-based derivative
            if not t.requires_grad:
                t = t.requires_grad_(True)
        
            sigma_t = self.scheme.sigma(t)
            # d/dt log sigma(t)
            dlog_sigma_dt = dlog_dt(self.scheme.sigma)(t)
        
            x0_hat = self.denoise_fn(state, sigma=sigma_t, cond=params.get("cond"))
        
            coeff = dlog_sigma_dt  # scalar tensor
            drift_lig  = coeff * (state.ligand - x0_hat.ligand)
            drift_pock = coeff * (state.pocket - x0_hat.pocket)
        
            # (Optional but IMPORTANT for consistency) Project COM here so the
            # divergence you estimate matches the vector field we integrate.
            drift_coords = torch.cat([drift_lig[:, :self.n_dims], drift_pock[:, :self.n_dims]], 0)
            masks = torch.cat([state.ligand_mask, state.pocket_mask])
            drift_coords_centered = remove_mean_batch(drift_coords, masks)
            drift_lig  = torch.cat([drift_coords_centered[:len(drift_lig)],  drift_lig[:,  self.n_dims:]], dim=1)
            drift_pock = torch.cat([drift_coords_centered[len(drift_lig):], drift_pock[:, self.n_dims:]], dim=1)
        
            return MolecularState(
                ligand=drift_lig,
                pocket=drift_pock,
                ligand_mask=state.ligand_mask,
                pocket_mask=state.pocket_mask,
                batch_size=state.batch_size,
            )

        def _molecular_diffusion(
            molecular_state: MolecularState,
            t: Tensor,
            params: MolecularParams
        ) -> MolecularState:
            """Placeholder diffusion - not used for deterministic ODE"""
            return MolecularState(
                ligand=torch.zeros_like(molecular_state.ligand),
                pocket=torch.zeros_like(molecular_state.pocket),
                ligand_mask=molecular_state.ligand_mask,
                pocket_mask=molecular_state.pocket_mask,
                batch_size=molecular_state.batch_size
            )
        
        return MolecularSdeDynamics(_molecular_drift, _molecular_diffusion)
    
    def _estimate_divergence_single(
        self, 
        molecular_state: MolecularState, 
        t: Tensor, 
        params: MolecularParams,
        dynamics: MolecularSdeDynamics
    ) -> Tensor:
        """Estimate tr(∇f) using Hutchinson estimator for single sample"""
        
        assert molecular_state.batch_size == 1, "This method expects single sample"
        
        total_trace = 0.0
        
        for _ in range(self.num_hutchinson_samples):
            # Sample random vectors ε ~ N(0,I)
            eps_lig = torch.randn_like(molecular_state.ligand)
            eps_pocket = torch.randn_like(molecular_state.pocket)
            
            # Make coordinate noise COM-free
            eps_coords_combined = torch.cat([
                eps_lig[:, :self.n_dims], 
                eps_pocket[:, :self.n_dims]
            ], dim=0)
            combined_mask = torch.cat([molecular_state.ligand_mask, molecular_state.pocket_mask])
            eps_coords_centered = remove_mean_batch(eps_coords_combined, combined_mask)
            
            eps_lig[:, :self.n_dims] = eps_coords_centered[:len(molecular_state.ligand)]
            eps_pocket[:, :self.n_dims] = eps_coords_centered[len(molecular_state.ligand):]
            
            # Compute ε^T ∇f using vector-Jacobian product
            trace_contrib = self._compute_vjp_trace(
                molecular_state, t, params, dynamics, eps_lig, eps_pocket
            )
            total_trace += trace_contrib
        
        return total_trace / self.num_hutchinson_samples
    
    def _compute_vjp_trace(
        self, 
        molecular_state: MolecularState, 
        t: Tensor, 
        params: MolecularParams,
        dynamics: MolecularSdeDynamics,
        eps_lig: Tensor, 
        eps_pocket: Tensor
    ) -> Tensor:
        """Compute ε^T ∇f using autograd for Hutchinson estimator"""
        
        # Create state with gradients enabled
        state_lig = molecular_state.ligand.detach().requires_grad_(True)
        state_pocket = molecular_state.pocket.detach().requires_grad_(True)
        
        temp_state = MolecularState(
            ligand=state_lig,
            pocket=state_pocket,
            ligand_mask=molecular_state.ligand_mask,
            pocket_mask=molecular_state.pocket_mask,
            batch_size=molecular_state.batch_size
        )
        
        # Compute drift f(x,t)
        drift_state = dynamics.drift(temp_state, t, params["drift"])
        
        # Flatten everything for easier computation
        drift_flat = torch.cat([
            drift_state.ligand.reshape(-1), 
            drift_state.pocket.reshape(-1)
        ])
        eps_flat = torch.cat([
            eps_lig.reshape(-1), 
            eps_pocket.reshape(-1)
        ])
        
        # Compute vector-Jacobian product
        vjp_result = torch.autograd.grad(
            outputs=drift_flat,
            inputs=[state_lig, state_pocket],
            grad_outputs=eps_flat,
            create_graph=False,
            retain_graph=False,
            allow_unused=True
        )
        
        # Sum up trace contributions
        trace = 0.0
        if vjp_result[0] is not None:
            trace = trace + torch.sum(vjp_result[0] * eps_lig)
        if vjp_result[1] is not None:
            trace = trace + torch.sum(vjp_result[1] * eps_pocket)
        
        return trace
    
    def _forward_integrate_with_divergence_single(
        self, 
        clean_state: MolecularState, 
        cond: TensorMapping | None = None
    ) -> Tuple[MolecularState, Tensor]:
        """Forward integrate probability flow ODE while tracking divergence for single sample"""
        
        assert clean_state.batch_size == 1, "This method expects single sample"
        
        # Create dynamics
        dynamics = self._create_molecular_dynamics(cond)
        params = dict(drift=dict(cond=cond), diffusion=dict())
        
        current_state = clean_state
        total_divergence = torch.tensor(0.0, device=clean_state.ligand.device)
        
        for i in range(len(self.tspan) - 1):
            t_curr = self.tspan[i]
            t_next = self.tspan[i + 1]
            dt = t_next - t_curr
            
            # Estimate divergence at current point
            div_estimate = self._estimate_divergence_single(current_state, t_curr, params, dynamics)
            total_divergence = total_divergence + div_estimate * dt
            
            # Use Heun's method for better accuracy
            current_state = self._heun_step(current_state, t_curr, t_next, dynamics, params)
        
        return current_state, total_divergence
    
    def _heun_step(
        self, 
        state: MolecularState, 
        t_curr: Tensor, 
        t_next: Tensor,
        dynamics: MolecularSdeDynamics,
        params: MolecularParams
    ) -> MolecularState:
        """Heun's method (improved Euler) for better numerical accuracy"""
        dt = t_next - t_curr
        
        # First stage: Euler predictor
        k1 = dynamics.drift(state, t_curr, params["drift"])
        
        # Predicted state
        pred_ligand = state.ligand + k1.ligand * dt
        pred_pocket = state.pocket + k1.pocket * dt
        
        state_pred = MolecularState(
            ligand=pred_ligand,
            pocket=pred_pocket,
            ligand_mask=state.ligand_mask,
            pocket_mask=state.pocket_mask,
            batch_size=state.batch_size
        )
        
        # Second stage: Corrector
        k2 = dynamics.drift(state_pred, t_next, params["drift"])
        
        # Average the slopes
        new_ligand = state.ligand + 0.5 * (k1.ligand + k2.ligand) * dt
        new_pocket = state.pocket + 0.5 * (k1.pocket + k2.pocket) * dt
        
        # Apply COM constraint (this is redundant since I did it befor)
        # But I prefer doing it again so that small errors don't accumulate.
        combined_coords = torch.cat([
            new_ligand[:, :self.n_dims],
            new_pocket[:, :self.n_dims]
        ], dim=0)
        combined_mask = torch.cat([state.ligand_mask, state.pocket_mask])
        combined_coords_centered = remove_mean_batch(combined_coords, combined_mask)
        
        new_ligand[:, :self.n_dims] = combined_coords_centered[:len(state.ligand)]
        new_pocket[:, :self.n_dims] = combined_coords_centered[len(state.ligand):]
        
        return MolecularState(
            ligand=new_ligand,
            pocket=new_pocket,
            ligand_mask=state.ligand_mask,
            pocket_mask=state.pocket_mask,
            batch_size=state.batch_size
        )
    
    def _get_degrees_of_freedom_single(self, molecular_state: MolecularState) -> int:
        """Get degrees of freedom accounting for COM constraint for single sample"""
        assert molecular_state.batch_size == 1, "This method expects single sample"
        total_atoms = len(molecular_state.ligand) + len(molecular_state.pocket)
        # Remove 3 DOF for COM constraint in 3D (single molecule)
        return (total_atoms - 1) * self.n_dims
    
    def _evaluate_terminal_likelihood_single(self, terminal_state: MolecularState) -> Tensor:
        """Evaluate log p_T(x_T) for Gaussian at final time for single sample"""
        
        assert terminal_state.batch_size == 1, "This method expects single sample"
        
        # Get final noise parameters
        final_t = self.tspan[-1]
        final_sigma = self.scheme.sigma(final_t)
        final_scale = self.scheme.scale(final_t)
        
        # Extract coordinates and features
        coords_lig = terminal_state.ligand[:, :self.n_dims]
        coords_pocket = terminal_state.pocket[:, :self.n_dims]
        coords = torch.cat([coords_lig, coords_pocket], dim=0)
        
        features_lig = terminal_state.ligand[:, self.n_dims:]
        features_pocket = terminal_state.pocket[:, self.n_dims:]
        features = torch.cat([features_lig, features_pocket], dim=0)
        
        # Account for reduced dimensionality due to COM constraint
        coord_dof = self._get_degrees_of_freedom_single(terminal_state)
        feature_dim = features.shape[1]
        total_feature_dof = features.shape[0] * feature_dim
        
        # Gaussian likelihood for coordinates (COM-free)
        sigma_total = final_sigma * final_scale
        coord_norm_squared = torch.sum(coords ** 2)
        coord_log_prob = (
            -0.5 * coord_norm_squared / (sigma_total ** 2) - 
            0.5 * coord_dof * torch.log(2 * torch.pi * sigma_total ** 2)
        )
        
        # Gaussian likelihood for features (no constraint)
        feature_norm_squared = torch.sum(features ** 2)
        feature_log_prob = (
            -0.5 * feature_norm_squared / (sigma_total ** 2) - 
            0.5 * total_feature_dof * torch.log(2 * torch.pi * sigma_total ** 2)
        )
        
        return coord_log_prob + feature_log_prob
    
    def evaluate_likelihood(
        self, 
        molecular_state: MolecularState, 
        cond: TensorMapping | None = None
    ) -> Tensor:
        """
        Evaluate log-likelihood of clean molecular state for each sample in batch.
        
        Args:
            molecular_state: Clean molecular state to evaluate (batch)
            cond: Optional conditioning information
            
        Returns:
            log_likelihoods: Tensor of shape (batch_size,) with log probability for each sample
        """
        
        batch_size = molecular_state.batch_size
        log_likelihoods = torch.zeros(batch_size, device=molecular_state.ligand.device)
        
        # Process each sample in the batch separately
        for sample_idx in range(batch_size):
            # Extract single sample
            single_sample = self._extract_single_sample(molecular_state, sample_idx)
            
            # Extract conditioning for this sample if provided
            sample_cond = None
            if cond is not None:
                # Assume conditioning has same batch structure - you may need to adapt this
                sample_cond = {}
                for key, value in cond.items():
                    if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                        sample_cond[key] = value[sample_idx:sample_idx+1]  # Keep batch dimension
                    else:
                        sample_cond[key] = value
            
            # Forward integrate from clean to noise using probability flow ODE
            terminal_state, divergence_integral = self._forward_integrate_with_divergence_single(
                single_sample, sample_cond
            )
            
            # Evaluate terminal Gaussian likelihood
            terminal_log_prob = self._evaluate_terminal_likelihood_single(terminal_state)
            
            # Apply change of variables formula
            log_likelihood = terminal_log_prob + divergence_integral
            log_likelihoods[sample_idx] = log_likelihood
        
        return log_likelihoods


def create_molecular_likelihood_evaluator_from_model(
    model,
    num_steps: int = 50,
    num_hutchinson_samples: int = 1,
    config: Dict = {'clip_max': 100.0},
) -> MolecularLikelihoodEvaluator:
    """Create a molecular likelihood evaluator from a trained model"""
    
    # Use EXACT same time schedule as sampling (but reversed for forward integration)
    tspan_sampling = edm_noise_decay(scheme=model.scheme, num_steps=num_steps)
    
    # For likelihood evaluation, we need FORWARD integration: t=0 → t=1
    # So reverse the sampling schedule
    tspan_likelihood = torch.flip(tspan_sampling, dims=[0])
    
    # Create the SAME denoiser wrapper as sampling
    molecular_denoise_fn = create_molecular_denoiser_wrapper(
        model.denoiser, model.scheme, requires_grad=True
    )
    
    # Create evaluator
    evaluator = MolecularLikelihoodEvaluator(
        scheme=model.scheme,
        denoise_fn=molecular_denoise_fn,
        tspan=tspan_likelihood,  # Forward time schedule
        num_hutchinson_samples=num_hutchinson_samples,
        n_dims=model.n_dims,
        atom_nf=model.atom_nf,
        residue_nf=model.residue_nf,
        joint_nf=model.joint_nf,
    )
    evaluator._model = model
    
    return evaluator


def test_per_sample_likelihood_evaluator():
    """Test the per-sample molecular likelihood evaluator"""
    
    print("Testing Per-Sample Molecular Likelihood Evaluator (CUDA)...")
    
    try:
        # Import the molecular diffusion model
        from molecular_diffusion import MolecularDenoisingModel
        
        atom_nf = 4
        residue_nf = 5
        
        # Create dummy model for testing
        model = MolecularDenoisingModel(
            atom_nf=atom_nf,
            residue_nf=residue_nf,
            joint_nf=8,
            hidden_nf=16,
            n_layers=1
        )
        model.initialize()
        
        # Create likelihood evaluator
        print("Creating per-sample likelihood evaluator...")
        evaluator = create_molecular_likelihood_evaluator_from_model(
            model=model,
            num_steps=10,  # Small for testing
            num_hutchinson_samples=5,
        )
        
        # Create a batch with multiple molecular states
        ligand_sizes = [3, 4, 2]  # 3 samples
        pocket_sizes = [8, 6, 5]  # 3 samples
        batch_size = len(ligand_sizes)
        
        total_lig_atoms = sum(ligand_sizes)
        total_pocket_atoms = sum(pocket_sizes)
        
        # Create masks
        ligand_mask = torch.cat([
            torch.full((size,), i, dtype=torch.long, device='cuda')
            for i, size in enumerate(ligand_sizes)
        ])
        pocket_mask = torch.cat([
            torch.full((size,), i, dtype=torch.long, device='cuda')
            for i, size in enumerate(pocket_sizes)
        ])
        
        # Create clean molecular state
        ligand_coords = torch.randn(total_lig_atoms, 3, device='cuda') * 0.1
        ligand_features = torch.zeros(total_lig_atoms, 8, device='cuda')  # joint_nf
        ligand_data = torch.cat([ligand_coords, ligand_features], dim=1)
        
        pocket_coords = torch.randn(total_pocket_atoms, 3, device='cuda') * 0.1
        pocket_features = torch.zeros(total_pocket_atoms, 8, device='cuda')  # joint_nf
        pocket_data = torch.cat([pocket_coords, pocket_features], dim=1)
        
        # Make COM-free for each sample separately
        for i in range(batch_size):
            lig_mask = ligand_mask == i
            poc_mask = pocket_mask == i
            
            sample_coords = torch.cat([ligand_coords[lig_mask], pocket_coords[poc_mask]], dim=0)
            sample_mask = torch.cat([
                torch.zeros(lig_mask.sum(), dtype=torch.long, device='cuda'),
                torch.zeros(poc_mask.sum(), dtype=torch.long, device='cuda')
            ])
            
            sample_coords_centered = remove_mean_batch(sample_coords, sample_mask)
            
            ligand_data[lig_mask, :3] = sample_coords_centered[:lig_mask.sum()]
            pocket_data[poc_mask, :3] = sample_coords_centered[lig_mask.sum():]
        
        clean_state = MolecularState(
            ligand=ligand_data,
            pocket=pocket_data,
            ligand_mask=ligand_mask,
            pocket_mask=pocket_mask,
            batch_size=batch_size
        )
        
        print("Evaluating likelihood of clean molecular states...")
        log_likelihoods = evaluator.evaluate_likelihood(clean_state)
        
        print(f"✅ Per-sample likelihood evaluation successful!")
        print(f"  Batch size: {batch_size}")
        print(f"  Output shape: {log_likelihoods.shape}")
        print(f"  Log-likelihoods: {log_likelihoods}")
        
        # Verify we get one likelihood per sample
        assert log_likelihoods.shape == (batch_size,), f"Expected shape ({batch_size},), got {log_likelihoods.shape}"
        
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during likelihood evaluation: {e}")
        import traceback
        traceback.print_exc()


def process_dataset(dataset, evaluator, device, dataset_name):
    """Process a dataset and return likelihood statistics"""
    print(f"\nProcessing {dataset_name}...")
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, follow_batch=['lig_coords', 'prot_coords'])
    
    all_likelihoods = []
    batch_count = 0
    
    for batch in dataloader:
        batch = batch.to(device)
        print(batch)
        
        lig_features = batch.lig_features
        pocket_features = batch.prot_features
        print(f"lig_features.shape: {lig_features.shape}")
        print(f"pocket_features.shape: {pocket_features.shape}")

        # Convert one-hot features to embeddings
        ligand_embeddings = evaluator._model.denoiser.atom_encoder(lig_features)  # [N_lig, joint_nf]
        pocket_embeddings = evaluator._model.denoiser.residue_encoder(pocket_features)  # [N_pocket, joint_nf]
        
        print(f"ligand_embeddings.shape: {ligand_embeddings.shape}")
        print(f"pocket_embeddings.shape: {pocket_embeddings.shape}")        

        lig_coords = batch.lig_coords
        pocket_coords = batch.prot_coords
        ligand_mask = batch.lig_coords_batch
        pocket_mask = batch.prot_coords_batch

        print(f"lig_coords.shape: {lig_coords.shape}")
        print(f"pocket_coords.shape: {pocket_coords.shape}")
        print(f"ligand_mask.shape: {ligand_mask.shape}")
        print(f"pocket_mask.shape: {pocket_mask.shape}")

        # Concatenate coords and embeddings
        ligand_data = torch.cat([lig_coords, ligand_embeddings], dim=1)
        pocket_data = torch.cat([pocket_coords, pocket_embeddings], dim=1)

        # Center-of-mass correction for this molecule
        combined_coords = torch.cat([lig_coords, pocket_coords], dim=0)
        combined_mask = torch.cat([ligand_mask, pocket_mask])
        combined_coords_centered = remove_mean_batch(combined_coords, combined_mask)
        ligand_data[:, :3] = combined_coords_centered[:lig_coords.shape[0]]
        pocket_data[:, :3] = combined_coords_centered[lig_coords.shape[0]:]

        print()
        print(f"ligand_data.shape: {ligand_data.shape}")
        print(f"pocket_data.shape: {pocket_data.shape}")
        print(f"ligand_mask.shape: {ligand_mask.shape}")
        print(f"pocket_mask.shape: {pocket_mask.shape}")
        print()

        # Separate the batch into individual graphs
        print(torch.bincount(ligand_mask).tolist())
        print(torch.bincount(pocket_mask).tolist())

        ligand_data = torch.split(ligand_data, torch.bincount(ligand_mask).tolist())
        pocket_data = torch.split(pocket_data, torch.bincount(pocket_mask).tolist())

        for i in range(len(ligand_data)):
            print(f"Ligand data shape: {ligand_data[i].shape}")
            print(f"Pocket data shape: {pocket_data[i].shape}")

            ligand_mask = torch.zeros(ligand_data[i].shape[0], dtype=torch.long, device=device)
            pocket_mask = torch.zeros(pocket_data[i].shape[0], dtype=torch.long, device=device)

            # Create molecular state
            state = MolecularState(
                ligand=ligand_data[i],
                pocket=pocket_data[i],
                ligand_mask=ligand_mask,
                pocket_mask=pocket_mask,
                batch_size=1
            )

            # Evaluate likelihood for this molecule
            log_likelihood = evaluator.evaluate_likelihood(state)
            print(f"Complex {batch.id[i]} Log-likelihood: {log_likelihood.item()}", flush=True)
            all_likelihoods.append(log_likelihood.item())

            print()
            # batch_count += 1
            # if batch_count > 10:
            #     break

        # Count the number of saved likelihoods that are not NAN
        nan_count = torch.isnan(torch.tensor(all_likelihoods)).sum().item()
        if len(all_likelihoods) - nan_count >= 500:
            break
        else:
            print(f"Number of saved likelihoods: {len(all_likelihoods)}")
            print(f"Number of NANs: {nan_count}")



    return all_likelihoods, [f"{dataset_name}_{i}" for i in range(len(all_likelihoods))]


def main(dataset_path, checkpoint_path, results_folder, num_steps, num_hutchinson_samples):
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    print(f"Number of integration steps: {num_steps}")
    print(f"Number of Hutchinson samples: {num_hutchinson_samples}")
    print()
    
    # Check file existence
    for path in [dataset_path, checkpoint_path]:
        name = os.path.basename(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found at {path}")
        print(f"Found {name}: {path}")
    
    # Load datasets and model
    print("\nLoading dataset and model...")
    dataset = torch.load(dataset_path)
    name = os.path.basename(dataset_path)

    model = load_checkpoint(checkpoint_path)
    
    
    # Create likelihood evaluator
    evaluator = create_molecular_likelihood_evaluator_from_model(
        model,
        num_steps=num_steps,
        num_hutchinson_samples=num_hutchinson_samples,
    )
    
    # Process dataset
    ood_likelihoods, ood_ids = process_dataset(dataset, evaluator, device, name)
    
    # Compute statistics
    ood_mean = np.mean(ood_likelihoods)
    ood_std = np.std(ood_likelihoods)
    
    # Print results
    print("\n" + "="*60)
    print("LIKELIHOOD RESULTS")
    print("="*60)
    print(f"{name} (n={len(ood_likelihoods)}):")
    print(f"  Mean log-likelihood: {ood_mean:.4f}")
    print(f"  Std log-likelihood:  {ood_std:.4f}")
    print(f"  Min log-likelihood:  {min(ood_likelihoods):.4f}")
    print(f"  Max log-likelihood:  {max(ood_likelihoods):.4f}")
    print("="*60)

    results_file = os.path.join(results_folder, f'{name}')
    torch.save(ood_likelihoods, results_file)

    print(f"\nLog-likelihoods saved to: {results_file}")


if __name__ == '__main__':
    # test_per_sample_likelihood_evaluator()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='dataset_cleansplit_train.pt', type=str)
    parser.add_argument('--checkpoint_path', default='training_runs/0725_115513_dataset_cleansplit_train/checkpoint_epoch_579.pt', type=str)
    parser.add_argument('--results_folder', default='likelihood_results', type=str)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--num_hutchinson_samples', type=int, default=20)
    args = parser.parse_args()

    main(args.dataset_path, args.checkpoint_path, args.results_folder, args.num_steps, args.num_hutchinson_samples)
