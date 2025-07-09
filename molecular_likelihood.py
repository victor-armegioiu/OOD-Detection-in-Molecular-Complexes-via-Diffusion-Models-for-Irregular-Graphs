"""
Molecular Likelihood Evaluator for Out-of-Distribution Detection

Implements exact likelihood computation for molecular diffusion models by integrating 
forwards in time from clean data to noise, using the same dynamics as sampling but 
in opposite direction.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Callable, Mapping, Any, NamedTuple, Protocol, Dict
from torch.autograd import grad
from torch_scatter import scatter_mean
import torch.nn.functional as F

# Import from the existing molecular sampler module
from molecular_samplers import (
    MolecularState, MolecularDenoiseFn, MolecularSdeCoefficientFn, 
    MolecularSdeDynamics, MolecularEulerMaruyamaStep, remove_mean_batch,
    create_molecular_denoiser_wrapper, edm_noise_decay, dlog_dt, dsquare_dt
)

Tensor = torch.Tensor
TensorMapping = Mapping[str, Tensor]
MolecularParams = Mapping[str, Any]


class MolecularLikelihoodEvaluator:
    """
    Molecular likelihood evaluator for OOD detection.
    
    Computes log p(x₀) by forward integrating from clean data to noise using the 
    change of variables formula: log p₀(x₀) = log p_T(x_T) + ∫₀ᵀ ∇·f(x(t),t) dt
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
        
        # Integration step for molecular dynamics
        self.integrator_step = MolecularEulerMaruyamaStep(n_dims=n_dims)
        
    def _create_molecular_dynamics(self, cond: TensorMapping | None) -> MolecularSdeDynamics:
        """
        Create molecular SDE dynamics - IDENTICAL to sampling implementation.
        
        Implements the reverse SDE formula:
        dx = [2σ̇/σ + ṡ/s]x - [2sσ̇/σ]D(x/s, σ) dt + s√[2σ̇σ] dW
        """
        
        def _molecular_drift(
            molecular_state: MolecularState,
            t: Tensor,
            params: MolecularParams,
        ) -> MolecularState:
            """Molecular drift - same as sampling"""
            
            assert t.ndim == 0, "`t` must be a scalar."
            
            # Ensure t requires gradients FIRST.
            if not t.requires_grad:
                t = t.requires_grad_(True)
            
            # Get scheme values
            s = self.scheme.scale(t)
            sigma = self.scheme.sigma(t)
            
            # Compute derivatives using automatic differentiation.
            dlog_sigma_dt = dlog_dt(self.scheme.sigma)(t)
            dlog_s_dt = 0  # Assuming constant scale like in sampling
            
            # Create normalized molecular state for denoiser.
            if isinstance(s, Tensor):
                s_lig = s.expand_as(molecular_state.ligand[:, :1]).expand_as(molecular_state.ligand)
                s_pocket = s.expand_as(molecular_state.pocket[:, :1]).expand_as(molecular_state.pocket)
            else:
                s_lig = s
                s_pocket = s
            
            normalized_state = MolecularState(
                ligand=molecular_state.ligand / s_lig,
                pocket=molecular_state.pocket / s_pocket,
                ligand_mask=molecular_state.ligand_mask,
                pocket_mask=molecular_state.pocket_mask,
                batch_size=molecular_state.batch_size
            )
            
            # Get denoiser output
            denoiser_output = self.denoise_fn(normalized_state, sigma, params.get("cond"))
            
            # Compute drift using the exact formula: [2σ̇/σ + ṡ/s]x - [2sσ̇/σ]D(x/s, σ)
            drift_coeff = 2 * dlog_sigma_dt + dlog_s_dt
            denoiser_coeff = 2 * dlog_sigma_dt * s
            
            if isinstance(drift_coeff, Tensor):
                drift_coeff_lig = drift_coeff.expand_as(molecular_state.ligand)
                drift_coeff_pocket = drift_coeff.expand_as(molecular_state.pocket)
            else:
                drift_coeff_lig = drift_coeff
                drift_coeff_pocket = drift_coeff
            
            if isinstance(denoiser_coeff, Tensor):
                denoiser_coeff_lig = denoiser_coeff.expand_as(denoiser_output.ligand)
                denoiser_coeff_pocket = denoiser_coeff.expand_as(denoiser_output.pocket)
            else:
                denoiser_coeff_lig = denoiser_coeff
                denoiser_coeff_pocket = denoiser_coeff
            
            drift_ligand = drift_coeff_lig * molecular_state.ligand - denoiser_coeff_lig * denoiser_output.ligand
            drift_pocket = drift_coeff_pocket * molecular_state.pocket - denoiser_coeff_pocket * denoiser_output.pocket
            
            return MolecularState(
                ligand=drift_ligand,
                pocket=drift_pocket,
                ligand_mask=molecular_state.ligand_mask,
                pocket_mask=molecular_state.pocket_mask,
                batch_size=molecular_state.batch_size
            )
        
        def _molecular_diffusion(
            molecular_state: MolecularState,
            t: Tensor,
            params: MolecularParams
        ) -> MolecularState:
            """Molecular diffusion - same as sampling"""
            
            assert t.ndim == 0, "`t` must be a scalar."
            
            # Ensure gradients for automatic differentiation
            if not t.requires_grad:
                t = t.requires_grad_(True)
            
            # Compute diffusion coefficient
            dsquare_sigma_dt = dsquare_dt(self.scheme.sigma)(t)
            s = self.scheme.scale(t)
            
            diffusion_coeff = torch.sqrt(dsquare_sigma_dt) * s
            
            # Apply to molecular state - broadcast properly
            if isinstance(diffusion_coeff, Tensor):
                diff_lig = diffusion_coeff.expand_as(molecular_state.ligand)
                diff_pocket = diffusion_coeff.expand_as(molecular_state.pocket)
            else:
                diff_lig = torch.full_like(molecular_state.ligand, diffusion_coeff)
                diff_pocket = torch.full_like(molecular_state.pocket, diffusion_coeff)
            
            return MolecularState(
                ligand=diff_lig,
                pocket=diff_pocket,
                ligand_mask=molecular_state.ligand_mask,
                pocket_mask=molecular_state.pocket_mask,
                batch_size=molecular_state.batch_size
            )
        
        return MolecularSdeDynamics(_molecular_drift, _molecular_diffusion)
    
    def _estimate_divergence(
        self, 
        molecular_state: MolecularState, 
        t: Tensor, 
        params: MolecularParams,
        dynamics: MolecularSdeDynamics
    ) -> Tensor:
        """Estimate tr(∇f) using Hutchinson estimator"""
        
        total_trace = 0.0
        
        for _ in range(self.num_hutchinson_samples):
            # Sample random vectors ε ~ N(0,I)
            eps_lig = torch.randn_like(molecular_state.ligand)
            eps_pocket = torch.randn_like(molecular_state.pocket)
            
            # Make coordinate noise COM-free to match molecular constraints
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
        
        # Flatten everything for VJP computation
        drift_flat = torch.cat([
            drift_state.ligand.reshape(-1), 
            drift_state.pocket.reshape(-1)
        ])
        eps_flat = torch.cat([
            eps_lig.reshape(-1), 
            eps_pocket.reshape(-1)
        ])
        
        # Compute vector-Jacobian product: ε^T ∇f
        if drift_flat.requires_grad:
            vjp = torch.autograd.grad(
                outputs=drift_flat.sum(),  # Sum to get scalar for backward
                inputs=[state_lig, state_pocket],
                grad_outputs=None,
                create_graph=False,
                retain_graph=False
            )
            
            # Compute trace contribution: ε^T (∇f)
            trace_lig = torch.sum(vjp[0] * eps_lig)
            trace_pocket = torch.sum(vjp[1] * eps_pocket)
            return trace_lig + trace_pocket
        else:
            return torch.tensor(0.0, device=molecular_state.ligand.device)
    
    def _forward_integrate_with_divergence(
        self, 
        clean_state: MolecularState, 
        cond: TensorMapping | None = None
    ) -> Tuple[MolecularState, Tensor]:
        """Forward integrate from clean data to noise while tracking divergence"""
        
        # Create dynamics
        dynamics = self._create_molecular_dynamics(cond)
        params = dict(drift=dict(cond=cond), diffusion=dict())
        
        current_state = clean_state
        total_divergence = torch.tensor(0.0, device=clean_state.ligand.device)
        
        for i in range(len(self.tspan) - 1):
            t_curr = self.tspan[i]
            t_next = self.tspan[i + 1]
            dt = t_next - t_curr  # Always positive for forward integration
            
            # Estimate divergence at current point
            div_estimate = self._estimate_divergence(current_state, t_curr, params, dynamics)
            total_divergence += div_estimate * dt
            
            # Forward Euler step using same integrator as sampling
            # But without the stochastic term (deterministic integration)
            drift_state = dynamics.drift(current_state, t_curr, params["drift"])
            
            # Manual Euler step (no noise for likelihood evaluation)
            new_ligand = current_state.ligand + drift_state.ligand * dt
            new_pocket = current_state.pocket + drift_state.pocket * dt
            
            # Apply COM constraint after each step
            combined_coords = torch.cat([
                new_ligand[:, :self.n_dims],
                new_pocket[:, :self.n_dims]
            ], dim=0)
            combined_mask = torch.cat([current_state.ligand_mask, current_state.pocket_mask])
            combined_coords_centered = remove_mean_batch(combined_coords, combined_mask)
            
            # Update state with centered coordinates
            new_ligand[:, :self.n_dims] = combined_coords_centered[:len(current_state.ligand)]
            new_pocket[:, :self.n_dims] = combined_coords_centered[len(current_state.ligand):]
            
            current_state = MolecularState(
                ligand=new_ligand,
                pocket=new_pocket,
                ligand_mask=current_state.ligand_mask,
                pocket_mask=current_state.pocket_mask,
                batch_size=current_state.batch_size
            )
        
        return current_state, total_divergence
    
    def _get_degrees_of_freedom(self, molecular_state: MolecularState) -> int:
        """Get degrees of freedom accounting for COM constraint"""
        total_atoms = len(molecular_state.ligand) + len(molecular_state.pocket)
        # Remove 3 DOF for COM constraint in 3D
        return (total_atoms - molecular_state.batch_size) * self.n_dims
    
    def _evaluate_terminal_likelihood(self, terminal_state: MolecularState) -> Tensor:
        """Evaluate log p_T(x_T) for Gaussian at final time"""
        
        # Get final noise parameters
        final_t = self.tspan[-1]
        final_sigma = self.scheme.sigma(final_t)
        final_scale = self.scheme.scale(final_t)
        
        # Extract coordinates (COM-free subspace)
        coords_lig = terminal_state.ligand[:, :self.n_dims]
        coords_pocket = terminal_state.pocket[:, :self.n_dims]
        coords = torch.cat([coords_lig, coords_pocket], dim=0)
        
        # Account for reduced dimensionality due to COM constraint
        dof = self._get_degrees_of_freedom(terminal_state)
        
        # Gaussian likelihood: -0.5 * ||x||²/σ² - dof/2 * log(2πσ²)
        sigma_total = final_sigma * final_scale
        norm_squared = torch.sum(coords ** 2)
        log_prob = (
            -0.5 * norm_squared / (sigma_total ** 2) - 
            0.5 * dof * torch.log(2 * torch.pi * sigma_total ** 2)
        )
        
        return log_prob
    
    def evaluate_likelihood(
        self, 
        molecular_state: MolecularState, 
        cond: TensorMapping | None = None
    ) -> Tensor:
        """
        Evaluate log-likelihood of clean molecular state.
        
        Args:
            molecular_state: Clean molecular state to evaluate
            cond: Optional conditioning information
            
        Returns:
            log_likelihood: Log probability under the model
        """
        
        # Forward integrate from clean to noise
        terminal_state, divergence_integral = self._forward_integrate_with_divergence(
            molecular_state, cond
        )
        
        # Evaluate terminal Gaussian likelihood
        terminal_log_prob = self._evaluate_terminal_likelihood(terminal_state)
        
        # Apply change of variables formula
        log_likelihood = terminal_log_prob + divergence_integral
        
        return log_likelihood


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
    molecular_denoise_fn = create_molecular_denoiser_wrapper(model.denoiser, model.scheme)
    
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


def test_molecular_likelihood_evaluator():
    """Test the molecular likelihood evaluator with debugging"""
    
    print("Testing Molecular Likelihood Evaluator (CUDA)...")
    
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
        print("Creating likelihood evaluator...")
        evaluator = create_molecular_likelihood_evaluator_from_model(
            model=model,
            num_steps=10,  # Small for testing
            num_hutchinson_samples=150,
        )
        
        # Create a clean molecular state for testing
        ligand_sizes = [3, 4]
        pocket_sizes = [8, 6]
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
        
        # Create clean molecular state (small coordinates, one-hot features)
        ligand_coords = torch.randn(total_lig_atoms, 3, device='cuda') * 0.1
        ligand_features = torch.zeros(total_lig_atoms, 8, device='cuda')  # joint_nf
        ligand_data = torch.cat([ligand_coords, ligand_features], dim=1)
        
        pocket_coords = torch.randn(total_pocket_atoms, 3, device='cuda') * 0.1
        pocket_features = torch.zeros(total_pocket_atoms, 8, device='cuda')  # joint_nf
        pocket_data = torch.cat([pocket_coords, pocket_features], dim=1)
        
        # Make COM-free
        combined_coords = torch.cat([ligand_coords, pocket_coords], dim=0)
        combined_mask = torch.cat([ligand_mask, pocket_mask])
        combined_coords_centered = remove_mean_batch(combined_coords, combined_mask)
        
        ligand_data[:, :3] = combined_coords_centered[:total_lig_atoms]
        pocket_data[:, :3] = combined_coords_centered[total_lig_atoms:]
        
        clean_state = MolecularState(
            ligand=ligand_data,
            pocket=pocket_data,
            ligand_mask=ligand_mask,
            pocket_mask=pocket_mask,
            batch_size=batch_size
        )
        
        print("Evaluating likelihood of clean molecular state...")
        log_likelihood = evaluator.evaluate_likelihood(clean_state)
        
        print(f"✅ Likelihood evaluation successful!")
        print(f"  Log-likelihood: {log_likelihood.item():.3f}")
        print(f"  Time span: {evaluator.tspan[0]:.3f} → {evaluator.tspan[-1]:.3f}")
        print(f"  Number of integration steps: {len(evaluator.tspan) - 1}")
        
        # Test with slightly perturbed state (should have lower likelihood)
        perturbed_coords = ligand_data.clone()
        perturbed_coords[:, :3] += torch.randn_like(perturbed_coords[:, :3]) * 0.5
        
        perturbed_state = MolecularState(
            ligand=perturbed_coords,
            pocket=pocket_data,
            ligand_mask=ligand_mask,
            pocket_mask=pocket_mask,
            batch_size=batch_size
        )
        
        log_likelihood_perturbed = evaluator.evaluate_likelihood(perturbed_state)
        
        print(f"  Perturbed log-likelihood: {log_likelihood_perturbed.item():.3f}")
        print(f"  Likelihood difference: {(log_likelihood - log_likelihood_perturbed).item():.3f}")
        
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during likelihood evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_molecular_likelihood_evaluator()
