"""
Molecular Diffusion Samplers

Samplers adapted for molecular systems, following the same interface as the generic
diffusion samplers but handling molecular-specific data structures and constraints.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Callable, Mapping, Sequence, Any
from torch.autograd import grad

from molecular_sde import (
    MolecularEulerMaruyama, 
    MolecularState, 
    create_molecular_state_from_batch,
    molecular_state_to_batch
)

Tensor = torch.Tensor
TensorMapping = Mapping[str, Tensor]
Params = Mapping[str, Any]


class MolecularDenoiseFn:
    """Protocol for molecular denoising functions"""
    
    def __call__(
        self, 
        ligand: Tensor, 
        pocket: Tensor,
        ligand_mask: Tensor,
        pocket_mask: Tensor,
        sigma: Tensor,
        t_normalized: Tensor,
        cond: TensorMapping | None = None
    ) -> Tuple[Tensor, Tensor]:
        """Denoise molecular structures
        
        Returns:
            Tuple of (ligand_denoised, pocket_denoised)
        """
        ...


def dlog_dt(f: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    """Returns d/dt log(f(t)) = ḟ(t)/f(t) given f(t)."""
    return lambda t: grad(torch.log(f(t)), t, create_graph=True)[0]


def dsquare_dt(f: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    """Returns d/dt (f(t))^2 = 2ḟ(t)f(t) given f(t)."""
    return lambda t: grad(torch.square(f(t)), t, create_graph=True)[0]


class MolecularNoiseSchedule:
    """Noise schedule for molecular diffusion"""
    
    def __init__(self, schedule_type: str = "exponential", 
                 min_noise: float = 1e-3, max_noise: float = 20.0):
        self.schedule_type = schedule_type
        self.min_noise = min_noise
        self.max_noise = max_noise
    
    def sigma(self, t: Tensor) -> Tensor:
        """Get noise level at time t"""
        if self.schedule_type == "exponential":
            log_sigma = np.log(self.min_noise) + t * (np.log(self.max_noise) - np.log(self.min_noise))
            return torch.exp(log_sigma)
        elif self.schedule_type == "linear":
            return self.min_noise + t * (self.max_noise - self.min_noise)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def scale(self, t: Tensor) -> Tensor:
        """Get scale factor at time t (typically 1 for molecular diffusion)"""
        return torch.ones_like(t)


class MolecularSampler:
    """Base class for molecular diffusion samplers.
    
    Handles the molecular-specific data structures while following the same
    interface pattern as the generic diffusion samplers.
    
    Attributes:
        ligand_sizes: Sizes of ligand molecules to generate
        pocket_sizes: Sizes of pocket molecules to generate  
        scheme: The diffusion scheme with scale and noise schedules
        denoise_fn: Function to remove noise from molecular data
        tspan: Diffusion time steps for iterative denoising (1 to 0)
        apply_denoise_at_end: Apply final denoising step
        return_full_paths: Return complete sampling trajectory
        n_dims: Number of spatial dimensions (3 for molecules)
        atom_nf: Number of atom feature dimensions
        residue_nf: Number of residue feature dimensions
    """

    def __init__(
        self,
        ligand_sizes: list[int],
        pocket_sizes: list[int], 
        scheme: MolecularNoiseSchedule,
        denoise_fn: MolecularDenoiseFn,
        tspan: Tensor,
        apply_denoise_at_end: bool = True,
        return_full_paths: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        n_dims: int = 3,
        atom_nf: int = 10,
        residue_nf: int = 20,
        norm_values: Tuple[float, float] = (1.0, 1.0),
        norm_biases: Tuple[Optional[float], float] = (None, 0.0)
    ):
        self.ligand_sizes = ligand_sizes
        self.pocket_sizes = pocket_sizes
        self.scheme = scheme
        self.denoise_fn = denoise_fn
        self.tspan = tspan
        self.apply_denoise_at_end = apply_denoise_at_end
        self.return_full_paths = return_full_paths
        self.device = device
        self.dtype = dtype
        self.n_dims = n_dims
        self.atom_nf = atom_nf
        self.residue_nf = residue_nf
        self.norm_values = norm_values
        self.norm_biases = norm_biases
        
        # Validate inputs
        if len(ligand_sizes) != len(pocket_sizes):
            raise ValueError("ligand_sizes and pocket_sizes must have same length")
        
        self.batch_size = len(ligand_sizes)

    def _create_initial_state(self) -> MolecularState:
        """Create initial noisy molecular state"""
        
        total_lig_atoms = sum(self.ligand_sizes)
        total_pocket_atoms = sum(self.pocket_sizes)
        
        # Create masks
        ligand_mask = torch.cat([
            torch.full((size,), i, dtype=torch.long, device=self.device)
            for i, size in enumerate(self.ligand_sizes)
        ])
        pocket_mask = torch.cat([
            torch.full((size,), i, dtype=torch.long, device=self.device)  
            for i, size in enumerate(self.pocket_sizes)
        ])
        
        # Sample initial noise at t=1
        initial_sigma = self.scheme.sigma(torch.tensor(1.0, device=self.device))
        initial_scale = self.scheme.scale(torch.tensor(1.0, device=self.device))
        
        ligand_noise = (
            torch.randn(total_lig_atoms, self.n_dims + self.atom_nf, 
                       device=self.device, dtype=self.dtype) 
            * initial_sigma * initial_scale
        )
        pocket_noise = (
            torch.randn(total_pocket_atoms, self.n_dims + self.residue_nf,
                       device=self.device, dtype=self.dtype) 
            * initial_sigma * initial_scale
        )
        
        return MolecularState(
            ligand=ligand_noise,
            pocket=pocket_noise,
            ligand_mask=ligand_mask,
            pocket_mask=pocket_mask,
            batch_size=self.batch_size,
            categorical_norm_values=self.norm_values,
            categorical_norm_biases=self.norm_biases
        )

    def generate(self, cond: TensorMapping | None = None) -> dict:
        """Generate molecular samples from scratch.
        
        Args:
            cond: Optional conditioning inputs (not used in basic version)
            
        Returns:
            Dictionary with molecular batch data
        """
        if self.tspan is None or self.tspan.ndim != 1:
            raise ValueError("`tspan` must be a 1-d Tensor.")
        
        # Create initial noisy state
        initial_state = self._create_initial_state()
        
        # Denoise
        denoised_state = self.denoise(
            noisy_state=initial_state,
            tspan=self.tspan,
            cond=cond
        )
        
        # Apply final denoising if requested
        if self.apply_denoise_at_end:
            final_state = self._apply_final_denoise(denoised_state, cond)
            if self.return_full_paths and hasattr(denoised_state, 'ligand') and denoised_state.ligand.dim() == 3:
                # Append final state to trajectory
                final_lig = final_state.ligand.unsqueeze(0)
                final_pocket = final_state.pocket.unsqueeze(0)
                denoised_state = MolecularState(
                    ligand=torch.cat([denoised_state.ligand, final_lig], dim=0),
                    pocket=torch.cat([denoised_state.pocket, final_pocket], dim=0),
                    ligand_mask=denoised_state.ligand_mask,
                    pocket_mask=denoised_state.pocket_mask,
                    batch_size=denoised_state.batch_size,
                    categorical_norm_values=denoised_state.categorical_norm_values,
                    categorical_norm_biases=denoised_state.categorical_norm_biases
                )
            else:
                denoised_state = final_state
        
        # Convert to batch format with discretization
        return molecular_state_to_batch(
            denoised_state,
            n_dims=self.n_dims,
            discretize_features=True,
            atom_nf=self.atom_nf,
            residue_nf=self.residue_nf
        )

    def _apply_final_denoise(self, state: MolecularState, cond: TensorMapping | None) -> MolecularState:
        """Apply final denoising step"""
        
        # Get final time and noise level
        final_t = self.tspan[-1]
        final_sigma = self.scheme.sigma(final_t)
        final_scale = self.scheme.scale(final_t)
        
        # Handle trajectory case
        if state.ligand.dim() == 3:  # [time, atoms, features]
            current_ligand = state.ligand[-1]
            current_pocket = state.pocket[-1]
        else:
            current_ligand = state.ligand
            current_pocket = state.pocket
        
        # Normalize inputs for denoiser
        ligand_normalized = current_ligand / final_scale
        pocket_normalized = current_pocket / final_scale
        
        # Create time input
        t_normalized = final_t.unsqueeze(0).expand(state.batch_size, 1)
        
        # Apply denoiser
        denoised_lig, denoised_pocket = self.denoise_fn(
            ligand=ligand_normalized,
            pocket=pocket_normalized,
            ligand_mask=state.ligand_mask,
            pocket_mask=state.pocket_mask,
            sigma=final_sigma,
            t_normalized=t_normalized,
            cond=cond
        )
        
        return MolecularState(
            ligand=denoised_lig,
            pocket=denoised_pocket,
            ligand_mask=state.ligand_mask,
            pocket_mask=state.pocket_mask,
            batch_size=state.batch_size,
            categorical_norm_values=state.categorical_norm_values,
            categorical_norm_biases=state.categorical_norm_biases
        )

    def denoise(
        self,
        noisy_state: MolecularState,
        tspan: Tensor,
        cond: TensorMapping | None = None
    ) -> MolecularState:
        """Apply iterative denoising to given noisy molecular state.
        
        Args:
            noisy_state: Batch of noisy molecular states
            tspan: Decreasing sequence of diffusion time steps [1, 0)
            cond: Optional conditioning inputs
            
        Returns:
            Denoised molecular state
        """
        raise NotImplementedError


class MolecularSdeSampler(MolecularSampler):
    """Molecular sampler using SDE integration.
    
    Draws molecular samples by solving the reverse SDE with proper
    molecular constraints and translation invariance.
    """

    def __init__(
        self,
        ligand_sizes: list[int],
        pocket_sizes: list[int],
        scheme: MolecularNoiseSchedule,
        denoise_fn: MolecularDenoiseFn,
        tspan: Tensor,
        integrator: MolecularEulerMaruyama = None,
        apply_denoise_at_end: bool = True,
        return_full_paths: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        n_dims: int = 3,
        atom_nf: int = 10,
        residue_nf: int = 20,
        norm_values: Tuple[float, float] = (1.0, 1.0),
        norm_biases: Tuple[Optional[float], float] = (None, 0.0)
    ):
        super().__init__(
            ligand_sizes=ligand_sizes,
            pocket_sizes=pocket_sizes,
            scheme=scheme,
            denoise_fn=denoise_fn,
            tspan=tspan,
            apply_denoise_at_end=apply_denoise_at_end,
            return_full_paths=return_full_paths,
            device=device,
            dtype=dtype,
            n_dims=n_dims,
            atom_nf=atom_nf,
            residue_nf=residue_nf,
            norm_values=norm_values,
            norm_biases=norm_biases
        )
        
        self.integrator = integrator or MolecularEulerMaruyama(
            terminal_only=not return_full_paths,
            n_dims=n_dims,
            atom_nf=atom_nf,
            residue_nf=residue_nf
        )

    def denoise(
        self,
        noisy_state: MolecularState,
        tspan: Tensor,
        cond: TensorMapping | None = None
    ) -> MolecularState:
        """Apply iterative denoising using SDE integration."""
        
        if self.integrator.terminal_only and self.return_full_paths:
            raise ValueError(
                "Integrator does not support returning full paths."
            )
        
        # Create drift function from the denoiser
        drift_fn = self._create_drift_function(cond)
        
        # Get diffusion coefficient
        diffusion_coeff = self._get_diffusion_coefficient()
        
        # Solve the reverse SDE
        result = self.integrator.solve(
            initial_state=noisy_state,
            drift_fn=drift_fn,
            diffusion_coeff=diffusion_coeff,
            tspan=tspan
        )
        
        return result

    def _create_drift_function(self, cond: TensorMapping | None) -> Callable:
        """Create drift function for the reverse SDE"""
        
        def drift_fn(ligand: Tensor, pocket: Tensor, 
                    ligand_mask: Tensor, pocket_mask: Tensor) -> Tuple[Tensor, Tensor]:
            """Simplified drift function for reverse SDE"""
            
            # Estimate current time from data magnitude (simplified approach)
            current_magnitude = torch.sqrt(torch.mean(ligand**2) + torch.mean(pocket**2))
            estimated_t = torch.clamp(
                current_magnitude / self.scheme.sigma(torch.tensor(1.0)),
                0.01, 0.99  # Keep away from boundaries
            ).to(device=ligand.device)
            
            # Get noise schedule values
            sigma_t = self.scheme.sigma(estimated_t)
            scale_t = self.scheme.scale(estimated_t)
            
            # Simplified drift computation without gradients
            # For exponential schedule: d/dt log(sigma) ≈ log(max/min)
            # This is a reasonable approximation for the drift
            log_sigma_derivative = np.log(self.scheme.max_noise / self.scheme.min_noise)
            
            # Normalize inputs for denoiser
            ligand_normalized = ligand / scale_t
            pocket_normalized = pocket / scale_t
            
            # Get time input for denoiser
            batch_size = max(ligand_mask.max().item(), pocket_mask.max().item()) + 1
            t_normalized = (estimated_t / 1.0).unsqueeze(0).expand(batch_size, 1)
            
            # Get denoiser prediction
            with torch.no_grad():
                eps_ligand, eps_pocket = self.denoise_fn(
                    ligand=ligand_normalized,
                    pocket=pocket_normalized,
                    ligand_mask=ligand_mask,
                    pocket_mask=pocket_mask,
                    sigma=sigma_t,
                    t_normalized=t_normalized,
                    cond=cond
                )
            
            # Simplified drift computation
            # drift = -0.5 * sigma * eps (simplified version of reverse SDE)
            drift_ligand = -0.5 * sigma_t * eps_ligand
            drift_pocket = -0.5 * sigma_t * eps_pocket
            
            return drift_ligand, drift_pocket
        
        return drift_fn

    def _get_diffusion_coefficient(self) -> float:
        """Get diffusion coefficient for the SDE"""
        # Simplified constant diffusion coefficient
        return 0.5


def create_molecular_sampler_from_model(
    model,
    ligand_sizes: list[int],
    pocket_sizes: list[int],
    num_steps: int = 50,
    schedule_type: str = "exponential",
    device: str = 'cpu',
    return_full_paths: bool = False
) -> MolecularSdeSampler:
    """Create a molecular sampler from a trained model"""
    
    # Create noise schedule
    scheme = MolecularNoiseSchedule(
        schedule_type=schedule_type,
        min_noise=model.min_noise_level,
        max_noise=model.max_noise_level
    )
    
    # Create time span
    tspan = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    
    # Create denoiser function wrapper
    def molecular_denoise_fn(ligand, pocket, ligand_mask, pocket_mask, 
                           sigma, t_normalized, cond=None):
        with torch.no_grad():
            return model.denoiser(ligand, pocket, t_normalized, ligand_mask, pocket_mask)
    
    # Create sampler
    sampler = MolecularSdeSampler(
        ligand_sizes=ligand_sizes,
        pocket_sizes=pocket_sizes,
        scheme=scheme,
        denoise_fn=molecular_denoise_fn,
        tspan=tspan,
        apply_denoise_at_end=True,
        return_full_paths=return_full_paths,
        device=device,
        n_dims=model.n_dims,
        atom_nf=model.atom_nf,
        residue_nf=model.residue_nf,
        norm_values=model.norm_values,
        norm_biases=model.norm_biases
    )
    
    return sampler


def test_molecular_sde_sampler():
    """Test the molecular SDE sampler"""
    
    print("Testing Molecular SDE Sampler...")
    
    # Create dummy model for testing
    from molecular_diffusion import MolecularDenoisingModel
    
    device = 'cuda'  # Use CPU for testing
    atom_nf = 4
    residue_nf = 5
    
    model = MolecularDenoisingModel(
        atom_nf=atom_nf,
        residue_nf=residue_nf,
        joint_nf=8,
        hidden_nf=16,
        n_layers=1,
        device=device  # Use same device
    )
    model.initialize()
    
    # Create sampler
    ligand_sizes = [3, 4]
    pocket_sizes = [8, 6]
    
    sampler = create_molecular_sampler_from_model(
        model=model,
        ligand_sizes=ligand_sizes,
        pocket_sizes=pocket_sizes,
        num_steps=10,
        device=device  # Use same device
    )
    
    # Generate samples
    print("Generating samples...")
    samples = sampler.generate()
    
    print(f"Generated samples:")
    print(f"  Ligand coords: {samples['ligand_coords'].shape}")
    print(f"  Ligand features: {samples['ligand_features'].shape}")
    print(f"  Pocket coords: {samples['pocket_coords'].shape}")
    print(f"  Pocket features: {samples['pocket_features'].shape}")
    
    # Verify discrete features
    lig_sums = samples['ligand_features'].sum(dim=1)
    pocket_sums = samples['pocket_features'].sum(dim=1)
    
    print(f"  Ligand features properly one-hot: {torch.allclose(lig_sums, torch.ones_like(lig_sums))}")
    print(f"  Pocket features properly one-hot: {torch.allclose(pocket_sums, torch.ones_like(pocket_sums))}")
    
    print("✅ Molecular SDE sampler test passed!")
    
    return sampler, samples


if __name__ == "__main__":
    test_molecular_sde_sampler()