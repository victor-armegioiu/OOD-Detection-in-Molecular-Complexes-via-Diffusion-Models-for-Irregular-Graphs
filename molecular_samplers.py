"""
Molecular SDE Samplers

Implements the same SDE sampling logic as the base framework but natively for molecular systems.
Follows exact reverse SDE dynamics with automatic differentiation, all CUDA-optimized.
"""

import torch
import argparse
import numpy as np
from typing import Tuple, Optional, Callable, Mapping, Any, NamedTuple, Protocol
from torch.autograd import grad
from torch_scatter import scatter_mean
from torch_geometric.data import Data

from molecular_diffusion import MolecularDiffusion, remove_mean_batch
from molecular_diffusion import MolecularDenoisingModel

Tensor = torch.Tensor
TensorMapping = Mapping[str, Tensor]
MolecularParams = Mapping[str, Any]


# ********************
# Molecular State and Protocols
# ********************

class MolecularState(NamedTuple):
    """Molecular state representation for SDE solving"""
    ligand: Tensor      # [total_lig_atoms, n_dims + atom_nf]
    pocket: Tensor      # [total_pocket_atoms, n_dims + residue_nf]
    ligand_mask: Tensor # [total_lig_atoms] - molecule indices
    pocket_mask: Tensor # [total_pocket_atoms] - molecule indices
    batch_size: int


class MolecularDenoiseFn(Protocol):
    def __call__(
        self, 
        molecular_state: MolecularState,
        sigma: Tensor, 
        cond: TensorMapping | None = None
    ) -> MolecularState: ...


class MolecularSdeCoefficientFn(Protocol):
    def __call__(
        self, 
        molecular_state: MolecularState, 
        t: Tensor, 
        params: MolecularParams
    ) -> MolecularState: ...


class MolecularSdeDynamics(NamedTuple):
    drift: MolecularSdeCoefficientFn
    diffusion: MolecularSdeCoefficientFn


# ********************
# Automatic Differentiation
# ********************

def dlog_dt(f: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    return lambda t: grad(torch.log(f(t)), t, create_graph=True)[0]


def dsquare_dt(f: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    return lambda t: grad(torch.square(f(t)), t, create_graph=True)[0]


# ********************
# Molecular SDE Solver 
# ********************

class MolecularEulerMaruyamaStep:    
    def __init__(self, n_dims: int = 3):
        self.n_dims = n_dims
    
    def step(
        self,
        dynamics: MolecularSdeDynamics,
        molecular_state: MolecularState,
        t0: Tensor,
        dt: Tensor,
        params: MolecularParams,
    ) -> MolecularState:
        """Makes one Euler-Maruyama integration step """
        
        # Check params structure (im just ensuring we have both the drift and the diffusion
        # coefficients.
        if not ("drift" in params.keys() and "diffusion" in params.keys()):
            raise ValueError("'params' must contain both 'drift' and 'diffusion' fields.")
        
        # Compute drift and diffusion coefficients
        drift_state = dynamics.drift(molecular_state, t0, params["drift"])
        diffusion_state = dynamics.diffusion(molecular_state, t0, params["diffusion"])
        
        # Sample molecular noise with proper COM handling
        noise_lig = torch.randn_like(molecular_state.ligand)
        noise_pocket = torch.randn_like(molecular_state.pocket)
        
        # Make coordinate noise COM-free (molecular-specific constraint)
        if self.n_dims > 0:
            noise_coords_combined = torch.cat([
                noise_lig[:, :self.n_dims], 
                noise_pocket[:, :self.n_dims]
            ], dim=0)
            combined_mask = torch.cat([molecular_state.ligand_mask, molecular_state.pocket_mask])
            noise_coords_centered = remove_mean_batch(noise_coords_combined, combined_mask)
            
            noise_lig[:, :self.n_dims] = noise_coords_centered[:len(molecular_state.ligand)]
            noise_pocket[:, :self.n_dims] = noise_coords_centered[len(molecular_state.ligand):]
        
        # Euler-Maruyama update.
        sqrt_dt = torch.sqrt(torch.abs(dt))
        
        new_ligand = (
            molecular_state.ligand
            + dt * drift_state.ligand
            + diffusion_state.ligand * noise_lig * sqrt_dt
        )
        
        new_pocket = (
            molecular_state.pocket
            + dt * drift_state.pocket
            + diffusion_state.pocket * noise_pocket * sqrt_dt
        )
        
        return MolecularState(
            ligand=new_ligand,
            pocket=new_pocket,
            ligand_mask=molecular_state.ligand_mask,
            pocket_mask=molecular_state.pocket_mask,
            batch_size=molecular_state.batch_size
        )


class MolecularEulerMaruyama(MolecularEulerMaruyamaStep):
    
    def __init__(self, time_axis_pos: int = 0, terminal_only: bool = False, n_dims: int = 3):
        super().__init__(n_dims)
        self.time_axis_pos = time_axis_pos
        self.terminal_only = terminal_only
    
    def __call__(
        self,
        dynamics: MolecularSdeDynamics,
        molecular_state0: MolecularState,
        tspan: Tensor,
        params: MolecularParams,
    ) -> MolecularState:
        
        if not self.terminal_only:
            # Store the entire path
            lig_path = [molecular_state0.ligand]
            pocket_path = [molecular_state0.pocket]
        
        current_state = molecular_state0
        
        for i in range(len(tspan) - 1):
            t0 = tspan[i]
            t_next = tspan[i + 1]
            dt = t_next - t0
            
            current_state = self.step(
                dynamics=dynamics,
                molecular_state=current_state,
                t0=t0,
                dt=dt,
                params=params,
            )
            
            # Apply molecular translation invariance after each step
            combined_coords = torch.cat([
                current_state.ligand[:, :self.n_dims],
                current_state.pocket[:, :self.n_dims]
            ], dim=0)
            combined_mask = torch.cat([current_state.ligand_mask, current_state.pocket_mask])
            combined_coords_centered = remove_mean_batch(combined_coords, combined_mask)
            
            # Update state with centered coordinates
            new_ligand = torch.cat([
                combined_coords_centered[:len(current_state.ligand)],
                current_state.ligand[:, self.n_dims:]
            ], dim=1)
            new_pocket = torch.cat([
                combined_coords_centered[len(current_state.ligand):],
                current_state.pocket[:, self.n_dims:]
            ], dim=1)
            
            current_state = MolecularState(
                ligand=new_ligand,
                pocket=new_pocket,
                ligand_mask=current_state.ligand_mask,
                pocket_mask=current_state.pocket_mask,
                batch_size=current_state.batch_size
            )
            
            if not self.terminal_only:
                lig_path.append(current_state.ligand.detach())
                pocket_path.append(current_state.pocket.detach())
        
        if self.terminal_only:
            return current_state
        else:
            # Stack trajectories
            ligand_trajectory = torch.stack(lig_path, dim=0)
            pocket_trajectory = torch.stack(pocket_path, dim=0)
            
            if self.time_axis_pos != 0:
                ligand_trajectory = ligand_trajectory.movedim(0, self.time_axis_pos)
                pocket_trajectory = pocket_trajectory.movedim(0, self.time_axis_pos)
            
            return MolecularState(
                ligand=ligand_trajectory,
                pocket=pocket_trajectory,
                ligand_mask=current_state.ligand_mask,
                pocket_mask=current_state.pocket_mask,
                batch_size=current_state.batch_size
            )


# ********************
# Molecular Sampler 
# ********************

class MolecularSampler:
    """    
    Handles molecular-specific data but exposes same interface as GenCFD.
    (will probably make this more specific to molecular stuff in time).
    """
    
    def __init__(
        self,
        ligand_sizes: list[int],
        pocket_sizes: list[int],
        scheme: MolecularDiffusion,
        denoise_fn: MolecularDenoiseFn,
        tspan: Tensor,
        apply_denoise_at_end: bool = True,
        return_full_paths: bool = False,
        n_dims: int = 3,
        atom_nf: int = 10,
        residue_nf: int = 20,
    ):
        self.ligand_sizes = ligand_sizes
        self.pocket_sizes = pocket_sizes
        self.scheme = scheme
        self.denoise_fn = denoise_fn
        self.tspan = tspan.cuda()
        self.apply_denoise_at_end = apply_denoise_at_end
        self.return_full_paths = return_full_paths
        self.n_dims = n_dims
        self.atom_nf = atom_nf
        self.residue_nf = residue_nf
        self.batch_size = len(ligand_sizes)
        
        if len(ligand_sizes) != len(pocket_sizes):
            raise ValueError("ligand_sizes and pocket_sizes must have same length")

    def generate(self, cond: TensorMapping | None = None) -> dict:
        """
        Generate molecular samples from scratch.
        
        Args:
            cond: Optional conditioning inputs
            
        Returns:
            Dictionary with molecular batch data (same format as input batches)
        """
        if self.tspan is None or self.tspan.ndim != 1:
            raise ValueError("`tspan` must be a 1-d Tensor.")
        
        # Create initial molecular state.
        molecular_state1 = self._create_initial_noisy_state()
        
        # Denoise iteratively
        denoised_state = self.denoise(
            noisy_state=molecular_state1,
            tspan=self.tspan,
            cond=cond
        )
        
        # Apply final denoising if requested.
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
                    batch_size=denoised_state.batch_size
                )
            else:
                denoised_state = final_state
        
        # Convert to molecular batch format.
        return self._molecular_state_to_batch(denoised_state, discretize_features=True)
    
    def denoise(
        self,
        noisy_state: MolecularState,
        tspan: Tensor,
        cond: TensorMapping | None = None
    ) -> MolecularState:
        raise NotImplementedError
    
    def _create_initial_noisy_state(self) -> MolecularState:
        """Create initial noisy molecular state"""
        
        total_lig_atoms = sum(self.ligand_sizes)
        total_pocket_atoms = sum(self.pocket_sizes)
        
        # Create masks
        ligand_mask = torch.cat([
            torch.full((size,), i, dtype=torch.long, device='cuda')
            for i, size in enumerate(self.ligand_sizes)
        ])
        pocket_mask = torch.cat([
            torch.full((size,), i, dtype=torch.long, device='cuda')
            for i, size in enumerate(self.pocket_sizes)
        ])
        
        # Sample initial noise (following x1 = randn() * sigma(1) * scale(1))
        initial_sigma = self.scheme.sigma(self.tspan[0])
        initial_scale = self.scheme.scale(self.tspan[0])
        
        ligand_noise = (
            torch.randn(total_lig_atoms, self.n_dims + self.atom_nf, device='cuda') 
            * initial_sigma * initial_scale
        )
        pocket_noise = (
            torch.randn(total_pocket_atoms, self.n_dims + self.residue_nf, device='cuda')
            * initial_sigma * initial_scale
        )
        
        # Make noise COM-free
        noise_coords_combined = torch.cat([
            ligand_noise[:, :self.n_dims],
            pocket_noise[:, :self.n_dims]
        ], dim=0)
        combined_mask = torch.cat([ligand_mask, pocket_mask])
        noise_coords_centered = remove_mean_batch(noise_coords_combined, combined_mask)
        
        ligand_noise[:, :self.n_dims] = noise_coords_centered[:total_lig_atoms]
        pocket_noise[:, :self.n_dims] = noise_coords_centered[total_lig_atoms:]
        
        return MolecularState(
            ligand=ligand_noise,
            pocket=pocket_noise,
            ligand_mask=ligand_mask,
            pocket_mask=pocket_mask,
            batch_size=self.batch_size
        )
    
    def _apply_final_denoise(self, state: MolecularState, cond: TensorMapping | None) -> MolecularState:
        """Apply final denoising"""
        
        # Get final time and noise level
        final_t = self.tspan[-1]
        final_sigma = self.scheme.sigma(final_t)
        final_scale = self.scheme.scale(final_t)
        
        # Handle trajectory case
        if state.ligand.dim() == 3:  # [time, atoms, features]
            current_state = MolecularState(
                ligand=state.ligand[-1],
                pocket=state.pocket[-1],
                ligand_mask=state.ligand_mask,
                pocket_mask=state.pocket_mask,
                batch_size=state.batch_size
            )
        else:
            current_state = state
        
        # Apply final denoising
        denoised_state = self.denoise_fn(current_state, final_sigma, cond)
        return denoised_state
    
    def _molecular_state_to_batch(self, state: MolecularState, discretize_features: bool = False) -> dict:
        """Convert molecular state back to batch format"""
        
        # Handle trajectory case
        if state.ligand.dim() == 3:  # [time, atoms, features]
            ligand_data = state.ligand[-1]
            pocket_data = state.pocket[-1]
        else:
            ligand_data = state.ligand
            pocket_data = state.pocket
        
        # Split coordinates and features
        lig_coords = ligand_data[:, :self.n_dims]
        lig_features = ligand_data[:, self.n_dims:]
        pocket_coords = pocket_data[:, :self.n_dims]
        pocket_features = pocket_data[:, self.n_dims:]
        
        # Unnormalize
        lig_coords_unnorm, lig_features_unnorm, pocket_coords_unnorm, pocket_features_unnorm = \
            self.scheme.unnormalize_molecular_data(
                lig_coords, lig_features, pocket_coords, pocket_features,
                discretize_features=discretize_features,
                atom_nf=self.atom_nf, residue_nf=self.residue_nf
            )
        
        return {
            'ligand_coords': lig_coords_unnorm,
            'ligand_features': lig_features_unnorm,
            'pocket_coords': pocket_coords_unnorm,
            'pocket_features': pocket_features_unnorm,
            'ligand_mask': state.ligand_mask,
            'pocket_mask': state.pocket_mask,
            'batch_size': state.batch_size
        }


# ********************
# Molecular SDE Sampler
# ********************

class MolecularSdeSampler(MolecularSampler):
    """
    Molecular version of the GenCFD SdeSampler.
    
    Implements exact reverse SDE dynamics with automatic differentiation
    but for molecular systems.
    """
    
    def __init__(
        self,
        ligand_sizes: list[int],
        pocket_sizes: list[int],
        scheme: MolecularDiffusion,
        denoise_fn: MolecularDenoiseFn,
        tspan: Tensor,
        integrator: MolecularEulerMaruyama = None,
        apply_denoise_at_end: bool = True,
        return_full_paths: bool = False,
        n_dims: int = 3,
        atom_nf: int = 10,
        residue_nf: int = 20,
    ):
        super().__init__(
            ligand_sizes=ligand_sizes,
            pocket_sizes=pocket_sizes,
            scheme=scheme,
            denoise_fn=denoise_fn,
            tspan=tspan,
            apply_denoise_at_end=apply_denoise_at_end,
            return_full_paths=return_full_paths,
            n_dims=n_dims,
            atom_nf=atom_nf,
            residue_nf=residue_nf,
        )
        
        self.integrator = integrator or MolecularEulerMaruyama(
            terminal_only=not return_full_paths,
            n_dims=n_dims
        )
        
        if self.integrator.terminal_only and self.return_full_paths:
            raise ValueError(
                f"Integrator does not support returning full paths."
            )

    def denoise(
        self,
        noisy_state: MolecularState,
        tspan: Tensor,
        cond: TensorMapping | None = None
    ) -> MolecularState:
        """Apply iterative denoising using SDE integration"""
        
        # Create SDE dynamics 
        dynamics = self._create_molecular_dynamics(cond)
        
        # Set up parameters 
        params = dict(
            drift=dict(cond=cond),
            diffusion=dict()
        )
        
        # Solve the reverse SDE
        result = self.integrator(
            dynamics=dynamics,
            molecular_state0=noisy_state,
            tspan=tspan,
            params=params
        )
        
        return result
    
    def _create_molecular_dynamics(self, cond: TensorMapping | None) -> MolecularSdeDynamics:
        """
        Create molecular SDE dynamics.
        
        Implements the reverse SDE formula:
        dx = [2σ̇/σ + ṡ/s]x - [2sσ̇/σ]D(x/s, σ) dt + s√[2σ̇σ] dW
        """
        
        def _molecular_drift(
            molecular_state: MolecularState,
            t: Tensor,
            params: MolecularParams,
        ) -> MolecularState:
            """Molecular drift"""
            
            assert t.ndim == 0, "`t` must be a scalar."
            
            # Ensure t requires gradients FIRST.
            if not t.requires_grad:
                t = t.requires_grad_(True)
            
            # Get scheme values
            s = self.scheme.scale(t)
            sigma = self.scheme.sigma(t)
            
            # Compute derivatives using automatic differentiation.
            dlog_sigma_dt = dlog_dt(self.scheme.sigma)(t)
            dlog_s_dt = 0#dlog_dt(self.scheme.scale)(t)
            
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
            """Molecular diffusion following GenCFD diffusion computation"""
            
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
                # Expand to match molecular state shapes
                diff_lig = diffusion_coeff.expand_as(molecular_state.ligand)
                diff_pocket = diffusion_coeff.expand_as(molecular_state.pocket)
            else:
                # Scalar case
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


# ********************
# Molecular Denoiser Wrapper
# ********************

def create_molecular_denoiser_wrapper(egnn_model, scheme) -> MolecularDenoiseFn:
    """    
    Wraps the EGNN to handle MolecularState inputs/outputs while maintaining
    the same mathematical interface as the GenCFD framework.
    """
    
    def molecular_denoiser(
        molecular_state: MolecularState,
        sigma: Tensor,
        cond: TensorMapping | None = None
    ) -> MolecularState:
        
        # Convert sigma to time for EGNN
        if isinstance(sigma, Tensor) and sigma.numel() == 1:
            sigma = sigma.item()
        
        # Create time input for EGNN (normalized by scheme's sigma_max)
        t_normalized = torch.full(
            (molecular_state.batch_size, 1), 
            sigma / scheme.sigma_max, 
            device=molecular_state.ligand.device
        )
        
        # Forward pass through EGNN
        with torch.no_grad():
            pred_noise_lig, pred_noise_pocket = egnn_model(
                molecular_state.ligand,
                molecular_state.pocket,
                t_normalized,
                molecular_state.ligand_mask,
                molecular_state.pocket_mask
            )
        
        return MolecularState(
            ligand=pred_noise_lig,
            pocket=pred_noise_pocket,
            ligand_mask=molecular_state.ligand_mask,
            pocket_mask=molecular_state.pocket_mask,
            batch_size=molecular_state.batch_size
        )
    
    return molecular_denoiser


def create_molecular_sampler_from_model(
    model,
    ligand_sizes: list[int],
    pocket_sizes: list[int],
    num_steps: int = 50,
    schedule_type: str = "exponential",
    return_full_paths: bool = False
) -> MolecularSdeSampler:
    """Create a molecular sampler from a trained model"""
    
    # Create time span 
    tspan = torch.linspace(1.0, 0.0, num_steps + 1, device='cuda')
    
    # Create denoiser wrapper - now passing the scheme for proper normalization
    molecular_denoise_fn = create_molecular_denoiser_wrapper(model.denoiser, model.scheme)
    
    # Create sampler
    sampler = MolecularSdeSampler(
        ligand_sizes=ligand_sizes,
        pocket_sizes=pocket_sizes,
        scheme=model.scheme,
        denoise_fn=molecular_denoise_fn,
        tspan=tspan,
        apply_denoise_at_end=True,
        return_full_paths=return_full_paths,
        n_dims=model.n_dims,
        atom_nf=model.atom_nf,
        residue_nf=model.residue_nf
    )
    
    return sampler


def load_checkpoint(checkpoint_path: str) -> MolecularDenoisingModel:
    """Load model from checkpoint"""
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # Recreate model 
    model_params = checkpoint['model_params']
    
    # Recreate the diffusion scheme from parameters
    from molecular_diffusion import molecular_exponential_noise_schedule, MolecularDiffusion
    
    scheme_params = model_params['scheme_params']
    sigma_schedule = molecular_exponential_noise_schedule(
        clip_max=scheme_params['sigma_max'],
        base=np.e**0.5,
        start=0.0,
        end=5.0
    )
    
    scheme = MolecularDiffusion.create_variance_exploding(
        sigma=sigma_schedule,
        coord_norm=scheme_params['coord_norm'],
        feature_norm=scheme_params['feature_norm'],
        feature_bias=scheme_params['feature_bias']
    )
    
    # Create noise sampling and weighting
    from molecular_diffusion import molecular_log_uniform_sampling, molecular_edm_weighting
    
    noise_sampling = molecular_log_uniform_sampling(
        scheme=scheme,
        clip_min=scheme_params['sigma_min'],
        uniform_grid=False
    )
    
    noise_weighting = molecular_edm_weighting(data_std=1.0)
    
    model = MolecularDenoisingModel(
        atom_nf=model_params['atom_nf'],
        residue_nf=model_params['residue_nf'],
        n_dims=model_params['n_dims'],
        joint_nf=model_params['joint_nf'],
        hidden_nf=model_params['hidden_nf'],
        n_layers=model_params['n_layers'],
        edge_embedding_dim=model_params['edge_embedding_dim'],
        update_pocket_coords=model_params['update_pocket_coords'],
        scheme=scheme,
        noise_sampling=noise_sampling,
        noise_weighting=noise_weighting
    )
    
    model.initialize()
    model.denoiser.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Model loaded successfully!")
    return model



def test_molecular_samplers(model, num_samples: int = 3):
    """Test the molecular SDE sampler"""
    
    print("Testing Molecular SDE Sampler (CUDA)...")
    
    # Create dummy model for testing

    # atom_nf = 4
    # residue_nf = 5
    
    # model = MolecularDenoisingModel(
    #     atom_nf=atom_nf,
    #     residue_nf=residue_nf,
    #     joint_nf=8,
    #     hidden_nf=16,
    #     n_layers=1
    # )
    # model.initialize()

    lig_size_range = (8, 24)
    pocket_size_range = (8, 20)

    # Create sampler
    ligand_sizes = np.random.randint(lig_size_range[0], lig_size_range[1], num_samples)
    pocket_sizes = np.random.randint(pocket_size_range[0], pocket_size_range[1], num_samples)
    
    sampler = create_molecular_sampler_from_model(
        model=model,
        ligand_sizes=ligand_sizes,
        pocket_sizes=pocket_sizes,
        num_steps=10
    )
    
    # Generate samples
    print("Generating samples...")
    samples = sampler.generate()

    print(samples)
    
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
    
    # Show some statistics
    ligand_types = torch.argmax(samples['ligand_features'], dim=1)
    pocket_types = torch.argmax(samples['pocket_features'], dim=1)
    
    print(f"  Ligand atom type distribution: {torch.bincount(ligand_types)}")
    print(f"  Pocket residue type distribution: {torch.bincount(pocket_types)}")
    
    print("✅ Molecular SDE sampler test passed!")
    
    return sampler, samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=3)
    args = parser.parse_args()

    model = load_checkpoint(args.model_path)

    _, samples = test_molecular_samplers(model, args.n_samples)

    for i in range(args.n_samples):
        graph = Data(
            ligand_coords=samples['ligand_coords'][samples['ligand_mask']==i].cpu(),
            ligand_features=samples['ligand_features'][samples['ligand_mask']==i].cpu(),
            pocket_coords=samples['pocket_coords'][samples['pocket_mask']==i].cpu(),
            pocket_features=samples['pocket_features'][samples['pocket_mask']==i].cpu()
        )
        torch.save(graph, args.model_path.replace(".pt", f"_sample_{i}.pt"))
    
