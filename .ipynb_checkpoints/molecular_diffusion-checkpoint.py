"""
Molecular Diffusion Schemes and Training

Implements the same diffusion logic as the base framework but natively for molecular systems.
All operations are CUDA-optimized with proper noise schedules, sampling, and weighting.
"""

import dataclasses
import torch
import numpy as np
from typing import Callable, Tuple, Optional, Protocol
from torch_scatter import scatter_mean

from egnn_dynamics import EGNNDynamics

Tensor = torch.Tensor
Numeric = float | int | Tensor
ScheduleFn = Callable[[Numeric], Numeric]

EPS = 1e-4
MIN_DIFFUSION_TIME = EPS
MAX_DIFFUSION_TIME = 1.0 - EPS


def remove_mean_batch(x: Tensor, indices: Tensor) -> Tensor:
    """Remove center of mass for each molecule in batch (CUDA)"""
    mean = scatter_mean(x, indices, dim=0)
    return x - mean[indices]


# ********************
# Molecular Invertible Schedules
# ********************

@dataclasses.dataclass(frozen=True)
class MolecularInvertibleSchedule:
    """Molecular version of InvertibleSchedule - same logic, molecular-native"""
    
    forward: ScheduleFn
    inverse: ScheduleFn
    device: str = 'cuda'
    
    def __call__(self, t: Numeric) -> Tensor:
        result = self.forward(t)
        if isinstance(result, (int, float)):
            result = torch.tensor(result, device=self.device, dtype=torch.float32)
        return result.to(self.device)


def _molecular_linear_rescale(in_min: float, in_max: float, out_min: float, out_max: float) -> MolecularInvertibleSchedule:
    in_range = in_max - in_min
    out_range = out_max - out_min
    fwd = lambda x: out_min + (x - in_min) / in_range * out_range
    inv = lambda y: in_min + (y - out_min) / out_range * in_range
    return MolecularInvertibleSchedule(fwd, inv, device='cuda')


# ********************
# Molecular Noise Schedulers 
# ********************

def molecular_exponential_noise_schedule(
    clip_max: float = 80.0,
    base: float = np.e**0.5,
    start: float = 0.0,
    end: float = 5.0,
) -> MolecularInvertibleSchedule:
    
    if not (start < end and base > 1.0):
        raise ValueError("Must have `base` > 1 and `start` < `end`.")

    in_rescale = _molecular_linear_rescale(
        in_min=MIN_DIFFUSION_TIME,
        in_max=MAX_DIFFUSION_TIME,
        out_min=start,
        out_max=end,
    )
    out_rescale = _molecular_linear_rescale(
        in_min=base**start, in_max=base**end, out_min=0.0, out_max=clip_max
    )

    def sigma(t):
        t_tensor = torch.as_tensor(t, device='cuda', dtype=torch.float32)
        base_tensor = torch.tensor(base, device='cuda', dtype=torch.float32)
        return out_rescale(torch.pow(base_tensor, in_rescale(t_tensor)))
    
    def inverse(y):
        y_tensor = torch.as_tensor(y, device='cuda', dtype=torch.float32)
        base_tensor = torch.tensor(base, device='cuda', dtype=torch.float32)
        return in_rescale.inverse(
            torch.log(out_rescale.inverse(y_tensor)) / torch.log(base_tensor)
        )
    
    return MolecularInvertibleSchedule(sigma, inverse, device='cuda')


def molecular_power_noise_schedule(
    clip_max: float = 80.0,
    p: float = 1.0,
    start: float = 0.0,
    end: float = 1.0,
) -> MolecularInvertibleSchedule:
    
    if not (0 <= start < end and p > 0):
        raise ValueError("Must have `p` > 0 and 0 <= `start` < `end`.")

    in_rescale = _molecular_linear_rescale(
        in_min=MIN_DIFFUSION_TIME, in_max=MAX_DIFFUSION_TIME, out_min=start, out_max=end
    )
    out_rescale = _molecular_linear_rescale(
        in_min=start**p, in_max=end**p, out_min=0.0, out_max=clip_max
    )

    def sigma(t):
        t_tensor = torch.as_tensor(t, device='cuda', dtype=torch.float32)
        return out_rescale(torch.pow(in_rescale(t_tensor), p))
    
    def inverse(y):
        y_tensor = torch.as_tensor(y, device='cuda', dtype=torch.float32)
        return in_rescale.inverse(torch.pow(out_rescale.inverse(y_tensor), 1 / p))

    return MolecularInvertibleSchedule(sigma, inverse, device='cuda')


# ********************
# Molecular Diffusion Scheme
# ********************

@dataclasses.dataclass(frozen=True)
class MolecularDiffusion:
    """
    Molecular diffusion scheme.
    
    Handles the molecular perturbation kernel:
    p(lig_t, pocket_t | lig_0, pocket_0) = N(molecular_t; s_t * molecular_0, s_t * σ_t * I)
    
    With molecular-specific normalization for coordinates and categorical features.
    """
    
    scale: ScheduleFn
    sigma: MolecularInvertibleSchedule
    
    # Molecular-specific parameters
    coord_norm: float = 1.0
    feature_norm: float = 1.0
    feature_bias: float = 0.0
    device: str = 'cuda'
    
    @property
    def sigma_max(self) -> float:
        return self.sigma(MAX_DIFFUSION_TIME).item()
    
    def normalize_molecular_data(self, ligand_coords: Tensor, ligand_features: Tensor,
                                pocket_coords: Tensor, pocket_features: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Apply molecular normalization"""
        
        # Normalize coordinates
        ligand_coords_norm = ligand_coords / self.coord_norm
        pocket_coords_norm = pocket_coords / self.coord_norm
        
        # Normalize categorical features (one-hot -> continuous)
        ligand_features_norm = (ligand_features.float() - self.feature_bias) / self.feature_norm
        pocket_features_norm = (pocket_features.float() - self.feature_bias) / self.feature_norm
        
        return ligand_coords_norm, ligand_features_norm, pocket_coords_norm, pocket_features_norm
    
    def unnormalize_molecular_data(self, ligand_coords: Tensor, ligand_features: Tensor,
                                  pocket_coords: Tensor, pocket_features: Tensor,
                                  discretize_features: bool = False, 
                                  atom_nf: int = 10, residue_nf: int = 20) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Unnormalize molecular data"""
        
        # Unnormalize coordinates
        ligand_coords_unnorm = ligand_coords * self.coord_norm
        pocket_coords_unnorm = pocket_coords * self.coord_norm
        
        # Unnormalize and optionally discretize features
        ligand_features_unnorm = ligand_features * self.feature_norm + self.feature_bias
        pocket_features_unnorm = pocket_features * self.feature_norm + self.feature_bias
        
        if discretize_features:
            ligand_features_unnorm = torch.nn.functional.one_hot(
                torch.argmax(ligand_features_unnorm, dim=1), atom_nf
            ).float()
            pocket_features_unnorm = torch.nn.functional.one_hot(
                torch.argmax(pocket_features_unnorm, dim=1), residue_nf
            ).float()
        
        return ligand_coords_unnorm, ligand_features_unnorm, pocket_coords_unnorm, pocket_features_unnorm

    @classmethod
    def create_variance_exploding(
        cls,
        sigma: MolecularInvertibleSchedule,
        coord_norm: float = 1.0,
        feature_norm: float = 1.0,
        feature_bias: float = 0.0,
    ) -> 'MolecularDiffusion':
        """Create variance exploding scheme"""
        
        def scale(t):
            return torch.ones_like(torch.as_tensor(t, device='cuda', dtype=torch.float32))
        
        return cls(
            scale=scale,
            sigma=sigma,
            coord_norm=coord_norm,
            feature_norm=feature_norm,
            feature_bias=feature_bias,
            device='cuda'
        )


# ********************
# Molecular Noise Sampling
# ********************

class MolecularNoiseLevelSampling(Protocol):
    """Protocol for molecular noise sampling"""
    def __call__(self, shape: Tuple[int, ...]) -> Tensor: ...


def molecular_log_uniform_sampling(
    scheme: MolecularDiffusion,
    clip_min: float = 1e-4,
    uniform_grid: bool = False,
) -> MolecularNoiseLevelSampling:

    def _noise_sampling(shape: Tuple[int, ...]) -> Tensor:
        if uniform_grid:
            s0 = torch.rand((), dtype=torch.float32, device='cuda')
            num_elements = int(np.prod(shape))
            step_size = 1 / num_elements
            grid = torch.linspace(0, 1 - step_size, num_elements, dtype=torch.float32, device='cuda')
            samples = torch.remainder(grid + s0, 1).reshape(shape)
        else:
            samples = torch.rand(shape, dtype=torch.float32, device='cuda')
        
        log_min = torch.log(torch.tensor(clip_min, dtype=torch.float32, device='cuda'))
        log_max = torch.log(torch.tensor(scheme.sigma_max, dtype=torch.float32, device='cuda'))
        samples = (log_max - log_min) * samples + log_min
        return torch.exp(samples)

    return _noise_sampling


def molecular_time_uniform_sampling(
    scheme: MolecularDiffusion,
    clip_min: float = 1e-4,
    uniform_grid: bool = False,
) -> MolecularNoiseLevelSampling:

    def _noise_sampling(shape: Tuple[int, ...]) -> Tensor:
        if uniform_grid:
            s0 = torch.rand((), dtype=torch.float32, device='cuda')
            num_elements = int(np.prod(shape))
            step_size = 1 / num_elements
            grid = torch.linspace(0, 1 - step_size, num_elements, dtype=torch.float32, device='cuda')
            samples = torch.remainder(grid + s0, 1).reshape(shape)
        else:
            samples = torch.rand(shape, dtype=torch.float32, device='cuda')
        
        min_t = scheme.sigma.inverse(clip_min)
        samples = (MAX_DIFFUSION_TIME - min_t) * samples + min_t
        return scheme.sigma(samples)

    return _noise_sampling


# ********************
# Molecular Noise Weighting 
# ********************

class MolecularNoiseLossWeighting(Protocol):
    def __call__(self, sigma: Tensor) -> Tensor: ...


def molecular_edm_weighting(data_std: float = 1.0) -> MolecularNoiseLossWeighting:

    def _weight_fn(sigma: Tensor) -> Tensor:
        data_std_tensor = torch.tensor(data_std, device=sigma.device, dtype=sigma.dtype)
        return (torch.square(data_std_tensor) + torch.square(sigma)) / torch.square(data_std_tensor * sigma)

    return _weight_fn


def molecular_uniform_weighting() -> MolecularNoiseLossWeighting:
    """Uniform weighting for molecular diffusion"""
    def _weight_fn(sigma: Tensor) -> Tensor:
        return torch.ones_like(sigma)
    return _weight_fn


# ********************
# Molecular Training Model
# ********************

@dataclasses.dataclass(frozen=True, kw_only=True)
class MolecularDenoisingModel:
    """
    Molecular denoising model for training.
    """
    
    # Molecular structure parameters
    atom_nf: int = 10
    residue_nf: int = 20
    n_dims: int = 3
    
    # EGNN parameters
    joint_nf: int = 16
    hidden_nf: int = 64
    n_layers: int = 4
    edge_embedding_dim: Optional[int] = 8
    update_pocket_coords: bool = True
    
    # Diffusion scheme
    scheme: MolecularDiffusion = None
    noise_sampling: MolecularNoiseLevelSampling = None
    noise_weighting: MolecularNoiseLossWeighting = None
    
    # Evaluation parameters
    num_eval_noise_levels: int = 5
    num_eval_cases_per_lvl: int = 1
    
    def __post_init__(self):
        """Initialize the denoiser and diffusion components"""
        
        # Create default scheme if not provided
        if self.scheme is None:
            sigma_schedule = molecular_exponential_noise_schedule(clip_max=80.0)
            scheme = MolecularDiffusion.create_variance_exploding(
                sigma=sigma_schedule,
                coord_norm=1.0,
                feature_norm=1.0,
                feature_bias=0.0
            )
            object.__setattr__(self, 'scheme', scheme)
        
        # Create default noise sampling if not provided
        if self.noise_sampling is None:
            noise_sampling = molecular_log_uniform_sampling(self.scheme)
            object.__setattr__(self, 'noise_sampling', noise_sampling)
        
        # Create default noise weighting if not provided
        if self.noise_weighting is None:
            noise_weighting = molecular_edm_weighting(data_std=1.0)
            object.__setattr__(self, 'noise_weighting', noise_weighting)
        
        # Create the EGNN denoiser
        denoiser = EGNNDynamics(
            atom_nf=self.atom_nf,
            residue_nf=self.residue_nf,
            n_dims=self.n_dims,
            joint_nf=self.joint_nf,
            hidden_nf=self.hidden_nf,
            device='cuda',
            n_layers=self.n_layers,
            condition_time=True,
            update_pocket_coords=self.update_pocket_coords,
            edge_embedding_dim=self.edge_embedding_dim
        )
        object.__setattr__(self, 'denoiser', denoiser)

    def initialize(self):
        """Initialize model weights"""
        print(f"✅ Initialized MolecularDenoisingModel with {sum(p.numel() for p in self.denoiser.parameters())} parameters")

    def loss_fn(self, batch: dict):
        """
        Compute molecular denoising loss.
        
        Args:
            batch: Molecular batch with ligand/pocket coords, features, masks
            
        Returns:
            loss: Training loss
            metrics: Dictionary of training metrics
        """
        
        # Extract molecular data (all on CUDA)
        lig_coords = batch['ligand_coords'].cuda()
        lig_features = batch['ligand_features'].cuda()
        pocket_coords = batch['pocket_coords'].cuda()
        pocket_features = batch['pocket_features'].cuda()
        lig_mask = batch['ligand_mask'].cuda()
        pocket_mask = batch['pocket_mask'].cuda()
        batch_size = batch['batch_size']
        
        # Normalize molecular data
        lig_coords_norm, lig_features_norm, pocket_coords_norm, pocket_features_norm = \
            self.scheme.normalize_molecular_data(lig_coords, lig_features, pocket_coords, pocket_features)
        
        # Remove center of mass for translation invariance
        combined_coords = torch.cat([lig_coords_norm, pocket_coords_norm], dim=0)
        combined_mask = torch.cat([lig_mask, pocket_mask], dim=0)
        combined_coords_centered = remove_mean_batch(combined_coords, combined_mask)
        
        lig_coords_centered = combined_coords_centered[:len(lig_coords)]
        pocket_coords_centered = combined_coords_centered[len(lig_coords):]
        
        # Concatenate coordinates and features
        xh_lig_clean = torch.cat([lig_coords_centered, lig_features_norm], dim=1)
        xh_pocket_clean = torch.cat([pocket_coords_centered, pocket_features_norm], dim=1)
        
        # Sample noise levels.
        sigma = self.noise_sampling(shape=(batch_size,))  # [batch_size]
        
        # Apply diffusion scheme.
        scale = self.scheme.scale(sigma)
        
        # Broadcast to all atoms/residues
        sigma_lig = sigma[lig_mask]
        sigma_pocket = sigma[pocket_mask]
        scale_lig = scale[lig_mask] if isinstance(scale, Tensor) and scale.numel() > 1 else scale
        scale_pocket = scale[pocket_mask] if isinstance(scale, Tensor) and scale.numel() > 1 else scale
        
        # Sample Gaussian noise with COM removal
        noise_lig = torch.randn_like(xh_lig_clean)
        noise_pocket = torch.randn_like(xh_pocket_clean)
        
        # Make coordinate noise COM-free
        noise_coords_combined = torch.cat([noise_lig[:, :self.n_dims], noise_pocket[:, :self.n_dims]], dim=0)
        noise_coords_centered = remove_mean_batch(noise_coords_combined, combined_mask)
        
        noise_lig[:, :self.n_dims] = noise_coords_centered[:len(lig_coords)]
        noise_pocket[:, :self.n_dims] = noise_coords_centered[len(lig_coords):]
        
        # Apply perturbation kernel: x_t = scale * x_0 + sigma * eps
        if isinstance(scale_lig, Tensor):
            scale_lig = scale_lig.unsqueeze(1)
            scale_pocket = scale_pocket.unsqueeze(1)
        
        xh_lig_noisy = scale_lig * xh_lig_clean + sigma_lig.unsqueeze(1) * noise_lig
        xh_pocket_noisy = scale_pocket * xh_pocket_clean + sigma_pocket.unsqueeze(1) * noise_pocket
        
        # Normalize time for denoiser
        t_normalized = (sigma / self.scheme.sigma_max).unsqueeze(1)  # [batch_size, 1]
        
        # Forward pass through denoiser
        pred_noise_lig, pred_noise_pocket = self.denoiser(
            xh_lig_noisy, xh_pocket_noisy, t_normalized, lig_mask, pocket_mask
        )
        
        # Compute loss weights
        weights = self.noise_weighting(sigma)  # [batch_size]
        weights_lig = weights[lig_mask]
        weights_pocket = weights[pocket_mask]
        
        # Compute weighted L2 loss
        loss_lig = torch.mean(weights_lig.unsqueeze(1) * (pred_noise_lig - noise_lig) ** 2)
        loss_pocket = torch.mean(weights_pocket.unsqueeze(1) * (pred_noise_pocket - noise_pocket) ** 2)
        
        total_loss = loss_lig + loss_pocket
        
        # Metrics
        metrics = {
            "loss": total_loss.item(),
            "loss_ligand": loss_lig.item(),
            "loss_pocket": loss_pocket.item(),
            "avg_sigma": sigma.mean().item(),
            "avg_scale": scale.mean().item() if isinstance(scale, Tensor) else scale
        }
        
        return total_loss, metrics

    def eval_fn(self, batch: dict) -> dict:
        """Evaluate denoising at multiple noise levels"""
        
        # Extract and normalize data
        lig_coords = batch['ligand_coords'].cuda()
        lig_features = batch['ligand_features'].cuda()
        pocket_coords = batch['pocket_coords'].cuda()
        pocket_features = batch['pocket_features'].cuda()
        lig_mask = batch['ligand_mask'].cuda()
        pocket_mask = batch['pocket_mask'].cuda()
        batch_size = batch['batch_size']
        
        # Test at multiple fixed noise levels.
        sigma_levels = torch.logspace(
            np.log10(1e-3), np.log10(self.scheme.sigma_max), 
            self.num_eval_noise_levels, device='cuda'
        )
        
        eval_losses = {}
        
        for i, sigma_val in enumerate(sigma_levels):
            # Prepare clean data (same as training)
            lig_coords_norm, lig_features_norm, pocket_coords_norm, pocket_features_norm = \
                self.scheme.normalize_molecular_data(lig_coords, lig_features, pocket_coords, pocket_features)
            
            combined_coords = torch.cat([lig_coords_norm, pocket_coords_norm], dim=0)
            combined_mask = torch.cat([lig_mask, pocket_mask], dim=0)
            combined_coords_centered = remove_mean_batch(combined_coords, combined_mask)
            
            lig_coords_centered = combined_coords_centered[:len(lig_coords)]
            pocket_coords_centered = combined_coords_centered[len(lig_coords):]
            
            xh_lig_clean = torch.cat([lig_coords_centered, lig_features_norm], dim=1)
            xh_pocket_clean = torch.cat([pocket_coords_centered, pocket_features_norm], dim=1)
            
            # Add noise at this level
            scale = self.scheme.scale(sigma_val)
            noise_lig = torch.randn_like(xh_lig_clean)
            noise_pocket = torch.randn_like(xh_pocket_clean)
            
            # COM-free noise
            noise_coords_combined = torch.cat([noise_lig[:, :self.n_dims], noise_pocket[:, :self.n_dims]], dim=0)
            noise_coords_centered = remove_mean_batch(noise_coords_combined, combined_mask)
            noise_lig[:, :self.n_dims] = noise_coords_centered[:len(lig_coords)]
            noise_pocket[:, :self.n_dims] = noise_coords_centered[len(lig_coords):]
            
            xh_lig_noisy = scale * xh_lig_clean + sigma_val * noise_lig
            xh_pocket_noisy = scale * xh_pocket_clean + sigma_val * noise_pocket
            
            # Denoise
            t_normalized = (sigma_val / self.scheme.sigma_max).unsqueeze(0).expand(batch_size, 1)
            
            with torch.no_grad():
                pred_noise_lig, pred_noise_pocket = self.denoiser(
                    xh_lig_noisy, xh_pocket_noisy, t_normalized, lig_mask, pocket_mask
                )
            
            # Compute evaluation loss
            loss_lig = torch.mean((pred_noise_lig - noise_lig) ** 2)
            loss_pocket = torch.mean((pred_noise_pocket - noise_pocket) ** 2)
            total_loss = loss_lig + loss_pocket
            
            eval_losses[f"denoise_lvl{i}"] = total_loss.item()
        
        return eval_losses


def test_molecular_diffusion():
    """Test the molecular diffusion implementation"""
    
    print("Testing Molecular Diffusion (CUDA)...")
    
    # Create test batch (all on CUDA)
    device = 'cuda'
    batch_size = 2
    atom_nf = 5
    residue_nf = 7
    
    batch = {
        'ligand_coords': torch.randn(8, 3, device=device),
        'ligand_features': torch.nn.functional.one_hot(
            torch.randint(0, atom_nf, (8,), device=device), atom_nf
        ).float(),
        'pocket_coords': torch.randn(16, 3, device=device),
        'pocket_features': torch.nn.functional.one_hot(
            torch.randint(0, residue_nf, (16,), device=device), residue_nf
        ).float(),
        'ligand_mask': torch.cat([torch.zeros(4), torch.ones(4)]).long().to(device),
        'pocket_mask': torch.cat([torch.zeros(8), torch.ones(8)]).long().to(device),
        'batch_size': 2
    }
    
    # Create model
    model = MolecularDenoisingModel(
        atom_nf=atom_nf,
        residue_nf=residue_nf,
        joint_nf=8,
        hidden_nf=32,
        n_layers=2
    )
    
    model.initialize()
    
    # Test loss computation
    loss, metrics = model.loss_fn(batch)
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # Test evaluation
    eval_metrics = model.eval_fn(batch)
    print(f"Eval metrics: {eval_metrics}")
    
    print("✅ Molecular diffusion test passed!")
    
    return model


if __name__ == "__main__":
    test_molecular_diffusion()