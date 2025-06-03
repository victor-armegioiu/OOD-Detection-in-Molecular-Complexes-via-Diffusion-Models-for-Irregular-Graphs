"""
Molecular diffusion pipeline (GenCFD-style with categorical features).
"""

import dataclasses
import torch
import torch.nn.functional as F
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
# Molecular Invertible Schedules (same as before)
# ********************

@dataclasses.dataclass(frozen=True)
class MolecularInvertibleSchedule:
    """Molecular version of InvertibleSchedule"""
    
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


def molecular_exponential_noise_schedule(
    clip_max: float = 10.0,  # Reduced from 80.0
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


# ********************
# Diffusion Scheme (Simplified)
# ********************

@dataclasses.dataclass(frozen=True)
class MolecularDiffusion:
    """
    Simplified GenCFD-style diffusion for molecular data.
    """
    
    sigma: MolecularInvertibleSchedule
    
    # Molecular-specific parameters
    coord_norm: float = 1.0
    feature_norm: float = 1.0
    feature_bias: float = 0.0
    categorical_temperature: float = 1.0
    
    device: str = 'cuda'
    
    @property
    def sigma_max(self) -> float:
        return self.sigma(MAX_DIFFUSION_TIME).item()
    
    def scale(self, t):
        """Scale function for compatibility with samplers (always returns 1.0 for variance exploding)"""
        return torch.ones_like(torch.as_tensor(t, device=self.device, dtype=torch.float32))
    
    def normalize_molecular_data(self, ligand_coords: Tensor, ligand_features: Tensor,
                                pocket_coords: Tensor, pocket_features: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Apply molecular normalization (kept for compatibility)"""
        
        # Normalize coordinates
        ligand_coords_norm = ligand_coords / self.coord_norm
        pocket_coords_norm = pocket_coords / self.coord_norm
        
        # Normalize categorical features (one-hot -> continuous centered around 0)
        ligand_features_norm = (ligand_features.float() - self.feature_bias) / self.feature_norm
        pocket_features_norm = (pocket_features.float() - self.feature_bias) / self.feature_norm
        
        return ligand_coords_norm, ligand_features_norm, pocket_coords_norm, pocket_features_norm

    def unnormalize_molecular_data(self, ligand_coords: Tensor, ligand_features: Tensor,
                                  pocket_coords: Tensor, pocket_features: Tensor,
                                  discretize_features: bool = False, 
                                  atom_nf: int = 10, residue_nf: int = 20) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Unnormalize molecular data (kept for compatibility)"""
        
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
        categorical_temperature: float = 1.0,
    ) -> 'MolecularDiffusion':
        """Create variance exploding scheme"""
        
        return cls(
            sigma=sigma,
            coord_norm=coord_norm,
            feature_norm=feature_norm,
            feature_bias=feature_bias,
            categorical_temperature=categorical_temperature,
            device='cuda'
        )


# ********************
# Noise Sampling and Weighting 
# ********************

class MolecularNoiseLevelSampling(Protocol):
    def __call__(self, shape: Tuple[int, ...]) -> Tensor: ...


def molecular_log_uniform_sampling(
    scheme: MolecularDiffusion,
    clip_min: float = 1e-3,  # Increased from 1e-4
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


class MolecularNoiseLossWeighting(Protocol):
    def __call__(self, sigma: Tensor) -> Tensor: ...

def molecular_edm_weighting(
    data_std: float = 1.0, device=None
) -> MolecularNoiseLossWeighting:
    """Weighting proposed in Karras et al. (https://arxiv.org/abs/2206.00364).

    This weighting ensures the effective weights are uniform across noise levels
    (see appendix B.6, eqns 139 to 144).

    Args:
      data_std: the standard deviation of the data.

    Returns:
      The weighting function.
    """

    def _weight_fn(sigma: Tensor) -> Tensor:
        return (
            torch.square(torch.tensor(data_std, device=device)) + torch.square(sigma)
        ) / torch.square(data_std * sigma)

    return _weight_fn


# ********************
# Training Model (GenCFD-Style)
# ********************

@dataclasses.dataclass(frozen=True, kw_only=True)
class MolecularDenoisingModel:
    
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
    
    # Loss weighting parameters
    coord_loss_weight: float = 1.0
    categorical_loss_weight: float = 1.0
    
    # Diffusion scheme
    scheme: MolecularDiffusion = None
    noise_sampling: MolecularNoiseLevelSampling = None
    noise_weighting: MolecularNoiseLossWeighting = None
    
    def __post_init__(self):
        """Initialize the denoiser and diffusion components"""
        
        # Create default scheme if not provided
        if self.scheme is None:
            sigma_schedule = molecular_exponential_noise_schedule(clip_max=10.0)  # Much smaller
            scheme = MolecularDiffusion.create_variance_exploding(
                sigma=sigma_schedule,
                coord_norm=1.0,
                feature_norm=1.0,
                feature_bias=0.0,
                categorical_temperature=1.0
            )
            object.__setattr__(self, 'scheme', scheme)
        
        # Create default noise sampling if not provided
        if self.noise_sampling is None:
            noise_sampling = molecular_log_uniform_sampling(self.scheme)
            object.__setattr__(self, 'noise_sampling', noise_sampling)
        
        # Create default noise weighting if not provided
        if self.noise_weighting is None:
            noise_weighting = molecular_edm_weighting()
            object.__setattr__(self, 'edm_noise_weighting', noise_weighting)
        
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
        GenCFD-style loss function with categorical features
        
        Args:
            batch: Molecular batch with ligand/pocket coords, features, masks
            
        Returns:
            loss: Training loss
            metrics: Dictionary of training metrics
        """
        
        # Extract molecular data (all on CUDA)
        lig_coords = batch['ligand_coords'].cuda()
        lig_features = batch['ligand_features'].cuda()  # One-hot features
        pocket_coords = batch['pocket_coords'].cuda()
        pocket_features = batch['pocket_features'].cuda()  # One-hot features
        lig_mask = batch['ligand_mask'].cuda()
        pocket_mask = batch['pocket_mask'].cuda()
        batch_size = batch['batch_size']
        
        # Store clean one-hot features for categorical loss
        lig_features_clean = lig_features.clone()
        pocket_features_clean = pocket_features.clone()
        
        # Normalize molecular data (coordinates only, keep features as-is for now)
        lig_coords_norm = lig_coords / self.scheme.coord_norm
        pocket_coords_norm = pocket_coords / self.scheme.coord_norm
        
        # Convert one-hot features to normalized continuous (for noise addition)
        lig_features_norm = (lig_features.float() - self.scheme.feature_bias) / self.scheme.feature_norm
        pocket_features_norm = (pocket_features.float() - self.scheme.feature_bias) / self.scheme.feature_norm
        
        # Remove center of mass for translation invariance
        combined_coords = torch.cat([lig_coords_norm, pocket_coords_norm], dim=0)
        combined_mask = torch.cat([lig_mask, pocket_mask], dim=0)
        combined_coords_centered = remove_mean_batch(combined_coords, combined_mask)
        
        lig_coords_centered = combined_coords_centered[:len(lig_coords)]
        pocket_coords_centered = combined_coords_centered[len(lig_coords):]
        
        # Concatenate coordinates and features (clean targets)
        xh_lig_clean = torch.cat([lig_coords_centered, lig_features_norm], dim=1)
        xh_pocket_clean = torch.cat([pocket_coords_centered, pocket_features_norm], dim=1)
        
        # Sample noise levels - GenCFD style (much simpler)
        sigma = self.noise_sampling(shape=(batch_size,))  # [batch_size]
        
        # Generate Gaussian noise for the entire state
        noise_lig = torch.randn_like(xh_lig_clean)
        noise_pocket = torch.randn_like(xh_pocket_clean)
        
        # Make coordinate noise COM-free (but not feature noise)
        noise_coords_combined = torch.cat([noise_lig[:, :self.n_dims], noise_pocket[:, :self.n_dims]], dim=0)
        noise_coords_centered = remove_mean_batch(noise_coords_combined, combined_mask)
        noise_lig[:, :self.n_dims] = noise_coords_centered[:len(lig_coords)]
        noise_pocket[:, :self.n_dims] = noise_coords_centered[len(lig_coords):]
        
        # Add noise
        sigma_expanded_lig = sigma[lig_mask].unsqueeze(1)  # [N_lig, 1]
        sigma_expanded_pocket = sigma[pocket_mask].unsqueeze(1)  # [N_pocket, 1]
        
        xh_lig_noisy = xh_lig_clean + sigma_expanded_lig * noise_lig
        xh_pocket_noisy = xh_pocket_clean + sigma_expanded_pocket * noise_pocket
        
        # Normalize time for denoiser (sigma -> [0,1])
        sigma_max = self.scheme.sigma_max
        t_normalized = (sigma / sigma_max).unsqueeze(1)  # [batch_size, 1]
        
        # Forward pass through denoiser - predict CLEAN signal (GenCFD style)
        pred_clean_lig, pred_clean_pocket = self.denoiser(
            xh_lig_noisy, xh_pocket_noisy, t_normalized, lig_mask, pocket_mask
        )
        
        # Simple weighting - GenCFD style (much simpler than EDM)
        weights = self.noise_weighting(sigma)  # [batch_size]
        weights_lig = weights[lig_mask].unsqueeze(1)  # [N_lig, 1]
        weights_pocket = weights[pocket_mask].unsqueeze(1)  # [N_pocket, 1]
        
        # Coordinate loss: L2 between predicted and clean coordinates
        pred_coords_lig = pred_clean_lig[:, :self.n_dims]
        pred_coords_pocket = pred_clean_pocket[:, :self.n_dims]
        clean_coords_lig = xh_lig_clean[:, :self.n_dims]
        clean_coords_pocket = xh_pocket_clean[:, :self.n_dims]
        
        coord_loss_lig = torch.mean(weights_lig * (pred_coords_lig - clean_coords_lig) ** 2)
        coord_loss_pocket = torch.mean(weights_pocket * (pred_coords_pocket - clean_coords_pocket) ** 2)
        coord_loss = coord_loss_lig + coord_loss_pocket
        
        # Categorical loss: Handle features specially
        pred_features_lig = pred_clean_lig[:, self.n_dims:]
        pred_features_pocket = pred_clean_pocket[:, self.n_dims:]
        clean_features_lig = xh_lig_clean[:, self.n_dims:]
        clean_features_pocket = xh_pocket_clean[:, self.n_dims:]
        
        # For categorical features, convert predictions back to logits and use cross-entropy
        # First unnormalize predictions
        pred_features_lig_unnorm = pred_features_lig * self.scheme.feature_norm + self.scheme.feature_bias
        pred_features_pocket_unnorm = pred_features_pocket * self.scheme.feature_norm + self.scheme.feature_bias
        
        # Convert to logits and apply cross-entropy with clean one-hot targets
        pred_logits_lig = pred_features_lig_unnorm / self.scheme.categorical_temperature
        pred_logits_pocket = pred_features_pocket_unnorm / self.scheme.categorical_temperature
        
        true_categories_lig = torch.argmax(lig_features_clean, dim=-1)
        true_categories_pocket = torch.argmax(pocket_features_clean, dim=-1)
        
        # Cross-entropy loss (weighted by noise level)
        ce_loss_lig = F.cross_entropy(pred_logits_lig, true_categories_lig, reduction='none')
        ce_loss_pocket = F.cross_entropy(pred_logits_pocket, true_categories_pocket, reduction='none')
        
        # Weight categorical loss by sigma (like coordinates)
        categorical_loss_lig = torch.mean(weights[lig_mask] * ce_loss_lig)
        categorical_loss_pocket = torch.mean(weights[pocket_mask] * ce_loss_pocket)
        categorical_loss = categorical_loss_lig + categorical_loss_pocket
        
        # Combine losses
        total_loss = (
            self.coord_loss_weight * coord_loss + 
            self.categorical_loss_weight * categorical_loss
        )
        
        # Metrics
        metrics = {
            "loss": total_loss.item(),
            "coord_loss": coord_loss.item(),
            "categorical_loss": categorical_loss.item(),
            "coord_loss_ligand": coord_loss_lig.item(),
            "coord_loss_pocket": coord_loss_pocket.item(),
            "categorical_loss_ligand": categorical_loss_lig.item(),
            "categorical_loss_pocket": categorical_loss_pocket.item(),
            "avg_sigma": sigma.mean().item(),
            "max_sigma": sigma.max().item(),
            "coord_pred_scale": torch.cat([pred_coords_lig, pred_coords_pocket]).abs().mean().item(),
            "feature_pred_scale_lig": pred_features_lig_unnorm.abs().mean().item(),
            "feature_pred_scale_pocket": pred_features_pocket_unnorm.abs().mean().item()
        }
        
        return total_loss, metrics

    def eval_fn(self, batch: dict) -> dict:
        """Evaluate denoising at multiple noise levels with proper loss breakdown"""
        
        # Extract and normalize data
        lig_coords = batch['ligand_coords'].cuda()
        lig_features = batch['ligand_features'].cuda()
        pocket_coords = batch['pocket_coords'].cuda()
        pocket_features = batch['pocket_features'].cuda()
        lig_mask = batch['ligand_mask'].cuda()
        pocket_mask = batch['pocket_mask'].cuda()
        batch_size = batch['batch_size']
        
        # Store clean features
        lig_features_clean = lig_features.clone()
        
        # Test at multiple fixed noise levels
        sigma_levels = torch.logspace(
            np.log10(1e-3), np.log10(self.scheme.sigma_max), 
            5, device='cuda'
        )
        
        eval_losses = {}
        
        for i, sigma_val in enumerate(sigma_levels):
            # Same forward process as training
            lig_coords_norm = lig_coords / self.scheme.coord_norm
            pocket_coords_norm = pocket_coords / self.scheme.coord_norm
            lig_features_norm = (lig_features.float() - self.scheme.feature_bias) / self.scheme.feature_norm
            pocket_features_norm = (pocket_features.float() - self.scheme.feature_bias) / self.scheme.feature_norm
            
            combined_coords = torch.cat([lig_coords_norm, pocket_coords_norm], dim=0)
            combined_mask = torch.cat([lig_mask, pocket_mask], dim=0)
            combined_coords_centered = remove_mean_batch(combined_coords, combined_mask)
            
            lig_coords_centered = combined_coords_centered[:len(lig_coords)]
            pocket_coords_centered = combined_coords_centered[len(lig_coords):]
            
            xh_lig_clean = torch.cat([lig_coords_centered, lig_features_norm], dim=1)
            xh_pocket_clean = torch.cat([pocket_coords_centered, pocket_features_norm], dim=1)
            
            # Add noise at this level
            noise_lig = torch.randn_like(xh_lig_clean)
            noise_pocket = torch.randn_like(xh_pocket_clean)
            
            # COM-free noise
            noise_coords_combined = torch.cat([noise_lig[:, :self.n_dims], noise_pocket[:, :self.n_dims]], dim=0)
            noise_coords_centered = remove_mean_batch(noise_coords_combined, combined_mask)
            noise_lig[:, :self.n_dims] = noise_coords_centered[:len(lig_coords)]
            noise_pocket[:, :self.n_dims] = noise_coords_centered[len(lig_coords):]
            
            xh_lig_noisy = xh_lig_clean + sigma_val * noise_lig
            xh_pocket_noisy = xh_pocket_clean + sigma_val * noise_pocket
            
            # Denoise
            t_normalized = (sigma_val / self.scheme.sigma_max).unsqueeze(0).expand(batch_size, 1)
            
            with torch.no_grad():
                pred_clean_lig, pred_clean_pocket = self.denoiser(
                    xh_lig_noisy, xh_pocket_noisy, t_normalized, lig_mask, pocket_mask
                )
            
            # Compute evaluation losses
            pred_coords_lig = pred_clean_lig[:, :self.n_dims]
            pred_features_lig = pred_clean_lig[:, self.n_dims:]
            clean_coords_lig = xh_lig_clean[:, :self.n_dims]
            
            coord_loss = torch.mean((pred_coords_lig - clean_coords_lig) ** 2)
            
            # For categorical evaluation
            pred_features_lig_unnorm = pred_features_lig * self.scheme.feature_norm + self.scheme.feature_bias
            pred_logits_lig = pred_features_lig_unnorm / self.scheme.categorical_temperature
            true_categories_lig = torch.argmax(lig_features_clean, dim=-1)
            categorical_loss = F.cross_entropy(pred_logits_lig, true_categories_lig)
            
            eval_losses[f"coord_loss_lvl{i}"] = coord_loss.item()
            eval_losses[f"categorical_loss_lvl{i}"] = categorical_loss.item()
            eval_losses[f"total_loss_lvl{i}"] = coord_loss.item() + categorical_loss.item()
        
        return eval_losses


def test_molecular_diffusion():
    """Test the GenCFD-style molecular diffusion implementation"""
    
    print("Testing GenCFD-Style Molecular Diffusion (CUDA)...")
    
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
    
    print(f"--- Testing GenCFD-style loss ---")
    
    # Create model
    sigma_schedule = molecular_exponential_noise_schedule(clip_max=10.0)  # Smaller max
    scheme = MolecularDiffusion.create_variance_exploding(
        sigma=sigma_schedule,
        categorical_temperature=1.0
    )
    
    model = MolecularDenoisingModel(
        atom_nf=atom_nf,
        residue_nf=residue_nf,
        joint_nf=8,
        hidden_nf=32,
        n_layers=2,
        scheme=scheme,
        coord_loss_weight=1.0,
        categorical_loss_weight=0.5
    )
    
    model.initialize()
    
    # Test loss computation
    loss, metrics = model.loss_fn(batch)
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Coord Loss: {metrics['coord_loss']:.4f}")
    print(f"Categorical Loss: {metrics['categorical_loss']:.4f}")
    print(f"Sigma range: [{metrics['avg_sigma']:.3f}, {metrics['max_sigma']:.3f}]")
    print(f"Coord prediction scale: {metrics['coord_pred_scale']:.3f}")
    print(f"Feature prediction scale (lig): {metrics['feature_pred_scale_lig']:.3f}")
    print(f"Feature prediction scale (pocket): {metrics['feature_pred_scale_pocket']:.3f}")
    
    # Test evaluation
    eval_metrics = model.eval_fn(batch)
    print(f"Eval metrics: {eval_metrics}")
    
    print("\n✅ GenCFD-style molecular diffusion test passed!")
    
    return model


if __name__ == "__main__":
    test_molecular_diffusion()
