"""
Molecular diffusion pipeline (GenCFD-style with categorical features).
"""

import dataclasses
import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, Tuple, Optional, Protocol
from torch_scatter import scatter_mean

from egnn_dynamics import EGNNDynamics, PreconditionedEGNNDynamics

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
class InvertibleSchedule:
    """Molecular version of InvertibleSchedule"""
    
    forward: ScheduleFn
    inverse: ScheduleFn
    device: str = 'cuda'
    
    def __call__(self, t: Numeric) -> Tensor:
        result = self.forward(t)
        if isinstance(result, (int, float)):
            result = torch.tensor(result, device=self.device, dtype=torch.float32)
        return result.to(self.device)


def _linear_rescale(in_min: float, in_max: float, out_min: float, out_max: float) -> InvertibleSchedule:
    in_range = in_max - in_min
    out_range = out_max - out_min
    fwd = lambda x: out_min + (x - in_min) / in_range * out_range
    inv = lambda y: in_min + (y - out_min) / out_range * in_range
    return InvertibleSchedule(fwd, inv, device='cuda')


def exponential_noise_schedule(
    clip_max: float = 100.0,
    base: float = np.e**0.5,
    start: float = 0.0,
    end: float = 5.0,
) -> InvertibleSchedule:
    
    if not (start < end and base > 1.0):
        raise ValueError("Must have `base` > 1 and `start` < `end`.")

    in_rescale = _linear_rescale(
        in_min=MIN_DIFFUSION_TIME,
        in_max=MAX_DIFFUSION_TIME,
        out_min=start,
        out_max=end,
    )
    out_rescale = _linear_rescale(
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
    
    return InvertibleSchedule(sigma, inverse, device='cuda')


# ********************
# Diffusion Scheme (Simplified)
# ********************

@dataclasses.dataclass(frozen=True)
class MolecularDiffusion:
    """
    Simplified GenCFD-style diffusion for molecular data.
    """
    
    sigma: InvertibleSchedule
    
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
        """Unnormalize molecular data with proper dimension handling"""
        
        # Unnormalize coordinates
        ligand_coords_unnorm = ligand_coords * self.coord_norm
        pocket_coords_unnorm = pocket_coords * self.coord_norm
        
        # Unnormalize features
        ligand_features_unnorm = ligand_features * self.feature_norm + self.feature_bias
        pocket_features_unnorm = pocket_features * self.feature_norm + self.feature_bias
        
        if discretize_features:
            # Debug info
            print(f"DEBUG unnormalize: ligand_features_unnorm.shape = {ligand_features_unnorm.shape}")
            print(f"DEBUG unnormalize: pocket_features_unnorm.shape = {pocket_features_unnorm.shape}")
            print(f"DEBUG unnormalize: ligand_features range [{ligand_features_unnorm.min():.3f}, {ligand_features_unnorm.max():.3f}]")
            print(f"DEBUG unnormalize: pocket_features range [{pocket_features_unnorm.min():.3f}, {pocket_features_unnorm.max():.3f}]")
            
            # Use dim=-1 to get argmax over the last dimension (feature classes)
            # Clamp to ensure indices are in valid range
            lig_indices = torch.argmax(ligand_features_unnorm, dim=-1)  # [N_lig]
            pocket_indices = torch.argmax(pocket_features_unnorm, dim=-1)  # [N_pocket]
            
            print(f"DEBUG: lig_indices.shape = {lig_indices.shape}, range [{lig_indices.min()}, {lig_indices.max()}]")
            print(f"DEBUG: pocket_indices.shape = {pocket_indices.shape}, range [{pocket_indices.min()}, {pocket_indices.max()}]")
            print(f"DEBUG: expected atom_nf = {atom_nf}, residue_nf = {residue_nf}")
            
            # Convert to one-hot
            ligand_features_unnorm = torch.nn.functional.one_hot(lig_indices, atom_nf).float()
            pocket_features_unnorm = torch.nn.functional.one_hot(pocket_indices, residue_nf).float()
            
            print(f"DEBUG: final ligand_features_unnorm.shape = {ligand_features_unnorm.shape}")
            print(f"DEBUG: final pocket_features_unnorm.shape = {pocket_features_unnorm.shape}")
        
        return ligand_coords_unnorm, ligand_features_unnorm, pocket_coords_unnorm, pocket_features_unnorm
    
    @classmethod
    def create_variance_exploding(
        cls,
        sigma: InvertibleSchedule,
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

class NoiseLevelSampling(Protocol):
    def __call__(self, shape: Tuple[int, ...]) -> Tensor: ...


def log_uniform_sampling(
    scheme: MolecularDiffusion,
    clip_min: float = 1e-4,
    uniform_grid: bool = False,
) -> NoiseLevelSampling:

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


class NoiseLossWeighting(Protocol):
    def __call__(self, sigma: Tensor) -> Tensor: ...

def edm_weighting(
    data_std: float = 1.0, device=None
) -> NoiseLossWeighting:
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
    noise_sampling: NoiseLevelSampling = None
    noise_weighting: NoiseLossWeighting = None
    geometric_regularization: bool = True
    geom_loss_weight: float = 0.1
    
    def __post_init__(self):
        """Initialize the denoiser and diffusion components"""
        
        # Create default scheme if not provided
        if self.scheme is None:
            sigma_schedule = exponential_noise_schedule(clip_max=100.0) 
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
            noise_sampling = log_uniform_sampling(self.scheme)
            object.__setattr__(self, 'noise_sampling', noise_sampling)
        
        # Create default noise weighting if not provided
        if self.noise_weighting is None:
            noise_weighting = edm_weighting()
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
            edge_embedding_dim=self.edge_embedding_dim,
            geometric_regularization=self.geometric_regularization,
            geom_loss_weight=self.geom_loss_weight
        )

        denoiser = PreconditionedEGNNDynamics(denoiser)
        from ema_pytorch import EMA
        denoise = EMA(denoiser, update_after_step=100)

        object.__setattr__(self, 'denoiser', denoiser)

    def initialize(self):
        """Initialize model weights"""
        print(f"✅ Initialized MolecularDenoisingModel with {sum(p.numel() for p in self.denoiser.parameters())} parameters")

    def loss_fn(self, batch: dict):
        """
        CDCD-style loss function: noise embeddings, predict logits, use cross-entropy
        """
        
        # Extract molecular data
        lig_coords = batch['ligand_coords'].cuda()
        lig_features = batch['ligand_features'].cuda()  # One-hot features
        pocket_coords = batch['pocket_coords'].cuda()
        pocket_features = batch['pocket_features'].cuda()  # One-hot features
        lig_mask = batch['ligand_mask'].cuda()
        pocket_mask = batch['pocket_mask'].cuda()
        batch_size = len(torch.unique(torch.cat([lig_mask, pocket_mask])))
        
        # Get true atom/residue type indices for cross-entropy loss
        true_atom_types = torch.argmax(lig_features, dim=-1)      # [N_lig] - indices
        true_residue_types = torch.argmax(pocket_features, dim=-1)  # [N_pocket] - indices
        
        # Convert one-hot to embeddings using encoders (DIFFERENTIABLE!)
        lig_embeddings_clean = self.denoiser.atom_encoder(lig_features.float())      # [N_lig, joint_nf]
        pocket_embeddings_clean = self.denoiser.residue_encoder(pocket_features.float())  # [N_pocket, joint_nf]
        
        # Normalize embeddings (CDCD style)
        lig_embeddings_clean = F.normalize(lig_embeddings_clean, dim=-1)
        pocket_embeddings_clean = F.normalize(pocket_embeddings_clean, dim=-1)
        
        # Normalize coordinates
        lig_coords_norm = lig_coords / self.scheme.coord_norm
        pocket_coords_norm = pocket_coords / self.scheme.coord_norm
        
        # Concatenate coordinates and embeddings (NOT one-hot!)
        xh_lig_clean = torch.cat([lig_coords_norm, lig_embeddings_clean], dim=1)  # [N_lig, 3 + joint_nf]
        xh_pocket_clean = torch.cat([pocket_coords_norm, pocket_embeddings_clean], dim=1)  # [N_pocket, 3 + joint_nf]
        
        # Sample noise levels
        sigma = self.noise_sampling(shape=(batch_size,))  # [batch_size]
        
        # Generate Gaussian noise for the entire state
        noise_lig = torch.randn_like(xh_lig_clean)
        
        
        # Add noise
        sigma_expanded_lig = sigma[lig_mask].unsqueeze(1)  # [N_lig, 1]
        sigma_expanded_pocket = sigma[pocket_mask].unsqueeze(1)  # [N_pocket, 1]
        
        xh_lig_noisy = xh_lig_clean + sigma_expanded_lig * noise_lig
        
        
        # Forward pass through denoiser - outputs coordinates + logits
        # print('In train:')
        # print('Sigma shape:', sigma.shape)
        # print('xh_lig_noisy shape:', xh_lig_noisy.shape)
        # print('xh_pocket_noisy shape:', xh_pocket_noisy.shape)
        # print('xh_pocket_noisy shape:', xh_pocket_noisy.shape)
        # print('lig_mask', lig_mask.shape)
        # print('pocket_mask', pocket_mask.shape)
        
        pred_output_lig, _ = self.denoiser(
            xh_lig_noisy, xh_pocket_clean, sigma, lig_mask, pocket_mask,
            target_atoms=lig_coords_norm, target_residues=pocket_coords_norm,
        )
        
        # Split predictions: coordinates + logits
        pred_coords_lig = pred_output_lig[:, :self.n_dims]              # [N_lig, 3]
        pred_logits_lig = pred_output_lig[:, self.n_dims:]              # [N_lig, atom_nf] - LOGITS
        
        # Loss weighting
        weights = self.noise_weighting(sigma)  # [batch_size]
        weights_lig = weights[lig_mask].unsqueeze(1)  # [N_lig, 1]
        weights_pocket = weights[pocket_mask].unsqueeze(1)  # [N_pocket, 1]
        
        # Coordinate loss: L2 between predicted and clean coordinates
        clean_coords_lig = xh_lig_clean[:, :self.n_dims]
        
        coord_loss_lig = torch.mean(weights_lig * (pred_coords_lig - clean_coords_lig) ** 2)
        coord_loss_pocket = torch.tensor(0.0, device = "cuda")
        coord_loss = coord_loss_lig + coord_loss_pocket
        
        # Categorical loss: Cross-entropy between logits and true atom types
        ce_loss_lig = F.cross_entropy(pred_logits_lig, true_atom_types, reduction='none')
        ce_loss_pocket = torch.zeros_like(true_residue_types, dtype=torch.float, device = "cuda")
        
        # Weight categorical loss by noise level (like coordinates)
        # categorical_loss_lig = torch.mean(weights[lig_mask] * ce_loss_lig)
        # categorical_loss_pocket = torch.mean(weights[pocket_mask] * ce_loss_pocket)
        # categorical_loss = categorical_loss_lig + categorical_loss_pocket

        categorical_loss_lig = ce_loss_lig.mean()
        categorical_loss_pocket = ce_loss_pocket.mean()
        categorical_loss = categorical_loss_lig + categorical_loss_pocket
        geometric_loss = self.denoiser.last_geometric_loss
        
        # Combine losses
        total_loss = (
            self.coord_loss_weight * coord_loss + 
            self.categorical_loss_weight * categorical_loss +
            geometric_loss
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
            "geometric_loss_total": geometric_loss.item(),
            "avg_sigma": sigma.mean().item(),
            "max_sigma": sigma.max().item(),
            "coord_pred_scale": pred_coords_lig.abs().mean().item(),
            "atom_accuracy": (torch.argmax(pred_logits_lig, dim=-1) == true_atom_types).float().mean().item(),
            "residue_accuracy": torch.tensor(1.0, device = "cuda")
        }
        return total_loss, metrics

    def eval_fn(self, batch: dict) -> dict:
        """Evaluate denoising at multiple noise levels with proper loss breakdown"""
        
        # Extract data
        lig_coords = batch['ligand_coords'].cuda()
        lig_features = batch['ligand_features'].cuda()  # One-hot features
        pocket_coords = batch['pocket_coords'].cuda()
        pocket_features = batch['pocket_features'].cuda()  # One-hot features
        lig_mask = batch['ligand_mask'].cuda()
        pocket_mask = batch['pocket_mask'].cuda()
        batch_size = batch['batch_size']
        
        # Get true categories for evaluation
        true_atom_types = torch.argmax(lig_features, dim=-1)
        true_residue_types = torch.argmax(pocket_features, dim=-1)
        
        # Cache training quantiles.
        if not hasattr(self, '_cached_sigma_levels'):
            training_sigmas = self.noise_sampling(shape=(10000,))
            sigma_levels = torch.quantile(training_sigmas, torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], device='cuda'))
            object.__setattr__(self, '_cached_sigma_levels', sigma_levels)
        else:
            sigma_levels = self._cached_sigma_levels


        eval_losses = {}
        
        for i, sigma_val in enumerate(sigma_levels):
            # Convert one-hot to embeddings (same as training)
            lig_embeddings_clean = self.denoiser.atom_encoder(lig_features.float())
            pocket_embeddings_clean = self.denoiser.residue_encoder(pocket_features.float())
            
            # Normalize embeddings (CDCD style)
            lig_embeddings_clean = F.normalize(lig_embeddings_clean, dim=-1)
            pocket_embeddings_clean = F.normalize(pocket_embeddings_clean, dim=-1)
            
            # Normalize coordinates
            lig_coords_norm = lig_coords / self.scheme.coord_norm
            pocket_coords_norm = pocket_coords / self.scheme.coord_norm
            
            # Concatenate coordinates and embeddings (NOT one-hot!)
            xh_lig_clean = torch.cat([lig_coords_norm, lig_embeddings_clean], dim=1)
            xh_pocket_clean = torch.cat([pocket_coords_norm, pocket_embeddings_clean], dim=1)
            
            # Add noise at this level
            noise_lig = torch.randn_like(xh_lig_clean)
            
            xh_lig_noisy = xh_lig_clean + sigma_val * noise_lig
            
            # Create sigma tensor for this evaluation (broadcast to batch_size)
            sigma_batch = sigma_val

            # print('In eval:')
            # print('Noise level:', i, sigma_val)
            # print('Sigma shape:', sigma_batch.shape)
            # print('xh_lig_noisy shape:', xh_lig_noisy.shape)
            # print('xh_pocket_noisy shape:', xh_pocket_noisy.shape)
            # print('xh_pocket_noisy shape:', xh_pocket_noisy.shape)
            # print('lig_mask', lig_mask.shape)
            # print('pocket_mask', pocket_mask.shape)
            
            
            # Denoise: clean pocket input
            with torch.no_grad():
                pred_output_lig, _ = self.denoiser(
                    xh_lig_noisy, xh_pocket_clean, sigma_batch, lig_mask, pocket_mask
                )
            
            
            # Split predictions: coordinates + logits
            pred_coords_lig = pred_output_lig[:, :self.n_dims]              # [N_lig, 3]
            pred_logits_lig = pred_output_lig[:, self.n_dims:]              # [N_lig, atom_nf] - LOGITS
            
            # Compute losses
            clean_coords_lig = xh_lig_clean[:, :self.n_dims]
            clean_coords_pocket = xh_pocket_clean[:, :self.n_dims]
            
            # Coordinate loss: L2
            coord_loss_lig = torch.mean((pred_coords_lig - clean_coords_lig) ** 2)
            coord_loss_pocket = torch.tensor(0.0, device = "cuda")
            coord_loss = coord_loss_lig + coord_loss_pocket
            
            # Categorical loss: Cross-entropy on logits
            categorical_loss_lig = F.cross_entropy(pred_logits_lig, true_atom_types)
            categorical_loss_pocket = torch.tensor(0.0, device = "cuda")
            categorical_loss = categorical_loss_lig + categorical_loss_pocket
            
            # Accuracy metrics
            atom_accuracy = (torch.argmax(pred_logits_lig, dim=-1) == true_atom_types).float().mean()
            residue_accuracy = torch.tensor(1.0, device = "cuda")
            
            eval_losses[f"coord_loss_lvl{i}"] = coord_loss.item()
            eval_losses[f"categorical_loss_lvl{i}"] = categorical_loss.item()
            eval_losses[f"total_loss_lvl{i}"] = coord_loss.item() + categorical_loss.item()
            eval_losses[f"atom_accuracy_lvl{i}"] = atom_accuracy.item()
            eval_losses[f"residue_accuracy_lvl{i}"] = residue_accuracy.item()
        
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
    sigma_schedule = exponential_noise_schedule(clip_max=100.0)  # Smaller max
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
        coord_loss_weight=2.0,
        categorical_loss_weight=1.0
    )
    
    model.initialize()
    
    # Test loss computation
    loss, metrics = model.loss_fn(batch)
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Coord Loss: {metrics['coord_loss']:.4f}")
    print(f"Categorical Loss: {metrics['categorical_loss']:.4f}")
    print(f"Sigma range: [{metrics['avg_sigma']:.3f}, {metrics['max_sigma']:.3f}]")
    print(f"Coord prediction scale: {metrics['coord_pred_scale']:.3f}")
    
    # Test evaluation
    eval_metrics = model.eval_fn(batch)
    print(f"Eval metrics: {eval_metrics}")
    
    print("\n✅ GenCFD-style molecular diffusion test passed!")
    
    return model


if __name__ == "__main__":
    test_molecular_diffusion()
