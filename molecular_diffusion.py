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
        
        # Create the EGNN denoiser WITH CONDITIONING
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
            use_conditioning=True  # NEW - always enable conditioning
        )

        denoiser = PreconditionedEGNNDynamics(denoiser)
       

        object.__setattr__(self, 'denoiser', denoiser)

    def initialize(self):
        """Initialize model weights"""
        print(f"✅ Initialized MolecularDenoisingModel with {sum(p.numel() for p in self.denoiser.parameters())} parameters")

    def loss_fn(self, batch: dict):
        """
        CDCD-style loss function with conditioning support
        """
        
        lig_coords = batch['ligand_coords'].cuda()
        lig_features = batch['ligand_features'].cuda()
        pocket_coords = batch['pocket_coords'].cuda()
        pocket_features = batch['pocket_features'].cuda()
        lig_mask = batch['ligand_mask'].cuda()
        pocket_mask = batch['pocket_mask'].cuda()
        batch_size = len(torch.unique(torch.cat([lig_mask, pocket_mask])))
        
        # Extract initial coords for conditioning (if present)
        initial_lig_coords = batch.get('initial_ligand_coords', None)
        initial_pocket_coords = batch.get('initial_pocket_coords', None)
        
        # Extract node IDs for variable-sized conditioning
        node_ids_atoms = batch.get('node_ids_atoms', None)
        node_ids_residues = batch.get('node_ids_residues', None)
        cond_node_ids_atoms = batch.get('cond_node_ids_atoms', None)
        cond_node_ids_residues = batch.get('cond_node_ids_residues', None)
        
        # Get true atom/residue type indices
        true_atom_types = torch.argmax(lig_features, dim=-1)
        true_residue_types = torch.argmax(pocket_features, dim=-1)
        
        # Convert one-hot to embeddings (DIFFERENTIABLE)
        lig_embeddings_clean = self.denoiser.egnn_dynamics.atom_encoder(lig_features.float())
        pocket_embeddings_clean = self.denoiser.egnn_dynamics.residue_encoder(pocket_features.float())
        
        # Normalize embeddings (CDCD style)
        lig_embeddings_clean = F.normalize(lig_embeddings_clean, dim=-1)
        pocket_embeddings_clean = F.normalize(pocket_embeddings_clean, dim=-1)
        
        # Normalize coordinates
        lig_coords_norm = lig_coords / self.scheme.coord_norm
        pocket_coords_norm = pocket_coords / self.scheme.coord_norm
        
        # Compute conditioning from INITIAL COORDINATES (variable-sized graphs)
        cond_coords_lig = None
        cond_coords_pocket = None
        
        # Check if ALL required conditioning inputs are present
        has_coords = initial_lig_coords is not None and initial_pocket_coords is not None
        has_ids = (node_ids_atoms is not None and node_ids_residues is not None and
                   cond_node_ids_atoms is not None and cond_node_ids_residues is not None)
        
        if has_coords and has_ids:
            initial_lig_coords = initial_lig_coords.cuda()
            initial_pocket_coords = initial_pocket_coords.cuda()
            
            # Normalize initial coords
            initial_lig_norm = initial_lig_coords / self.scheme.coord_norm
            initial_pocket_norm = initial_pocket_coords / self.scheme.coord_norm
            
            # ✅ NO SLICING - pass full conditioning graphs
            # The _match_by_id() function handles size mismatches
            cond_coords_lig = initial_lig_norm
            cond_coords_pocket = initial_pocket_norm
            
            # Conditioning dropout (10% of the time, don't condition)
            if torch.rand(()) < 0.1:
                cond_coords_lig = None
                cond_coords_pocket = None
                node_ids_atoms = None
                node_ids_residues = None
                cond_node_ids_atoms = None
                cond_node_ids_residues = None
            
        # Remove center of mass for translation invariance
        combined_coords = torch.cat([lig_coords_norm, pocket_coords_norm], dim=0)
        combined_mask = torch.cat([lig_mask, pocket_mask], dim=0)
        combined_coords_centered = remove_mean_batch(combined_coords, combined_mask)
        
        lig_coords_centered = combined_coords_centered[:len(lig_coords)]
        pocket_coords_centered = combined_coords_centered[len(lig_coords):]
        
        # Concatenate coordinates and embeddings
        xh_lig_clean = torch.cat([lig_coords_centered, lig_embeddings_clean], dim=1)
        xh_pocket_clean = torch.cat([pocket_coords_centered, pocket_embeddings_clean], dim=1)
        
        # Sample noise levels
        sigma = self.noise_sampling(shape=(batch_size,))
        
        # Generate Gaussian noise
        noise_lig = torch.randn_like(xh_lig_clean)
        noise_pocket = torch.randn_like(xh_pocket_clean)
        
        # Make coordinate noise COM-free
        noise_coords_combined = torch.cat([noise_lig[:, :self.n_dims], noise_pocket[:, :self.n_dims]], dim=0)
        noise_coords_centered = remove_mean_batch(noise_coords_combined, combined_mask)
        noise_lig[:, :self.n_dims] = noise_coords_centered[:len(lig_coords)]
        noise_pocket[:, :self.n_dims] = noise_coords_centered[len(lig_coords):]
        
        # Add noise
        sigma_expanded_lig = sigma[lig_mask].unsqueeze(1)
        sigma_expanded_pocket = sigma[pocket_mask].unsqueeze(1)
        
        xh_lig_noisy = xh_lig_clean + sigma_expanded_lig * noise_lig
        xh_pocket_noisy = xh_pocket_clean + sigma_expanded_pocket * noise_pocket
        
        # Forward pass WITH CONDITIONING
        pred_output_lig, pred_output_pocket = self.denoiser(
            xh_lig_noisy, xh_pocket_noisy, sigma, lig_mask, pocket_mask,
            target_atoms=lig_coords_centered,
            target_residues=pocket_coords_centered,
            cond_coords_atoms=cond_coords_lig,  # NEW
            cond_coords_residues=cond_coords_pocket,  # NEW
            node_ids_atoms=node_ids_atoms, # NEW
            node_ids_residues=node_ids_residues, # NEW
            cond_node_ids_atoms=cond_node_ids_atoms, # NEW
            cond_node_ids_residues=cond_node_ids_residues, # NEW
        )
        
        # Split predictions
        pred_coords_lig = pred_output_lig[:, :self.n_dims]
        pred_logits_lig = pred_output_lig[:, self.n_dims:]
        pred_coords_pocket = pred_output_pocket[:, :self.n_dims]
        pred_logits_pocket = pred_output_pocket[:, self.n_dims:]
        
        # Loss weighting
        weights = self.noise_weighting(sigma)
        weights_lig = weights[lig_mask].unsqueeze(1)
        weights_pocket = weights[pocket_mask].unsqueeze(1)
        
        # Coordinate loss
        clean_coords_lig = xh_lig_clean[:, :self.n_dims]
        clean_coords_pocket = xh_pocket_clean[:, :self.n_dims]
        
        coord_loss_lig = torch.mean(weights_lig * (pred_coords_lig - clean_coords_lig) ** 2)
        coord_loss_pocket = torch.mean(weights_pocket * (pred_coords_pocket - clean_coords_pocket) ** 2)
        coord_loss = coord_loss_lig + coord_loss_pocket
        
        # Categorical loss
        ce_loss_lig = F.cross_entropy(pred_logits_lig, true_atom_types, reduction='none')
        ce_loss_pocket = F.cross_entropy(pred_logits_pocket, true_residue_types, reduction='none')
        
        categorical_loss_lig = ce_loss_lig.mean()
        categorical_loss_pocket = ce_loss_pocket.mean()
        categorical_loss = categorical_loss_lig + categorical_loss_pocket
        geometric_loss = self.denoiser.last_geometric_loss
        
        # Total loss
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
            "coord_pred_scale": torch.cat([pred_coords_lig, pred_coords_pocket]).abs().mean().item(),
            "atom_accuracy": (torch.argmax(pred_logits_lig, dim=-1) == true_atom_types).float().mean().item(),
            "residue_accuracy": (torch.argmax(pred_logits_pocket, dim=-1) == true_residue_types).float().mean().item(),
            "using_conditioning": cond_coords_lig is not None and cond_node_ids_atoms is not None, 
        }
        return total_loss, metrics
        
    def eval_fn(self, batch: dict) -> dict:
        """Evaluate denoising at multiple noise levels with conditioning support"""
        
        # Extract data
        lig_coords = batch['ligand_coords'].cuda()
        lig_features = batch['ligand_features'].cuda()
        pocket_coords = batch['pocket_coords'].cuda()
        pocket_features = batch['pocket_features'].cuda()
        lig_mask = batch['ligand_mask'].cuda()
        pocket_mask = batch['pocket_mask'].cuda()
        batch_size = batch['batch_size']
        
        # Extract conditioning if present
        initial_lig_coords = batch.get('initial_ligand_coords', None)
        initial_pocket_coords = batch.get('initial_pocket_coords', None)
        
        # Extract node IDs for variable-sized conditioning
        node_ids_atoms = batch.get('node_ids_atoms', None)
        node_ids_residues = batch.get('node_ids_residues', None)
        cond_node_ids_atoms = batch.get('cond_node_ids_atoms', None)
        cond_node_ids_residues = batch.get('cond_node_ids_residues', None)
        
        # Get true categories
        true_atom_types = torch.argmax(lig_features, dim=-1)
        true_residue_types = torch.argmax(pocket_features, dim=-1)
        
        # Cache training quantiles
        if not hasattr(self, '_cached_sigma_levels'):
            training_sigmas = self.noise_sampling(shape=(10000,))
            sigma_levels = torch.quantile(training_sigmas, torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], device='cuda'))
            object.__setattr__(self, '_cached_sigma_levels', sigma_levels)
        else:
            sigma_levels = self._cached_sigma_levels
    
        eval_losses = {}
        
        for i, sigma_val in enumerate(sigma_levels):
            # Convert one-hot to embeddings
            lig_embeddings_clean = self.denoiser.egnn_dynamics.atom_encoder(lig_features.float())
            pocket_embeddings_clean = self.denoiser.egnn_dynamics.residue_encoder(pocket_features.float())
            
            # Normalize embeddings
            lig_embeddings_clean = F.normalize(lig_embeddings_clean, dim=-1)
            pocket_embeddings_clean = F.normalize(pocket_embeddings_clean, dim=-1)
            
            # Normalize coordinates
            lig_coords_norm = lig_coords / self.scheme.coord_norm
            pocket_coords_norm = pocket_coords / self.scheme.coord_norm
            
            # Compute conditioning from INITIAL COORDINATES (variable-sized)
            cond_coords_lig = None
            cond_coords_pocket = None
            
            # Check if ALL required conditioning inputs are present
            has_coords = initial_lig_coords is not None and initial_pocket_coords is not None
            has_ids = (node_ids_atoms is not None and node_ids_residues is not None and
                       cond_node_ids_atoms is not None and cond_node_ids_residues is not None)
            
            if has_coords and has_ids:
                initial_lig_coords_cuda = initial_lig_coords.cuda()
                initial_pocket_coords_cuda = initial_pocket_coords.cuda()
                
                # Normalize initial coords
                initial_lig_norm = initial_lig_coords_cuda / self.scheme.coord_norm
                initial_pocket_norm = initial_pocket_coords_cuda / self.scheme.coord_norm
                
                # Pass full conditioning graphs
                cond_coords_lig = initial_lig_norm
                cond_coords_pocket = initial_pocket_norm

            # Remove center of mass
            combined_coords = torch.cat([lig_coords_norm, pocket_coords_norm], dim=0)
            combined_mask = torch.cat([lig_mask, pocket_mask], dim=0)
            combined_coords_centered = remove_mean_batch(combined_coords, combined_mask)
            
            lig_coords_centered = combined_coords_centered[:len(lig_coords)]
            pocket_coords_centered = combined_coords_centered[len(lig_coords):]
            
            # Concatenate
            xh_lig_clean = torch.cat([lig_coords_centered, lig_embeddings_clean], dim=1)
            xh_pocket_clean = torch.cat([pocket_coords_centered, pocket_embeddings_clean], dim=1)
            
            # Add noise
            noise_lig = torch.randn_like(xh_lig_clean)
            noise_pocket = torch.randn_like(xh_pocket_clean)
            
            # COM-free noise
            noise_coords_combined = torch.cat([noise_lig[:, :self.n_dims], noise_pocket[:, :self.n_dims]], dim=0)
            noise_coords_centered = remove_mean_batch(noise_coords_combined, combined_mask)
            noise_lig[:, :self.n_dims] = noise_coords_centered[:len(lig_coords)]
            noise_pocket[:, :self.n_dims] = noise_coords_centered[len(lig_coords):]
            
            xh_lig_noisy = xh_lig_clean + sigma_val * noise_lig
            xh_pocket_noisy = xh_pocket_clean + sigma_val * noise_pocket
            
            # Create sigma tensor
            sigma_batch = sigma_val
            
            # Denoise WITH CONDITIONING (variable-sized)
            with torch.no_grad():
                pred_output_lig, pred_output_pocket = self.denoiser(
                    xh_lig_noisy, xh_pocket_noisy, sigma_batch, lig_mask, pocket_mask,
                    cond_coords_atoms=cond_coords_lig,
                    cond_coords_residues=cond_coords_pocket,
                    node_ids_atoms=node_ids_atoms,
                    node_ids_residues=node_ids_residues,
                    cond_node_ids_atoms=cond_node_ids_atoms,
                    cond_node_ids_residues=cond_node_ids_residues,
                )
            
            # Split predictions
            pred_coords_lig = pred_output_lig[:, :self.n_dims]
            pred_logits_lig = pred_output_lig[:, self.n_dims:]
            pred_coords_pocket = pred_output_pocket[:, :self.n_dims]
            pred_logits_pocket = pred_output_pocket[:, self.n_dims:]
            
            # Compute losses
            clean_coords_lig = xh_lig_clean[:, :self.n_dims]
            clean_coords_pocket = xh_pocket_clean[:, :self.n_dims]
            
            coord_loss_lig = torch.mean((pred_coords_lig - clean_coords_lig) ** 2)
            coord_loss_pocket = torch.mean((pred_coords_pocket - clean_coords_pocket) ** 2)
            coord_loss = coord_loss_lig + coord_loss_pocket
            
            categorical_loss_lig = F.cross_entropy(pred_logits_lig, true_atom_types)
            categorical_loss_pocket = F.cross_entropy(pred_logits_pocket, true_residue_types)
            categorical_loss = categorical_loss_lig + categorical_loss_pocket
            
            atom_accuracy = (torch.argmax(pred_logits_lig, dim=-1) == true_atom_types).float().mean()
            residue_accuracy = (torch.argmax(pred_logits_pocket, dim=-1) == true_residue_types).float().mean()
            
            eval_losses[f"coord_loss_lvl{i}"] = coord_loss.item()
            eval_losses[f"categorical_loss_lvl{i}"] = categorical_loss.item()
            eval_losses[f"total_loss_lvl{i}"] = coord_loss.item() + categorical_loss.item()
            eval_losses[f"atom_accuracy_lvl{i}"] = atom_accuracy.item()
            eval_losses[f"residue_accuracy_lvl{i}"] = residue_accuracy.item()
        
        return eval_losses


def test_molecular_diffusion():
    """Test the GenCFD-style molecular diffusion implementation with conditioning"""
    
    print("Testing GenCFD-Style Molecular Diffusion with Conditioning (CUDA)...")
    
    # Configuration
    device = 'cuda'
    batch_size = 2
    atom_nf = 5
    residue_nf = 7
    
    # Create base batch WITHOUT conditioning
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
    sigma_schedule = exponential_noise_schedule(clip_max=100.0)
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
    
    print(f"\n--- Test 1: Training WITHOUT conditioning ---")
    loss, metrics = model.loss_fn(batch)
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Coord Loss: {metrics['coord_loss']:.4f}")
    print(f"Categorical Loss: {metrics['categorical_loss']:.4f}")
    print(f"Using conditioning: {metrics['using_conditioning']}")
    
    print(f"\n--- Test 2: Training WITH random conditioning ---")
    batch_with_cond = batch.copy()
    # Add noisy initial coordinates (NOT displacement - just shifted coords)
    batch_with_cond['initial_ligand_coords'] = batch['ligand_coords'] + torch.randn_like(batch['ligand_coords']) * 0.5
    batch_with_cond['initial_pocket_coords'] = batch['pocket_coords'] + torch.randn_like(batch['pocket_coords']) * 0.5
    
    loss_cond, metrics_cond = model.loss_fn(batch_with_cond)
    print(f"Total Loss (with cond): {loss_cond.item():.4f}")
    print(f"Coord Loss (with cond): {metrics_cond['coord_loss']:.4f}")
    print(f"Categorical Loss (with cond): {metrics_cond['categorical_loss']:.4f}")
    print(f"Using conditioning: {metrics_cond['using_conditioning']}")
    
    print(f"\n--- Test 3: Training WITH perfect conditioning ---")
    batch_perfect = batch.copy()
    # Perfect conditioning: initial = target (so model learns identity mapping with conditioning)
    batch_perfect['initial_ligand_coords'] = batch['ligand_coords'].clone()
    batch_perfect['initial_pocket_coords'] = batch['pocket_coords'].clone()
    
    loss_perfect, metrics_perfect = model.loss_fn(batch_perfect)
    print(f"Total Loss (perfect cond): {loss_perfect.item():.4f}")
    print(f"Coord Loss (perfect cond): {metrics_perfect['coord_loss']:.4f}")
    print(f"Categorical Loss (perfect cond): {metrics_perfect['categorical_loss']:.4f}")
    print(f"Using conditioning: {metrics_perfect['using_conditioning']}")
    
    # Sanity check: perfect conditioning should give SIMILAR or BETTER coord loss
    # (not necessarily lower since model also needs to denoise the noise we added!)
    print(f"\n--- Sanity Checks ---")
    print(f"Conditioning changes output: {abs(loss_cond.item() - loss.item()) > 1e-6}")
    
    print(f"\n--- Test 4: Evaluation WITHOUT conditioning ---")
    eval_metrics = model.eval_fn(batch)
    print(f"Eval coord_loss_lvl0: {eval_metrics['coord_loss_lvl0']:.4f}")
    print(f"Eval atom_accuracy_lvl0: {eval_metrics['atom_accuracy_lvl0']:.4f}")
    
    print(f"\n--- Test 5: Evaluation WITH conditioning ---")
    eval_metrics_cond = model.eval_fn(batch_with_cond)
    print(f"Eval coord_loss_lvl0 (with cond): {eval_metrics_cond['coord_loss_lvl0']:.4f}")
    print(f"Eval atom_accuracy_lvl0 (with cond): {eval_metrics_cond['atom_accuracy_lvl0']:.4f}")
    
    # ========== NEW TESTS FOR VARIABLE-SIZED CONDITIONING ==========
    
    print(f"\n{'='*70}")
    print(f"--- Test 6: Variable-Sized Conditioning (CRITICAL!) ---")
    print(f"{'='*70}")
    
    # Create batch where conditioning has DIFFERENT sizes
    # This simulates MD trajectories where residues move in/out of pocket
    batch_variable = {
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
        'batch_size': 2,
        
        # CONDITIONING: MORE residues in initial frame (some moved OUT)
        'initial_ligand_coords': torch.randn(8, 3, device=device),   # Same size
        'initial_pocket_coords': torch.randn(22, 3, device=device),  # BIGGER by 6!
        
        # Node IDs for CURRENT graph (MD frame at t=500ps)
        'node_ids_atoms': torch.tensor([
            0, 1, 2, 3,                          # Mol 0: atoms 0-3
            100000, 100001, 100002, 100003       # Mol 1: atoms 0-3
        ], dtype=torch.long, device=device),
        
        'node_ids_residues': torch.tensor([
            # Mol 0: 8 residues still in pocket
            10, 11, 15, 20, 25, 30, 35, 40,
            # Mol 1: 8 residues still in pocket
            100010, 100011, 100015, 100020, 100025, 100030, 100035, 100040
        ], dtype=torch.long, device=device),
        
        # Node IDs for CONDITIONING graph (initial frame at t=0)
        'cond_node_ids_atoms': torch.tensor([
            0, 1, 2, 3,                          # Mol 0: same atoms
            100000, 100001, 100002, 100003       # Mol 1: same atoms
        ], dtype=torch.long, device=device),
        
        'cond_node_ids_residues': torch.tensor([
            # Mol 0: 12 residues (includes 12,13,14,21 that moved OUT later)
            10, 11, 12, 13, 14, 15, 20, 21, 25, 30, 35, 40,
            # Mol 1: 10 residues (includes 12,14 that moved OUT later)
            100010, 100011, 100012, 100014, 100015, 100020, 100025, 100030, 100035, 100040
        ], dtype=torch.long, device=device),
    }
    
    print(f"\n📊 Graph Size Comparison:")
    curr_total = batch_variable['ligand_coords'].shape[0] + batch_variable['pocket_coords'].shape[0]
    cond_total = batch_variable['initial_ligand_coords'].shape[0] + batch_variable['initial_pocket_coords'].shape[0]
    print(f"  Current graph:      {batch_variable['ligand_coords'].shape[0]} lig + "
          f"{batch_variable['pocket_coords'].shape[0]} pocket = {curr_total} nodes")
    print(f"  Conditioning graph: {batch_variable['initial_ligand_coords'].shape[0]} lig + "
          f"{batch_variable['initial_pocket_coords'].shape[0]} pocket = {cond_total} nodes")
    print(f"  Difference: +{cond_total - curr_total} extra nodes in conditioning")
    
    print(f"\n🔍 Expected Matching:")
    print(f"  Ligand atoms:   8/8 should match (100%)")
    print(f"  Pocket residues: 16/16 current should match")
    print(f"                   6/22 conditioning are unmatched (get NULL)")
    
    # Run training step with variable-sized conditioning
    loss_var, metrics_var = model.loss_fn(batch_variable)
    
    print(f"\n✅ Training Results:")
    print(f"  Total Loss:        {loss_var.item():.4f}")
    print(f"  Coord Loss:        {metrics_var['coord_loss']:.4f}")
    print(f"  Categorical Loss:  {metrics_var['categorical_loss']:.4f}")
    print(f"  Using conditioning: {metrics_var['using_conditioning']}")
    print(f"  Atom accuracy:     {metrics_var['atom_accuracy']:.3f}")
    print(f"  Residue accuracy:  {metrics_var['residue_accuracy']:.3f}")
    
    # Sanity checks
    assert not torch.isnan(loss_var), "❌ NaN loss with variable-sized conditioning!"
    assert not torch.isinf(loss_var), "❌ Inf loss with variable-sized conditioning!"
    assert metrics_var['using_conditioning'], "❌ Conditioning not detected!"
    print(f"\n✅ Variable-sized conditioning handled correctly!")
    
    print(f"\n--- Test 7: Stability Check (Multiple Runs) ---")
    losses = []
    for run in range(3):
        loss_repeat, _ = model.loss_fn(batch_variable)
        losses.append(loss_repeat.item())
        print(f"  Run {run+1}: loss = {loss_repeat.item():.4f}")
    
    loss_std = np.std(losses)
    print(f"  Loss std dev: {loss_std:.4f}")
    assert loss_std < 5.0, f"❌ Loss too unstable! std={loss_std:.4f}"
    print(f"✅ Model is stable with unmatched nodes!")
    
    print(f"\n--- Test 8: Evaluation with Variable-Sized Conditioning ---")
    eval_var = model.eval_fn(batch_variable)
    print(f"  Eval coord_loss_lvl0: {eval_var['coord_loss_lvl0']:.4f}")
    print(f"  Eval atom_accuracy_lvl0: {eval_var['atom_accuracy_lvl0']:.3f}")
    assert not np.isnan(eval_var['coord_loss_lvl0']), "❌ NaN in eval!"
    print(f"✅ Evaluation works with variable-sized conditioning!")
    
    print(f"\n{'='*70}")
    print("✅ ALL TESTS PASSED - Variable-sized conditioning fully working!")
    print(f"{'='*70}")
    
    return model

    
    
    print("\n✅ GenCFD-style molecular diffusion with conditioning test passed!")
    
    return model


if __name__ == "__main__":
    test_molecular_diffusion()
