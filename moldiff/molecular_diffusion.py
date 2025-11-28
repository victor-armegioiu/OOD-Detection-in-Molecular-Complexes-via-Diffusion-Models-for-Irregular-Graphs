"""
Molecular diffusion pipeline (GenCFD-style with categorical features).
"""

import dataclasses
import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, Tuple, Optional, Protocol, NamedTuple
from torch_scatter import scatter_mean

from moldiff.egnn_dynamics import EGNNDynamics, PreconditionedEGNNDynamics, ConditionalPreconditionedEGNNDynamics

Tensor = torch.Tensor
Numeric = float | int | Tensor
ScheduleFn = Callable[[Numeric], Numeric]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
    device: str = DEVICE

    def __call__(self, t: Numeric) -> Tensor:
        result = self.forward(t)
        if isinstance(result, (int, float)):
            result = torch.tensor(result, device=self.device, dtype=torch.float32)
        return result.to(self.device)


def _linear_rescale(in_min: float, in_max: float, out_min: float, out_max: float, device: str = DEVICE) -> InvertibleSchedule:
    in_range = in_max - in_min
    out_range = out_max - out_min
    fwd = lambda x: out_min + (x - in_min) / in_range * out_range
    inv = lambda y: in_min + (y - out_min) / out_range * in_range
    return InvertibleSchedule(fwd, inv, device=device)


def exponential_noise_schedule(clip_max: float = 100.0, base: float = np.e**0.5, start: float = 0.0, end: float = 5.0, device: str = DEVICE) -> InvertibleSchedule:

    if not (start < end and base > 1.0):
        raise ValueError("Must have `base` > 1 and `start` < `end`.")

    in_rescale = _linear_rescale(
        in_min=MIN_DIFFUSION_TIME,
        in_max=MAX_DIFFUSION_TIME,
        out_min=start,
        out_max=end,
        device=device,
    )
    out_rescale = _linear_rescale(in_min=base**start, in_max=base**end, out_min=0.0, out_max=clip_max, device=device)

    def sigma(t):
        t_tensor = torch.as_tensor(t, device=device, dtype=torch.float32)
        base_tensor = torch.tensor(base, device=device, dtype=torch.float32)
        return out_rescale(torch.pow(base_tensor, in_rescale(t_tensor)))

    def inverse(y):
        y_tensor = torch.as_tensor(y, device=device, dtype=torch.float32)
        base_tensor = torch.tensor(base, device=device, dtype=torch.float32)
        return in_rescale.inverse(torch.log(out_rescale.inverse(y_tensor)) / torch.log(base_tensor))

    return InvertibleSchedule(sigma, inverse, device=device)


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

    device: str = "cuda"

    @property
    def sigma_max(self) -> float:
        return self.sigma(MAX_DIFFUSION_TIME).item()

    def scale(self, t):
        """Scale function for compatibility with samplers (always returns 1.0 for variance exploding)"""
        return torch.ones_like(torch.as_tensor(t, device=self.device, dtype=torch.float32))

    def normalize_molecular_data(self, ligand_coords: Tensor, ligand_features: Tensor, pocket_coords: Tensor, pocket_features: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Apply molecular normalization (kept for compatibility)"""

        # Normalize coordinates
        ligand_coords_norm = ligand_coords / self.coord_norm
        pocket_coords_norm = pocket_coords / self.coord_norm

        # Normalize categorical features (one-hot -> continuous centered around 0)
        ligand_features_norm = (ligand_features.float() - self.feature_bias) / self.feature_norm
        pocket_features_norm = (pocket_features.float() - self.feature_bias) / self.feature_norm

        return ligand_coords_norm, ligand_features_norm, pocket_coords_norm, pocket_features_norm

    def unnormalize_molecular_data(
        self, ligand_coords: Tensor, ligand_features: Tensor, pocket_coords: Tensor, pocket_features: Tensor, discretize_features: bool = False, atom_nf: int = 10, residue_nf: int = 20
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
    def create_variance_exploding(  # the idea behind this class method is to create object, without having instantiated it before
        cls, sigma: InvertibleSchedule, coord_norm: float = 1.0, feature_norm: float = 1.0, feature_bias: float = 0.0, categorical_temperature: float = 1.0, device: str = DEVICE
    ) -> "MolecularDiffusion":
        """Create variance exploding scheme"""
        return cls(sigma=sigma, coord_norm=coord_norm, feature_norm=feature_norm, feature_bias=feature_bias, categorical_temperature=categorical_temperature, device=device)


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
            s0 = torch.rand((), dtype=torch.float32, device=scheme.device)
            num_elements = int(np.prod(shape))
            step_size = 1 / num_elements
            grid = torch.linspace(0, 1 - step_size, num_elements, dtype=torch.float32, device=scheme.device)
            samples = torch.remainder(grid + s0, 1).reshape(shape)
        else:
            samples = torch.rand(shape, dtype=torch.float32, device=scheme.device)

        log_min = torch.log(torch.tensor(clip_min, dtype=torch.float32, device=scheme.device))
        log_max = torch.log(torch.tensor(scheme.sigma_max, dtype=torch.float32, device=scheme.device))
        samples = (log_max - log_min) * samples + log_min
        return torch.exp(samples)

    return _noise_sampling


class NoiseLossWeighting(Protocol):
    def __call__(self, sigma: Tensor) -> Tensor: ...


def edm_weighting(data_std: float = 1.0, device=None) -> NoiseLossWeighting:
    """Weighting proposed in Karras et al. (https://arxiv.org/abs/2206.00364).

    This weighting ensures the effective weights are uniform across noise levels
    (see appendix B.6, eqns 139 to 144).

    Args:
      data_std: the standard deviation of the data.

    Returns:
      The weighting function.
    """

    def _weight_fn(sigma: Tensor) -> Tensor:
        return (torch.square(torch.tensor(data_std, device=device)) + torch.square(sigma)) / torch.square(data_std * sigma)

    return _weight_fn


# ********************
# Training Model (GenCFD-Style)
# ********************

class MolecularLossResults(NamedTuple):
    """Helper class to return losses from MolecularDenoisingModel._calculate_loss_from_noisy_embeddings"""
    total_loss: torch.Tensor
    coord_loss: torch.Tensor
    categorical_loss: torch.Tensor
    coord_loss_lig: torch.Tensor
    coord_loss_pocket: torch.Tensor
    categorical_loss_lig: torch.Tensor
    categorical_loss_pocket: torch.Tensor
    geometric_loss: torch.Tensor
    pred_coords_lig: torch.Tensor
    pred_coords_pocket: torch.Tensor
    pred_logits_lig: torch.Tensor
    pred_logits_pocket: torch.Tensor


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
    freeze_pocket_embeddings: bool = False
    # these EGNN parameters have been ignored so far, do we understand their meaning?
    attention: bool = False
    tanh:bool = False
    norm_constant: int = 0                  # this is probably handled independently by us
    inv_sublayers: int = 2                  # layers of the equivariant block
    sin_embedding: bool = False             # sin embedding for distances
    edge_cutoff_ligand: float|None = None 
    edge_cutoff_pocket: float|None = None 
    edge_cutoff_interaction: float|None = None
    reflection_equivariant: bool = True    # False would be SE(3) we're thus using E(3)

    # Loss weighting parameters
    coord_loss_weight: float = 1.0
    categorical_loss_weight: float = 1.0

    # Diffusion scheme
    scheme: MolecularDiffusion = None
    noise_sampling: NoiseLevelSampling = None
    noise_weighting: NoiseLossWeighting = None
    geometric_regularization: bool = True
    geom_loss_weight: float = 0.1
    device: str = DEVICE

    # Virtual Nodes for conditional sampling
    n_max_virtual_nodes: int = 0  # number of virtual nodes to add to pocket, 0 means no virtual nodes

    # # Whether to condition on pocket structure or train on complex jointly
    # pocket_conditioning: bool = False  # Whether to condition on pocket structure (always True for now)

    def __post_init__(self):
        """Initialize the denoiser and diffusion components"""

        # Create default scheme if not provided
        if self.scheme is None:
            sigma_schedule = exponential_noise_schedule(clip_max=100.0, device=self.device)
            scheme = MolecularDiffusion.create_variance_exploding(sigma=sigma_schedule, coord_norm=1.0, feature_norm=1.0, feature_bias=0.0, categorical_temperature=1.0, device=self.device)
            object.__setattr__(self, "scheme", scheme)

        # Create default noise sampling if not provided
        if self.noise_sampling is None:
            noise_sampling = log_uniform_sampling(self.scheme)
            object.__setattr__(self, "noise_sampling", noise_sampling)

        # Create default noise weighting if not provided
        if self.noise_weighting is None:
            noise_weighting = edm_weighting()
            object.__setattr__(self, "noise_weighting", noise_weighting)

        # Create the EGNN denoiser
        denoiser = EGNNDynamics(
            atom_nf=self.atom_nf,
            residue_nf=self.residue_nf,
            n_dims=self.n_dims,
            joint_nf=self.joint_nf,
            hidden_nf=self.hidden_nf,
            device=self.device,
            n_layers=self.n_layers,
            condition_time=True,
            update_pocket_coords=self.update_pocket_coords,
            freeze_pocket_embeddings=self.freeze_pocket_embeddings,
            edge_embedding_dim=self.edge_embedding_dim,
            geometric_regularization=self.geometric_regularization,
            geom_loss_weight=self.geom_loss_weight,
            # these EGNN parameters have been ignored so far, do we understand their meaning?
            attention=self.attention,
            tanh=self.tanh,
            norm_constant=self.norm_constant,                # this is probably handled independently by us
            inv_sublayers=self.inv_sublayers,                # layers of the equivariant block
            sin_embedding=self.sin_embedding,            # sin embedding for distances
            edge_cutoff_ligand=self.edge_cutoff_ligand, 
            edge_cutoff_pocket=self.edge_cutoff_pocket, 
            edge_cutoff_interaction=self.edge_cutoff_interaction,
            reflection_equivariant=self.reflection_equivariant,    # False would be SE(3) we're thus using E(3)
        )
        # to inherit to conditional class
        if not self.update_pocket_coords: object.__setattr__(self, "egnn_dynamics_net", denoiser)

        denoiser = PreconditionedEGNNDynamics(denoiser)

        from ema_pytorch import EMA

        denoise = EMA(denoiser, update_after_step=100)

        object.__setattr__(self, "denoiser", denoiser)
        

    def initialize(self):
        """Initialize model weights"""
        print(f"✅ Initialized MolecularDenoisingModel with {sum(p.numel() for p in self.denoiser.parameters())} parameters")

    def loss_fn(self, batch: dict):
        """
        CDCD-style loss function: noise embeddings, predict logits, use cross-entropy
        """

        # Extract molecular data
        lig_coords = batch["ligand_coords"].to(self.device)
        lig_features = batch["ligand_features"].to(self.device)  # One-hot features
        pocket_coords = batch["pocket_coords"].to(self.device)
        pocket_features = batch["pocket_features"].to(self.device)  # One-hot features
        lig_mask = batch["ligand_mask"].to(self.device)
        pocket_mask = batch["pocket_mask"].to(self.device)
        batch_size = len(torch.unique(torch.cat([lig_mask, pocket_mask])))

        # Get true atom/residue type indices for cross-entropy loss
        true_atom_types = torch.argmax(lig_features, dim=-1)  # [N_lig] - indices
        true_residue_types = torch.argmax(pocket_features, dim=-1)  # [N_pocket] - indices

        # Convert one-hot to embeddings using encoders (DIFFERENTIABLE!)
        lig_embeddings_clean = self.denoiser.atom_encoder(lig_features.float())  # [N_lig, joint_nf]
        pocket_embeddings_clean = self.denoiser.residue_encoder(pocket_features.float())  # [N_pocket, joint_nf]

        # Normalize embeddings (CDCD style)
        lig_embeddings_clean = F.normalize(lig_embeddings_clean, dim=-1)
        pocket_embeddings_clean = F.normalize(pocket_embeddings_clean, dim=-1)

        # Normalize coordinates
        lig_coords_norm = lig_coords / self.scheme.coord_norm
        pocket_coords_norm = pocket_coords / self.scheme.coord_norm

        lig_coords_centered, pocket_coords_centered, combined_mask = self._anchor_reference_frame(
            lig_coords_norm, 
            pocket_coords_norm, 
            lig_mask, 
            pocket_mask
        )
        
        # Concatenate coordinates and embeddings (NOT one-hot!)
        xh_lig_clean = torch.cat([lig_coords_centered, lig_embeddings_clean], dim=1)  # [N_lig, 3 + joint_nf]
        xh_pocket_clean = torch.cat([pocket_coords_centered, pocket_embeddings_clean], dim=1)  # [N_pocket, 3 + joint_nf]

        # Sample noise levels
        sigma = self.noise_sampling(shape=(batch_size,))  # [batch_size]

        xh_lig_noisy, xh_pocket_noisy = self._noise_clean_embeddings(
            xh_lig_clean, 
            xh_pocket_clean, 
            combined_mask, 
            lig_coords, 
            sigma, 
            lig_mask, 
            pocket_mask
        )

        # Forward pass through denoiser - outputs coordinates + logits
        # print('In train:')
        # print('Sigma shape:', sigma.shape)
        # print('xh_lig_noisy shape:', xh_lig_noisy.shape)
        # print('xh_pocket_noisy shape:', xh_pocket_noisy.shape)
        # print('xh_pocket_noisy shape:', xh_pocket_noisy.shape)
        # print('lig_mask', lig_mask.shape)
        # print('pocket_mask', pocket_mask.shape)

        pred_output_lig, pred_output_pocket = self.denoiser(
            xh_lig_noisy.to(self.device),
            xh_pocket_noisy.to(self.device),
            sigma.to(self.device) if hasattr(sigma, "to") else sigma,
            lig_mask,
            pocket_mask,
            target_atoms=lig_coords_centered,
            target_residues=pocket_coords_centered,
        )

        losses = self._calculate_loss_from_noisy_embeddings(
            pred_output_lig, 
            pred_output_pocket, 
            sigma, 
            lig_mask, 
            pocket_mask, 
            batch_size, 
            xh_lig_clean, 
            xh_pocket_clean, 
            true_atom_types, 
            true_residue_types
        )

        # Metrics
        metrics = {
            "loss": losses.total_loss.item(),
            "coord_loss": losses.coord_loss.item(),
            "categorical_loss": losses.categorical_loss.item(),
            "coord_loss_ligand": losses.coord_loss_lig.item(),
            "coord_loss_pocket": losses.coord_loss_pocket.item(),
            "categorical_loss_ligand": losses.categorical_loss_lig.item(),
            "categorical_loss_pocket": losses.categorical_loss_pocket.item(),
            "geometric_loss_total": losses.geometric_loss.item(),
            "avg_sigma": sigma.mean().item(),
            "max_sigma": sigma.max().item(),
            "coord_pred_scale": torch.cat([losses.pred_coords_lig, losses.pred_coords_pocket]).abs().mean().item(),
            "atom_accuracy": (torch.argmax(losses.pred_logits_lig, dim=-1) == true_atom_types).float().mean().item(),
            "residue_accuracy": (torch.argmax(losses.pred_logits_pocket, dim=-1) == true_residue_types).float().mean().item(),
            # debugging and testing:
            "pred_residue_coords": pred_output_pocket[:, :self.n_dims], # to double check that coords stay constant
            "input_residue_coords": xh_pocket_noisy[:, :self.n_dims],
        }
        return losses.total_loss, metrics

    def eval_fn(self, batch: dict) -> dict:
        """Evaluate denoising at multiple noise levels with proper loss breakdown"""

        # Extract data
        lig_coords = batch["ligand_coords"].to(self.device)
        lig_features = batch["ligand_features"].to(self.device)  # One-hot features
        pocket_coords = batch["pocket_coords"].to(self.device)
        pocket_features = batch["pocket_features"].to(self.device)  # One-hot features
        lig_mask = batch["ligand_mask"].to(self.device)
        pocket_mask = batch["pocket_mask"].to(self.device)
        batch_size = batch["batch_size"]

        # Get true categories for evaluation
        true_atom_types = torch.argmax(lig_features, dim=-1)
        true_residue_types = torch.argmax(pocket_features, dim=-1)

        # Cache training quantiles.
        if not hasattr(self, "_cached_sigma_levels"):
            training_sigmas = self.noise_sampling(shape=(10000,))
            sigma_levels = torch.quantile(training_sigmas, torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], device=self.device))
            object.__setattr__(self, "_cached_sigma_levels", sigma_levels)
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

            lig_coords_centered, pocket_coords_centered, combined_mask = self._anchor_reference_frame(
                lig_coords_norm, 
                pocket_coords_norm, 
                lig_mask, 
                pocket_mask
            )

            # Concatenate coordinates and embeddings (NOT one-hot!)
            xh_lig_clean = torch.cat([lig_coords_centered, lig_embeddings_clean], dim=1)
            xh_pocket_clean = torch.cat([pocket_coords_centered, pocket_embeddings_clean], dim=1)

            # Create sigma tensor for this evaluation (broadcast to batch_size)
            sigma_batch = torch.full((batch_size, ), sigma_val, device = self.device)

            xh_lig_noisy, xh_pocket_noisy = self._noise_clean_embeddings(
                xh_lig_clean, 
                xh_pocket_clean, 
                combined_mask, 
                lig_coords, 
                sigma_batch, 
                lig_mask, 
                pocket_mask
            )



            # print('In eval:')
            # print('Noise level:', i, sigma_val)
            # print('Sigma shape:', sigma_batch.shape)
            # print('xh_lig_noisy shape:', xh_lig_noisy.shape)
            # print('xh_pocket_noisy shape:', xh_pocket_noisy.shape)
            # print('xh_pocket_noisy shape:', xh_pocket_noisy.shape)
            # print('lig_mask', lig_mask.shape)
            # print('pocket_mask', pocket_mask.shape)

            # Denoise
            with torch.no_grad():
                pred_output_lig, pred_output_pocket = self.denoiser(xh_lig_noisy, xh_pocket_noisy, sigma_batch, lig_mask, pocket_mask)

            losses = self._calculate_loss_from_noisy_embeddings(
                pred_output_lig, 
                pred_output_pocket, 
                sigma_batch, 
                lig_mask, 
                pocket_mask, 
                batch_size, 
                xh_lig_clean, 
                xh_pocket_clean, 
                true_atom_types, 
                true_residue_types
            )

            # Accuracy metrics
            atom_accuracy = (torch.argmax(losses.pred_logits_lig, dim=-1) == true_atom_types).float().mean()
            residue_accuracy = (torch.argmax(losses.pred_logits_pocket, dim=-1) == true_residue_types).float().mean()

            eval_losses[f"coord_loss_lvl{i}"] = losses.coord_loss.item()
            eval_losses[f"categorical_loss_lvl{i}"] = losses.categorical_loss.item()
            eval_losses[f"total_loss_lvl{i}"] = losses.coord_loss.item() + losses.categorical_loss.item()
            eval_losses[f"atom_accuracy_lvl{i}"] = atom_accuracy.item()
            eval_losses[f"residue_accuracy_lvl{i}"] = residue_accuracy.item()

        return eval_losses
    
    def _anchor_reference_frame(self, lig_coords_norm, pocket_coords_norm, lig_mask, pocket_mask):

        # Remove center of mass for translation invariance
        combined_coords = torch.cat([lig_coords_norm, pocket_coords_norm], dim=0)
        combined_mask = torch.cat([lig_mask, pocket_mask], dim=0)
        combined_coords_centered = remove_mean_batch(combined_coords, combined_mask)

        lig_coords_centered = combined_coords_centered[: len(lig_coords_norm)]
        pocket_coords_centered = combined_coords_centered[len(lig_coords_norm) :]

        return lig_coords_centered, pocket_coords_centered, combined_mask


    def _noise_clean_embeddings(self, xh_lig_clean, xh_pocket_clean, combined_mask, lig_coords, sigma, lig_mask, pocket_mask) -> Tuple:

        # Generate Gaussian noise for the entire state
        noise_lig = torch.randn_like(xh_lig_clean, device=self.device)
        noise_pocket = torch.randn_like(xh_pocket_clean, device=self.device)

        # Make coordinate noise COM-free (but not embedding noise)
        noise_coords_combined = torch.cat([noise_lig[:, : self.n_dims], noise_pocket[:, : self.n_dims]], dim=0)
        noise_coords_centered = remove_mean_batch(noise_coords_combined, combined_mask)
        noise_lig[:, : self.n_dims] = noise_coords_centered[: len(lig_coords)]
        noise_pocket[:, : self.n_dims] = noise_coords_centered[len(lig_coords) :]


        # Add noise
        sigma_expanded_lig = sigma[lig_mask].unsqueeze(1)  # [N_lig, 1]
        sigma_expanded_pocket = sigma[pocket_mask].unsqueeze(1)  # [N_pocket, 1]

        xh_lig_noisy = xh_lig_clean + sigma_expanded_lig * noise_lig
        xh_pocket_noisy = xh_pocket_clean + sigma_expanded_pocket * noise_pocket

        return xh_lig_noisy, xh_pocket_noisy
    
    def _calculate_loss_from_noisy_embeddings(
            self, 
            pred_output_lig, 
            pred_output_pocket, 
            sigma, 
            lig_mask, 
            pocket_mask, 
            batch_size, 
            xh_lig_clean, 
            xh_pocket_clean, 
            true_atom_types, 
            true_residue_types

    ) -> MolecularLossResults:
        # Split predictions: coordinates + logits
        pred_coords_lig = pred_output_lig[:, : self.n_dims]  # [N_lig, 3]
        pred_logits_lig = pred_output_lig[:, self.n_dims :]  # [N_lig, atom_nf] - LOGITS
        pred_coords_pocket = pred_output_pocket[:, : self.n_dims]  # [N_pocket, 3]
        pred_logits_pocket = pred_output_pocket[:, self.n_dims :]  # [N_pocket, residue_nf] - LOGITS

        # Loss weighting
        weights = self.noise_weighting(sigma)  # [batch_size]
        weights_lig = weights[lig_mask].unsqueeze(1)  # [N_lig, 1]
        weights_pocket = weights[pocket_mask].unsqueeze(1) 

        # Coordinate loss: L2 between predicted and clean coordinates
        clean_coords_lig = xh_lig_clean[:, : self.n_dims]
        clean_coords_pocket = xh_pocket_clean[:, : self.n_dims]

        coord_loss_lig = torch.mean(weights_lig * (pred_coords_lig - clean_coords_lig) ** 2)
        coord_loss_pocket = torch.mean(weights_pocket * (pred_coords_pocket - clean_coords_pocket) ** 2)
        coord_loss = coord_loss_lig + coord_loss_pocket

        # Categorical loss: Cross-entropy between logits and true atom types
        ce_loss_lig = F.cross_entropy(pred_logits_lig, true_atom_types, reduction="none")
        ce_loss_pocket = F.cross_entropy(pred_logits_pocket, true_residue_types, reduction="none")

        # Weight categorical loss by noise level (like coordinates)
        # categorical_loss_lig = torch.mean(weights[lig_mask] * ce_loss_lig)
        # categorical_loss_pocket = torch.mean(weights[pocket_mask] * ce_loss_pocket)
        # categorical_loss = categorical_loss_lig + categorical_loss_pocket

        categorical_loss_lig = ce_loss_lig.mean()
        categorical_loss_pocket = ce_loss_pocket.mean()
        categorical_loss = categorical_loss_lig + categorical_loss_pocket
        geometric_loss = self.denoiser.last_geometric_loss

        # Combine losses
        total_loss = self.coord_loss_weight * coord_loss + self.categorical_loss_weight * categorical_loss + geometric_loss

        return MolecularLossResults(
            total_loss,
            coord_loss,
            categorical_loss,
            coord_loss_lig,
            coord_loss_pocket,
            categorical_loss_lig,
            categorical_loss_pocket,
            geometric_loss,
            pred_coords_lig,
            pred_coords_pocket,
            pred_logits_lig,
            pred_logits_pocket,
        )
    
    # dataclass(frozen=True, kw_only=True)
class ConditionalMolecularDenoisingModel(MolecularDenoisingModel): 
    
    def __post_init__(self):
        """Initializes the noiser and denoiser components"""        
        super().__post_init__()

        # check denoiser and attach to object
        assert not self.denoiser.egnn_dynamics.update_pocket_coords, \
        "Conditional Denoiser cannot be used with EGNN.update_pocket_coords = True"

        denoiser = ConditionalPreconditionedEGNNDynamics(self.egnn_dynamics_net)
        object.__setattr__(self, "denoiser", denoiser)



    def initialize(self):
        """Initialize model weights"""
        print(f"✅ Initialized ConditionalMolecularDenoisingModel with {sum(p.numel() for p in self.denoiser.parameters())} parameters")
    
    def null_residue_loss_fn(self, batch: dict):
        """
        CDCD-style loss function for null residue class for CFG: noise embeddings, predict logits, use cross-entropy
        """

        # Extract molecular data
        lig_coords = batch["ligand_coords"].to(self.device)
        lig_features = batch["ligand_features"].to(self.device)  # One-hot features
        lig_mask = batch["ligand_mask"].to(self.device)
        batch_size = len(torch.unique(lig_mask))

        # Get true atom/residue type indices for cross-entropy loss
        true_atom_types = torch.argmax(lig_features, dim=-1)  # [N_lig] - indices

        # Convert one-hot to embeddings using encoders (DIFFERENTIABLE!)
        lig_embeddings_clean = self.denoiser.atom_encoder(lig_features.float())  # [N_lig, joint_nf]

        # Normalize embeddings (CDCD style) and coordinates
        lig_embeddings_clean = F.normalize(lig_embeddings_clean, dim=-1)
        lig_coords_norm = lig_coords / self.scheme.coord_norm


        # No COM removal required

        # Concatenate coordinates and embeddings (NOT one-hot!)
        xh_lig_clean = torch.cat([lig_coords_norm, lig_embeddings_clean], dim=1)  # [N_lig, 3 + joint_nf]
   
        # Sample noise levels
        sigma = self.noise_sampling(shape=(batch_size,))  # [batch_size]

        xh_lig_noisy = self._noise_clean_embeddings_null_residue(xh_lig_clean, sigma, lig_mask)

        # Forward pass through denoiser - outputs coordinates + logits
        # print('In train:')
        # print('Sigma shape:', sigma.shape)
        # print('xh_lig_noisy shape:', xh_lig_noisy.shape)
        # print('xh_pocket_noisy shape:', xh_pocket_noisy.shape)
        # print('xh_pocket_noisy shape:', xh_pocket_noisy.shape)
        # print('lig_mask', lig_mask.shape)
        # print('pocket_mask', pocket_mask.shape)

        pred_output_lig = self.denoiser.null_residue_forward(
            xh_atoms = xh_lig_noisy.to(self.device), 
            sigma = sigma.to(self.device) if hasattr(sigma, "to") else sigma, 
            mask_atoms = lig_mask, 
            target_atoms=lig_coords_norm
        )

        losses = self._calculate_loss_from_noisy_embeddings_null_residue(
            pred_output_lig, 
            sigma, 
            lig_mask,
            xh_lig_clean, 
            true_atom_types
        )

        # Metrics
        metrics = {
            "loss": losses.total_loss.item(),
            "coord_loss": losses.coord_loss.item(),
            "categorical_loss": losses.categorical_loss.item(),
            "coord_loss_ligand": losses.coord_loss_lig.item(),
            "coord_loss_pocket": float("nan"),
            "categorical_loss_ligand": losses.categorical_loss_lig.item(),
            "categorical_loss_pocket": float("nan"),
            "geometric_loss_total": losses.geometric_loss.item(),
            "avg_sigma": sigma.mean().item(),
            "max_sigma": sigma.max().item(),
            "coord_pred_scale": losses.pred_coords_lig.abs().mean().item(),
            "atom_accuracy": (torch.argmax(losses.pred_logits_lig, dim=-1) == true_atom_types).float().mean().item(),
            "residue_accuracy": float("nan"),
        }
        return losses.total_loss, metrics
    
    
    def _anchor_reference_frame(self, lig_coords_norm, pocket_coords_norm, lig_mask, pocket_mask):
        """Conditional modeling requires no COM removal"""

        # POCKET-COM
        combined_mask = torch.cat([lig_mask, pocket_mask], dim=0)

        # Subtract pocket center of mass
        pocket_com = scatter_mean(pocket_coords_norm, pocket_mask, dim=0)
        lig_coords_zero_pocket_com = lig_coords_norm - pocket_com[lig_mask]
        pocket_coords_zero_pocket_com = pocket_coords_norm - pocket_com[pocket_mask]


        return lig_coords_zero_pocket_com, pocket_coords_zero_pocket_com, combined_mask



    def _noise_clean_embeddings(self, xh_lig_clean, xh_pocket_clean, combined_mask, lig_coords, sigma, lig_mask, pocket_mask) -> Tuple:
    
        # Generate Gaussian noise for the entire state
        noise_lig = torch.randn_like(xh_lig_clean, device=self.device)
        # embedding_noise_pocket = torch.randn_like(xh_pocket_clean[:, self.n_dims:], device=self.device)

        # Add noise
        sigma_expanded_lig = sigma[lig_mask].unsqueeze(1)  # [N_lig, 1]
        # sigma_expanded_pocket = sigma[pocket_mask].unsqueeze(1)  # [N_pocket, 1]

        xh_lig_noisy = xh_lig_clean + sigma_expanded_lig * noise_lig
        xh_pocket_noisy = xh_pocket_clean.clone() # in pocket conditioning only embeddings are noised
        # xh_pocket_noisy[:, self.n_dims:] = xh_pocket_clean[:, self.n_dims:] + sigma_expanded_pocket * embedding_noise_pocket

        # # DEBUG PRINT
        # print(f"TRaining: Pocket State BEFORE EGNN Pass:\n{xh_pocket_noisy}")

        return xh_lig_noisy, xh_pocket_noisy
    
    def _calculate_loss_from_noisy_embeddings(
            self, 
            pred_output_lig, 
            pred_output_pocket, 
            sigma, 
            lig_mask, 
            pocket_mask, 
            batch_size, 
            xh_lig_clean, 
            xh_pocket_clean, 
            true_atom_types, 
            true_residue_types

    ) -> MolecularLossResults:
        
        # # DEBUG PRINT
        # print(f"Training: Pocket State AFTER EGNN Pass:\n{pred_output_pocket}")

        # TODO: clarify whether categorical loss on pocket embeddings stills applies, 
        # so far all I have changed is the coord loss of pocket to be zero

        # Split predictions: coordinates + logits
        pred_coords_lig = pred_output_lig[:, :self.n_dims]  # [N_lig, 3]
        pred_logits_lig = pred_output_lig[:, self.n_dims :]  # [N_lig, atom_nf] - LOGITS
        pred_coords_pocket = pred_output_pocket[:, : self.n_dims]  # [N_pocket, 3]
        pred_logits_pocket = pred_output_pocket[:, self.n_dims :]  # [N_pocket, residue_nf] - LOGITS

        # Loss weighting
        weights = self.noise_weighting(sigma)  # [batch_size]
        weights_lig = weights[lig_mask].unsqueeze(1)  # [N_lig, 1]
        # weights_pocket = weights[pocket_mask].unsqueeze(1) 

        # Coordinate loss: L2 between predicted and clean coordinates
        clean_coords_lig = xh_lig_clean[:, : self.n_dims]
        # clean_coords_pocket = xh_pocket_clean[:, : self.n_dims]

        coord_loss_lig = torch.mean(weights_lig * (pred_coords_lig - clean_coords_lig) ** 2)
        coord_loss_pocket = torch.zeros_like(coord_loss_lig) # torch.mean(weights_pocket * (pred_coords_pocket - clean_coords_pocket) ** 2)
        coord_loss = coord_loss_lig + coord_loss_pocket

        # Categorical loss: Cross-entropy between logits and true atom types
        ce_loss_lig = F.cross_entropy(pred_logits_lig, true_atom_types, reduction="none")
        # Categorical Loss is zero is embeddings frozen
        ce_loss_pocket = torch.zeros_like(ce_loss_lig) # F.cross_entropy(pred_logits_pocket, true_residue_types, reduction="none") if not self.freeze_pocket_embeddings else  torch.zeros_like(ce_loss_lig) 

        # Weight categorical loss by noise level (like coordinates)
        # categorical_loss_lig = torch.mean(weights[lig_mask] * ce_loss_lig)
        # categorical_loss_pocket = torch.mean(weights[pocket_mask] * ce_loss_pocket)
        # categorical_loss = categorical_loss_lig + categorical_loss_pocket

        categorical_loss_lig = ce_loss_lig.mean()
        categorical_loss_pocket = ce_loss_pocket.mean()
        categorical_loss = categorical_loss_lig + categorical_loss_pocket
        geometric_loss = self.denoiser.last_geometric_loss

        # Combine losses
        total_loss = self.coord_loss_weight * coord_loss + self.categorical_loss_weight * categorical_loss + geometric_loss

        return MolecularLossResults(
            total_loss,
            coord_loss,
            categorical_loss,
            coord_loss_lig,
            coord_loss_pocket,
            categorical_loss_lig,
            categorical_loss_pocket,
            geometric_loss,
            pred_coords_lig,
            pred_coords_pocket,
            pred_logits_lig,
            true_residue_types, # instead of predicted return true (no need to predict pocket)
        )

    
    def _noise_clean_embeddings_null_residue(self, xh_lig_clean, sigma, lig_mask) -> Tuple:
    
        # Generate Gaussian noise for the entire state
        noise_lig = torch.randn_like(xh_lig_clean, device=self.device)

        # # Make coordinate noise for ligands COM-free (but not embedding noise))
        # noise_lig[:, :self.n_dims] = remove_mean_batch(noise_lig[:, :self.n_dims], lig_mask)

        # Add noise
        sigma_expanded_lig = sigma[lig_mask].unsqueeze(1)  # [N_lig, 1]

        xh_lig_noisy = xh_lig_clean + sigma_expanded_lig * noise_lig

        return xh_lig_noisy
            
    def _calculate_loss_from_noisy_embeddings_null_residue(
            self, 
            pred_output_lig, 
            sigma, 
            lig_mask, 
            xh_lig_clean, 
            true_atom_types

    ) -> MolecularLossResults:

        # Split predictions: coordinates + logits
        pred_coords_lig = pred_output_lig[:, :self.n_dims]  # [N_lig, 3]
        pred_logits_lig = pred_output_lig[:, self.n_dims :]  # [N_lig, atom_nf] - LOGITS

        # Loss weighting
        weights = self.noise_weighting(sigma)  # [batch_size]
        weights_lig = weights[lig_mask].unsqueeze(1)  # [N_lig, 1]

        # Coordinate loss for ligand only: L2 between predicted and clean coordinates
        clean_coords_lig = xh_lig_clean[:, : self.n_dims]

        coord_loss_lig = torch.mean(weights_lig * (pred_coords_lig - clean_coords_lig) ** 2)
        coord_loss = coord_loss_lig.clone() # for consistency

        # Categorical loss: Cross-entropy between logits and true atom types
        ce_loss_lig = F.cross_entropy(pred_logits_lig, true_atom_types, reduction="none")

        # Weight categorical loss by noise level (like coordinates)
        # categorical_loss_lig = torch.mean(weights[lig_mask] * ce_loss_lig)
        # categorical_loss_pocket = torch.mean(weights[pocket_mask] * ce_loss_pocket)
        # categorical_loss = categorical_loss_lig + categorical_loss_pocket

        categorical_loss_lig = ce_loss_lig.mean()
        categorical_loss = categorical_loss_lig.clone() # for consistency 
        geometric_loss = self.denoiser.last_geometric_loss

        # Combine losses
        total_loss = self.coord_loss_weight * coord_loss + self.categorical_loss_weight * categorical_loss + geometric_loss

        nan_tensor = torch.full_like(pred_coords_lig[:, :1], float("nan")) # nan tensor of random shape to fullfill typing

        return MolecularLossResults(
            total_loss,
            coord_loss,
            categorical_loss,
            coord_loss_lig,
            nan_tensor,
            categorical_loss_lig,
            nan_tensor,
            geometric_loss,
            pred_coords_lig,
            nan_tensor,
            pred_logits_lig,
            nan_tensor
        )



def test_molecular_diffusion():
    """Test the GenCFD-style molecular diffusion implementation"""

    print("Testing GenCFD-Style Molecular Diffusion (CUDA)...")

    # Create test batch (all on CUDA)
    device = "cuda"
    batch_size = 2
    atom_nf = 5
    residue_nf = 7

    batch = {
        "ligand_coords": torch.randn(8, 3, device=device),
        "ligand_features": torch.nn.functional.one_hot(torch.randint(0, atom_nf, (8,), device=device), atom_nf).float(),
        "pocket_coords": torch.randn(16, 3, device=device),
        "pocket_features": torch.nn.functional.one_hot(torch.randint(0, residue_nf, (16,), device=device), residue_nf).float(),
        "ligand_mask": torch.cat([torch.zeros(4), torch.ones(4)]).long().to(device),
        "pocket_mask": torch.cat([torch.zeros(8), torch.ones(8)]).long().to(device),
        "batch_size": 2,
    }

    print(f"--- Testing GenCFD-style loss ---")

    # Create model
    sigma_schedule = exponential_noise_schedule(clip_max=100.0)  # Smaller max
    scheme = MolecularDiffusion.create_variance_exploding(sigma=sigma_schedule, categorical_temperature=1.0)

    model = MolecularDenoisingModel(atom_nf=atom_nf, residue_nf=residue_nf, joint_nf=8, hidden_nf=32, n_layers=2, scheme=scheme, coord_loss_weight=2.0, categorical_loss_weight=1.0)

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

def test_conditional_molecular_diffusion():
    """Test the conditional molecular diffusion implementation"""

    print("Testing GenCFD-Style Molecular Diffusion (CUDA)...")

    if not torch.cuda.is_available():
        raise RuntimeError("Cuda is not available!")

    # Create test batch (all on CUDA)
    device = "cuda"
    batch_size = 2
    atom_nf = 5
    residue_nf = 7

    batch = {
        "ligand_coords": torch.randn(8, 3, device=device),
        "ligand_features": torch.nn.functional.one_hot(torch.randint(0, atom_nf, (8,), device=device), atom_nf).float(),
        "pocket_coords": torch.randn(16, 3, device=device),
        "pocket_features": torch.nn.functional.one_hot(torch.randint(0, residue_nf, (16,), device=device), residue_nf).float(),
        "ligand_mask": torch.cat([torch.zeros(4), torch.ones(4)]).long().to(device),
        "pocket_mask": torch.cat([torch.zeros(8), torch.ones(8)]).long().to(device),
        "batch_size": 2,
    }
    combined_mask = torch.cat([batch["ligand_mask"], batch["pocket_mask"]], dim=0)
    batch["com"] = scatter_mean(torch.cat([batch["ligand_coords"], batch["pocket_coords"]], dim=0), combined_mask, dim=0)
    

    # Create model
    sigma_schedule = exponential_noise_schedule(clip_max=100.0)  # Smaller max
    scheme = MolecularDiffusion.create_variance_exploding(sigma=sigma_schedule, categorical_temperature=1.0)

    model = ConditionalMolecularDenoisingModel(
        atom_nf=atom_nf, 
        residue_nf=residue_nf, 
        joint_nf=8, 
        hidden_nf=32, 
        n_layers=2,
        scheme=scheme, 
        update_pocket_coords = False,
        coord_loss_weight=2.0, 
        categorical_loss_weight=1.0)

    model.initialize()

    assert not model.denoiser.egnn_dynamics.update_pocket_coords, "Pocket coords are updated in denoiser"

    print(f"--- Testing conditional loss with residues---")

    # Test loss computation
    loss, metrics = model.loss_fn(batch)
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Coord Loss: {metrics['coord_loss']:.4f}")
    print(f"Categorical Loss: {metrics['categorical_loss']:.4f}")
    print(f"Sigma range: [{metrics['avg_sigma']:.3f}, {metrics['max_sigma']:.3f}]")
    print(f"Coord prediction scale: {metrics['coord_pred_scale']:.3f}")
    # assert stable pocket coords: input stable
    assert torch.allclose(metrics["input_residue_coords"], batch["pocket_coords"], atol=1e6, rtol=1e5), \
        f"Input coords to network don't match batch input: \n Input: {batch['pocket_coords']} \n Output: {metrics['input_residue_coords']}"
    # assert stable pocket coords: out stable
    assert torch.allclose(metrics["pred_residue_coords"], batch["pocket_coords"], atol=1e6, rtol=1e5), \
        f"Pocket coord drift by network:\n Input: {batch['pocket_coords']} \n Output: {metrics['pred_residue_coords']}"

    print(f"--- Testing null residue loss ---")
    loss, metrics = model.null_residue_loss_fn(batch)
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
    # test_molecular_diffusion()
    test_conditional_molecular_diffusion()
