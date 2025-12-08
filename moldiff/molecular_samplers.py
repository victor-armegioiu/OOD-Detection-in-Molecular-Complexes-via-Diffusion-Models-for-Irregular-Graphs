"""
Molecular SDE Samplers

Implements the same SDE sampling logic as the base framework but natively for molecular systems.
Follows exact reverse SDE dynamics with automatic differentiation, all CUDA-optimized.
"""
import wandb # remove after debug

import torch
import numpy as np
from typing import Tuple, Optional, Callable, Mapping, Any, NamedTuple, Protocol, Dict
from torch.autograd import grad
from torch_scatter import scatter_mean
import torch.nn.functional as F


from moldiff.molecular_diffusion import MolecularDiffusion, remove_mean_batch, MolecularDenoisingModel, ConditionalMolecularDenoisingModel
import moldiff.molecular_diffusion as molecular_diffusion

Tensor = torch.Tensor
TensorMapping = Mapping[str, Tensor]
MolecularParams = Mapping[str, Any]

def add_back_pocket_com(state, pc, n_dims):
    
    # add back the pocket com 
    if state.ligand.dim() == 2:
        state.ligand[:, :n_dims] += pc[state.ligand_mask]
        state.pocket[:, :n_dims] += pc[state.pocket_mask]
    else:  # [T, N, D]
        T = state.ligand.shape[0]
        state.ligand[:, :, :n_dims] += pc[state.ligand_mask].unsqueeze(0).expand(T, -1, -1)
        state.pocket[:, :, :n_dims] += pc[state.pocket_mask].unsqueeze(0).expand(T, -1, -1)
    return state


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
    def __init__(self, n_dims: int = 3, com_mode: str = "joint"):
        self.n_dims = n_dims
        self.com_mode = com_mode

        assert com_mode in ["joint", "cond"], "com_mode must be in [joint, cond]"
    
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
        
        # Make coordinate noise COM-free (molecular-specific constraint) if we model jointly
        if self.n_dims > 0 and self.com_mode == "joint":
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
        
        # xt+dt​=xt​+f(xt​,t)dt+g(xt​,t)ξdt
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

        # if com_mode == "cond": 
        #     assert torch.allclose(new_pocket, molecular_state.pocket)
        
        return MolecularState(
            ligand=new_ligand,
            pocket=new_pocket,
            ligand_mask=molecular_state.ligand_mask,
            pocket_mask=molecular_state.pocket_mask,
            batch_size=molecular_state.batch_size
        )
    

class MolecularEulerMaruyama(MolecularEulerMaruyamaStep):
    
    def __init__(self, time_axis_pos: int = 0, terminal_only: bool = False, n_dims: int = 3, com_mode: str = "joint"):
        super().__init__(n_dims, com_mode)
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
            # only if modeling jointly
            if self.com_mode == "joint":
                combined_mask = torch.cat([current_state.ligand_mask, current_state.pocket_mask])
                combined_coords_centered = remove_mean_batch(combined_coords, combined_mask)
            else: 
                combined_coords_centered = combined_coords.clone()
                
            # Update state with centered coordinates
            new_ligand = torch.cat([
                combined_coords_centered[:len(current_state.ligand)],
                current_state.ligand[:, self.n_dims:] # dicrete features
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
        joint_nf: int = 16,
    ):
        self.ligand_sizes = ligand_sizes
        self.pocket_sizes = pocket_sizes
        self.scheme = scheme
        self.denoise_fn = denoise_fn
        self.tspan = tspan.cuda() if torch.cuda.is_available() else tspan.cpu()
        self.apply_denoise_at_end = apply_denoise_at_end
        self.return_full_paths = return_full_paths
        self.n_dims = n_dims
        self.atom_nf = atom_nf
        self.residue_nf = residue_nf
        self.joint_nf = joint_nf
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
        
        # # DEBUG PRINT
        # print(f"Conditional Batch:\n{cond['pocket_state']}")
        
        # Create initial molecular state.
        molecular_state1 = self._create_initial_noisy_state()

        if cond is not None:
            assert all([key in cond.keys() for key in ["pocket_state", "pocket_mask", "pocket_com"]])
            # first, set ligand noise relative to pocket reference frame
            molecular_state1.ligand[:, :self.n_dims] -= cond["pocket_com"][molecular_state1.ligand_mask]

            # second, attach the zero com pocket and ligand noise states
            molecular_state1 = molecular_state1._replace(
                pocket=cond["pocket_state"],
                pocket_mask=cond["pocket_mask"]
            )
            
        
        # # DEBUG PRINT
        # print(f"Pocket State going into self.denoise in generate :\n{molecular_state1.pocket}")

        
        # # DEBUG PRINT
        # print(f"Pocket State going into self.denoise in generate :\n{molecular_state1.pocket}")

        
        # Denoise iteratively
        denoised_state = self.denoise(
            noisy_state=molecular_state1,
            tspan=self.tspan,
            cond=cond
        )

        # # DEBUG PRINT
        # print(f"Pocket State coming out of self.denoise in generate :\n{denoised_state.pocket}")

        # DEBUG PRINT
        if cond is not None:
            print("Checking if pocket states remain unchanged throughout sampling...")
            test1 = torch.allclose(cond['pocket_state'], molecular_state1.pocket)
            test2 = torch.allclose(cond['pocket_state'], denoised_state.pocket)
            test3 = torch.allclose(molecular_state1.pocket, denoised_state.pocket)
            print(f"Test results:\n  cond vs initial: {test1}\n  cond vs denoised: {test2}\n  initial vs denoised: {test3}")
            if test1 and test2 and test3:
                print("✅ Pocket state remained constant throughout sampling!")
            else:
                print("❌ Pocket state changed during sampling!")
        
        
        
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

        # add back pocket com
        if cond is not None and "pocket_com" in cond:
            current_pocket_coms = scatter_mean(denoised_state.pocket[:, :self.n_dims], 
                                               denoised_state.pocket_mask, dim = 0)
            assert torch.allclose(current_pocket_coms, torch.zeros_like(current_pocket_coms), atol=1e-4, rtol = 1e-4), \
                    f"Pocket not zero-COM before add-back; check anchoring:\n{current_pocket_coms}"
            
            denoised_state = add_back_pocket_com(
                denoised_state, 
                cond["pocket_com"], # shape [B, 3], in normalized units
                self.n_dims)
        
        # Convert to molecular batch format.
        samples = self._molecular_state_to_batch(denoised_state, discretize_features=True)

        return samples
    
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
            torch.full((size,), i, dtype=torch.long, device=self.tspan.device)
            for i, size in enumerate(self.ligand_sizes)
        ])
        pocket_mask = torch.cat([
            torch.full((size,), i, dtype=torch.long, device=self.tspan.device)
            for i, size in enumerate(self.pocket_sizes)
        ])
        
        # Sample initial noise (following x1 = randn() * sigma(1) * scale(1))
        initial_sigma = self.scheme.sigma(self.tspan[0])
        initial_scale = self.scheme.scale(self.tspan[0])
        
        ligand_noise = (
            torch.randn(total_lig_atoms, self.n_dims + self.joint_nf, device=self.tspan.device) 
            * initial_sigma * initial_scale
        )
        pocket_noise = (
            torch.randn(total_pocket_atoms, self.n_dims + self.joint_nf, device=self.tspan.device)
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
            ligand=ligand_noise / self.scheme.coord_norm,
            pocket=pocket_noise / self.scheme.coord_norm,
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
        
        if discretize_features:
            # OPTION 1: Use the embeddings to get probabilities via similarity
            with torch.no_grad():
                # Get the vocabulary embeddings
                atom_embeddings = self._model.denoiser.egnn_dynamics.atom_encoder(
                    torch.eye(self.atom_nf, device=lig_features.device)
                )
                atom_embeddings = F.normalize(atom_embeddings, dim=-1)
                
                residue_embeddings = self._model.denoiser.egnn_dynamics.residue_encoder(
                    torch.eye(self.residue_nf, device=pocket_features.device)
                )
                residue_embeddings = F.normalize(residue_embeddings, dim=-1)
                
                # Normalize current embeddings
                lig_features_norm = F.normalize(lig_features, dim=-1)
                pocket_features_norm = F.normalize(pocket_features, dim=-1)
                
                # Compute similarities (this preserves the diversity!)
                lig_similarities = torch.matmul(lig_features_norm, atom_embeddings.T)
                pocket_similarities = torch.matmul(pocket_features_norm, residue_embeddings.T)
                
                # Convert to one-hot
                lig_types = torch.argmax(lig_similarities, dim=-1)
                pocket_types = torch.argmax(pocket_similarities, dim=-1)
                
                ligand_features_final = F.one_hot(lig_types, self.atom_nf).float()
                pocket_features_final = F.one_hot(pocket_types, self.residue_nf).float()
        else:
            ligand_features_final = lig_features
            pocket_features_final = pocket_features
        
        # Unnormalize coordinates
        lig_coords_unnorm = lig_coords * self.scheme.coord_norm
        pocket_coords_unnorm = pocket_coords * self.scheme.coord_norm
        
        return {
            'ligand_coords': lig_coords_unnorm,
            'ligand_features': ligand_features_final,
            'pocket_coords': pocket_coords_unnorm,
            'pocket_features': pocket_features_final,
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
        joint_nf: int = 16,
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
            joint_nf=joint_nf,
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
            # Note that d/dt log(s) = d/dt s / s
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
                ligand=molecular_state.ligand,
                pocket=molecular_state.pocket,
                ligand_mask=molecular_state.ligand_mask,
                pocket_mask=molecular_state.pocket_mask,
                batch_size=molecular_state.batch_size
            )
            
            # Get denoiser output
            denoiser_output = self.denoise_fn(normalized_state, sigma, params.get("cond"))
            
            # Compute drift using the exact formula: [2σ̇/σ + ṡ/s]x - [2sσ̇/σ]D(x/s, σ)
            drift_coeff = 2 * dlog_sigma_dt + dlog_s_dt
            denoiser_coeff = 2 * dlog_sigma_dt * s
            
            # Ensure correct shape for coefficients of protein and ligand
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
    
class ConditionalMolecularSdeSampler(MolecularSdeSampler): 

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
        joint_nf: int = 16, 
        n_max_virtual_nodes: int = 0
    ):
        # initialize integrator that doesn't remove COM
        integrator = integrator or MolecularEulerMaruyama(
            terminal_only=not return_full_paths,
            n_dims=n_dims, 
            com_mode = "cond"
        )
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
            joint_nf=joint_nf,
            integrator=integrator
        )
        self.n_max_virtual_nodes = n_max_virtual_nodes
    
    def _create_initial_noisy_state(self) -> MolecularState:
        """Create initial noisy molecular state"""

        if self.n_max_virtual_nodes > 0:
            # add the virtual nodes that are expected to be removed in expectation to the size
            self.ligand_sizes = torch.tensor(self.ligand_sizes, dtype=torch.long, device=self.tspan.device) + self.n_max_virtual_nodes//2
        
        total_lig_atoms = sum(self.ligand_sizes)
        total_pocket_atoms = sum(self.pocket_sizes)
        
        # Create masks
        ligand_mask = torch.cat([
            torch.full((size,), i, dtype=torch.long, device=self.tspan.device)
            for i, size in enumerate(self.ligand_sizes)
        ])

        
        # Sample initial noise (following x1 = randn() * sigma(1) * scale(1))
        initial_sigma = self.scheme.sigma(self.tspan[0])
        initial_scale = self.scheme.scale(self.tspan[0])
        
        ligand_noise = (
            torch.randn(total_lig_atoms, self.n_dims + self.joint_nf, device=self.tspan.device) 
            * initial_sigma * initial_scale
        )

        
        return MolecularState(
            ligand=ligand_noise / self.scheme.coord_norm,
            pocket= torch.zeros(total_pocket_atoms, self.n_dims + self.joint_nf, device=self.tspan.device), # unused, just to initialize state object
            ligand_mask=ligand_mask,
            pocket_mask= torch.zeros(total_pocket_atoms, dtype=torch.long, device=self.tspan.device), # unused,  just to initialize state object
            batch_size=self.batch_size
        )
    
        
    def _create_initial_conditioning_dict(self, pocket_batch, guidance_scale):

        assert hasattr(self, "_model"), "Attach DenoiserModel to SdeSampler!"

        pocket_coords = pocket_batch["pocket_coords"].to(self._model.device)
        pocket_features = pocket_batch["pocket_features"].to(self._model.device)  # One-hot features
        pocket_mask = pocket_batch["pocket_mask"].to(self._model.device)


        # normalize pocket coordinates
        normalized_pocket_coordinates = pocket_coords / self.scheme.coord_norm
        # POCKET-COM
        pocket_com = scatter_mean(normalized_pocket_coordinates, pocket_mask, dim=0)
        normalized_pocket_coordinates_zero_com = normalized_pocket_coordinates - pocket_com[pocket_mask]
        # encode pocket features and normalize
        encoded_pocket_features = self._model.denoiser.residue_encoder(pocket_features)
        encoded_pocket_features = F.normalize(encoded_pocket_features, dim=-1) 


        # concat for initial encoded pocket state
        pocket_state = torch.cat([normalized_pocket_coordinates_zero_com, 
                                  encoded_pocket_features], dim=1)

        assert pocket_state.shape[1] == self.n_dims + self.joint_nf, f"Feature dim mismatch in pocket conditioning init: got {pocket_state.shape[1]} but expected {self.n_dims + self.joint_nf}"


        return {
            "pocket_state": pocket_state, 
            "pocket_mask": pocket_mask, 
            "guidance_scale":guidance_scale, 
            "pocket_com": pocket_com
            }

    def _create_molecular_dynamics(self, cond: TensorMapping | None) -> MolecularSdeDynamics:
        """
        Create molecular SDE dynamics.
        
        Implements the reverse SDE formula:
        dx = [2σ̇/σ + ṡ/s]x - [2sσ̇/σ]D(x/s, σ) dt + s√[2σ̇σ] dW
        """

        # calculate dynamics with inherited function: MolecularSdeDynamics
        dynamics = super()._create_molecular_dynamics(cond)

        def _molecular_drift(
            molecular_state: MolecularState,
            t: Tensor,
            params: MolecularParams,
        ) -> MolecularState:
            """Molecular drift"""

            # print("Pocket_state:", molecular_state.pocket)
            
            drift_molecular_state = dynamics.drift(molecular_state, t, params)

            return MolecularState(
                ligand=drift_molecular_state.ligand,
                pocket=torch.zeros_like(drift_molecular_state.pocket), # TODO check whether correct
                ligand_mask=drift_molecular_state.ligand_mask,
                pocket_mask=drift_molecular_state.pocket_mask,
                batch_size=drift_molecular_state.batch_size
            )
        
        def _molecular_diffusion(
            molecular_state: MolecularState,
            t: Tensor,
            params: MolecularParams
        ) -> MolecularState:
            """Molecular diffusion following GenCFD diffusion computation"""
            
            diff_molecular_state = dynamics.diffusion(molecular_state, t, params)

            return MolecularState(
                ligand=diff_molecular_state.ligand,
                pocket=torch.zeros_like(diff_molecular_state.pocket), # TODO check whether correct
                ligand_mask=diff_molecular_state.ligand_mask,
                pocket_mask=diff_molecular_state.pocket_mask,
                batch_size=diff_molecular_state.batch_size
            )
        
        return MolecularSdeDynamics(_molecular_drift, _molecular_diffusion)
    


# ********************
# Molecular Denoiser Wrapper
# ********************

def create_molecular_denoiser_wrapper(
        egnn_model, 
        scheme, 
        requires_grad: bool = False,
        conditional_sampling: bool = False
) -> MolecularDenoiseFn:
    """    
    Wraps the EGNN to handle MolecularState inputs/outputs with CDCD-style
    score interpolation for categorical features.
    
    Model predicts: coordinates (clean) + feature embeddings (for score interpolation)

    conditional_sampling enables classifier free weighted sampling: denoiser must have been trained that way!
    """
    
    def molecular_denoiser(
        molecular_state: MolecularState,
        sigma: Tensor,
        cond: TensorMapping | None = None,
    ) -> MolecularState:

        import contextlib
        context = contextlib.nullcontext() if requires_grad else torch.no_grad()
    
        with context:
            pred_output_lig, pred_output_pocket = egnn_model(
                molecular_state.ligand,
                molecular_state.pocket,
                sigma,
                molecular_state.ligand_mask,
                molecular_state.pocket_mask
            )
            
        
        # Split EGNN output: coordinates + logits
        n_dims = 3  # assuming 3D coordinates
        
        # Ligand
        pred_coords_lig = pred_output_lig[:, :n_dims]              # [N_lig, 3] - clean coords
        pred_logits_lig = pred_output_lig[:, n_dims:]              # [N_lig, atom_nf] - logits
        
        # Pocket  
        pred_coords_pocket = pred_output_pocket[:, :n_dims]        # [N_pocket, 3] - clean coords
        pred_logits_pocket = pred_output_pocket[:, n_dims:]        # [N_pocket, residue_nf] - logits
        
        # Convert logits to probabilities
        probs_lig = F.softmax(pred_logits_lig, dim=-1)             # [N_lig, atom_nf]
        probs_pocket = F.softmax(pred_logits_pocket, dim=-1)       # [N_pocket, residue_nf]
        
        # Get normalized embeddings from the model
        atom_embeddings = egnn_model.egnn_dynamics.atom_encoder(
            torch.eye(pred_logits_lig.size(-1), device=pred_logits_lig.device)
        )  # [atom_nf, joint_nf]
        atom_embeddings = F.normalize(atom_embeddings, dim=-1)
        
        residue_embeddings = egnn_model.egnn_dynamics.residue_encoder(
            torch.eye(pred_logits_pocket.size(-1), device=pred_logits_pocket.device)
        )  # [residue_nf, joint_nf] 
        residue_embeddings = F.normalize(residue_embeddings, dim=-1)
        
        # Score interpolation: Σ p_i * (e_i - x_noisy) / σ²
        # This gives us the "effective clean embedding prediction"
        # Expand sigma for per-atom computation
        if sigma.dim() == 0:
            sigma_expanded_lig = sigma.expand(molecular_state.ligand.size(0))
            sigma_expanded_pocket = sigma.expand(molecular_state.pocket.size(0))
        else:
            sigma_expanded_lig = sigma[molecular_state.ligand_mask]
            sigma_expanded_pocket = sigma[molecular_state.pocket_mask]
        
        # Weighted sum of embeddings (what CDCD calls x̂_0)
        interpolated_embeddings_lig = torch.matmul(probs_lig, atom_embeddings)      # [N_lig, joint_nf]
        interpolated_embeddings_pocket = torch.matmul(probs_pocket, residue_embeddings)  # [N_pocket, joint_nf]
        
        # For the SDE, we need to return the "effective clean prediction"
        # This combines clean coordinates with interpolated embeddings
        effective_clean_lig = torch.cat([pred_coords_lig, interpolated_embeddings_lig], dim=1)
        effective_clean_pocket = torch.cat([pred_coords_pocket, interpolated_embeddings_pocket], dim=1)
        
        return MolecularState(
            ligand=effective_clean_lig,
            pocket=effective_clean_pocket,
            ligand_mask=molecular_state.ligand_mask,
            pocket_mask=molecular_state.pocket_mask,
            batch_size=molecular_state.batch_size
        )
    
    def conditional_molecular_denoiser(
        molecular_state: MolecularState,
        sigma: Tensor,
        cond: TensorMapping | None = None
    ) -> MolecularState:

        assert hasattr(egnn_model, "null_residue_forward"), "Molecular Denoiser must have method \"null_resiude_forward\""

        import contextlib
        context = contextlib.nullcontext() if requires_grad else torch.no_grad()

        assert cond, "For Conditional sampling cond with keys \"pocket_state\", \"pocket_mask\" and \"guidance_scale\" must be given"

        pocket_state = cond["pocket_state"] 
        pocket_mask = cond["pocket_mask"]
        guidance_scale = cond["guidance_scale"]

        # Debugging:
        # print("Ligand Shape:", molecular_state.ligand.shape)
        # print("Ligand NF", molecular_state.ligand[0, :])
        # print("Pocket Shape:", pocket_state.shape)

        # Split EGNN output: coordinates + logits
        n_dims = egnn_model.egnn_dynamics.n_dims  # assuming 3D coordinates


        # classifier-free guidance sampling
        with context:
            # conditional denoising
            cond_pred_output_lig, cond_pred_output_pocket = egnn_model( # ConditionalPreconditionEGNNDynamics
                molecular_state.ligand,
                pocket_state,
                sigma,
                molecular_state.ligand_mask,
                pocket_mask
            )
            # assert whether denoiser leaves coordinates untouched
            assert torch.allclose(pocket_state[:, :n_dims], cond_pred_output_pocket[:, :n_dims], atol=1e6, rtol=1e6), \
                "Denoiser must not denoise fixed pocket coords in conditional sampling."
            
            # unconditional denoising
            uncond_pred_output_lig = egnn_model.null_residue_forward(
                molecular_state.ligand, 
                sigma, 
                molecular_state.ligand_mask
            )

            # prediction guiding for ligand
            pred_output_lig = cond_pred_output_lig + guidance_scale * (cond_pred_output_lig - uncond_pred_output_lig)
            # conditional "prediction" pocket: only embeddings needed
            pred_output_pocket = cond_pred_output_pocket.clone()

            # TODO try once without CFG and see if it even makes a difference:
            # pred_output_lig = cond_pred_output_lig.clone()

            
        

        
        # Ligand
        pred_coords_lig = pred_output_lig[:, :n_dims]              # [N_lig, 3] - clean coords
        pred_logits_lig = pred_output_lig[:, n_dims:]              # [N_lig, atom_nf] - logits
        
        # Pocket  
        # pred_coords_pocket = pred_output_pocket[:, :n_dims]        # [N_pocket, 3] - clean coords
        # pred_coords_pocket = molecular_state.pocket[:, :n_dims]    # keep constant
        # pred_logits_pocket = pred_output_pocket[:, n_dims:]        # [N_pocket, residue_nf] - logits
        
        # Convert logits to probabilities
        probs_lig = F.softmax(pred_logits_lig, dim=-1)             # [N_lig, atom_nf]
        # probs_pocket = F.softmax(pred_logits_pocket, dim=-1)       # [N_pocket, residue_nf]

        # NOTE temporary debug
        # none_idx = -1 
        # p_none_mean = probs_lig[:, none_idx].mean().item()
        # p_none_max  = probs_lig[:, none_idx].max().item()

        # wandb.log(
        #     {
        #         "sampling/p_none_mean": p_none_mean,
        #         "sampling/p_none_max":  p_none_max,
        #         "sampling/sample_step_sigma": float(sigma if sigma.dim() == 0 else sigma[0]),
        #     }
        # )

        
        # Get normalized embeddings from the model
        atom_embeddings = egnn_model.egnn_dynamics.atom_encoder(
            torch.eye(pred_logits_lig.size(-1), device=pred_logits_lig.device)
        )  # [atom_nf, joint_nf]
        atom_embeddings = F.normalize(atom_embeddings, dim=-1)
        
        # residue_embeddings = egnn_model.egnn_dynamics.residue_encoder(
        #     torch.eye(pred_logits_pocket.size(-1), device=pred_logits_pocket.device)
        # )  # [residue_nf, joint_nf] 
        # residue_embeddings = F.normalize(residue_embeddings, dim=-1)
        
        # Score interpolation: Σ p_i * (e_i - x_noisy) / σ²
        # This gives us the "effective clean embedding prediction"
        # Expand sigma for per-atom computation
        if sigma.dim() == 0:
            sigma_expanded_lig = sigma.expand(molecular_state.ligand.size(0))
            # sigma_expanded_pocket = sigma.expand(molecular_state.pocket.size(0))
        else:
            sigma_expanded_lig = sigma[molecular_state.ligand_mask]
            # sigma_expanded_pocket = sigma[molecular_state.pocket_mask]
        
        # Weighted sum of embeddings (what CDCD calls x̂_0)
        interpolated_embeddings_lig = torch.matmul(probs_lig, atom_embeddings)      # [N_lig, joint_nf]
        # interpolated_embeddings_pocket = torch.matmul(probs_pocket, residue_embeddings)  # [N_pocket, joint_nf]
        
        # For the SDE, we need to return the "effective clean prediction"
        # This combines clean coordinates with interpolated embeddings
        effective_clean_lig = torch.cat([pred_coords_lig, interpolated_embeddings_lig], dim=1)
        # effective_clean_pocket = torch.cat([pred_coords_pocket, interpolated_embeddings_pocket], dim=1)

        
        return MolecularState(
            ligand=effective_clean_lig,
            pocket=pocket_state,
            ligand_mask=molecular_state.ligand_mask,
            pocket_mask=pocket_mask,
            batch_size=molecular_state.batch_size
        )
    
    if not conditional_sampling: 
        return molecular_denoiser
    else:
        return conditional_molecular_denoiser


def edm_noise_decay(
    scheme: molecular_diffusion.MolecularDiffusion,
    rho: int = 7,
    num_steps: int = 256,
    end_sigma: float | None = 1e-3, # corresponds to sigma min in equation
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> Tensor:
    """Time steps corresponding to Eq. 5 in Karras et al."""

    rho_inv = torch.tensor(1.0 / rho)
    sigma_schedule = torch.arange(num_steps, dtype=dtype, device=device) / (
        num_steps - 1
    )
    sigma_schedule *= torch.pow(end_sigma, rho_inv) - torch.pow(
        scheme.sigma_max, rho_inv
    )
    sigma_schedule += torch.pow(scheme.sigma_max, rho_inv)
    sigma_schedule = torch.pow(sigma_schedule, rho)
    return scheme.sigma.inverse(sigma_schedule)
    

def create_molecular_sampler_from_model(
    model: MolecularDenoisingModel,
    ligand_sizes: list[int],
    pocket_sizes: list[int],
    num_steps: int = 50,
    schedule_type: str = "exponential_noise_schedule",
    return_full_paths: bool = False,
    config: Dict = {'clip_max': 100.0},
) -> MolecularSdeSampler | ConditionalMolecularSdeSampler:
    """Create a molecular sampler from a trained model"""
    
    # Create time span 
    # tspan = torch.linspace(1.0, 0.0, num_steps + 1, device=self.tspan.device)
    tspan = edm_noise_decay(scheme=model.scheme, num_steps=num_steps)
    
    # Create denoiser wrapper - now passing the scheme for proper normalization
    molecular_denoise_fn = create_molecular_denoiser_wrapper(model.denoiser, model.scheme, conditional_sampling=not model.update_pocket_coords)
    
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
        residue_nf=model.residue_nf,
        joint_nf=model.joint_nf,
    ) if model.update_pocket_coords else \
    ConditionalMolecularSdeSampler(
        ligand_sizes=ligand_sizes,
        pocket_sizes=pocket_sizes,
        scheme=model.scheme,
        denoise_fn=molecular_denoise_fn,
        tspan=tspan,
        apply_denoise_at_end=True,
        return_full_paths=return_full_paths,
        n_dims=model.n_dims,
        atom_nf=model.atom_nf,
        residue_nf=model.residue_nf,
        joint_nf=model.joint_nf, 
        n_max_virtual_nodes=model.n_max_virtual_nodes
    )
    sampler._model = model
    
    return sampler


def test_molecular_samplers():
    """Test the molecular SDE sampler with better debugging"""
    
    print("Testing Molecular SDE Sampler (CUDA)...")
    
    # Create dummy model for testing

    
    atom_nf = 4
    residue_nf = 5
    
    model = MolecularDenoisingModel(
        atom_nf=atom_nf,
        residue_nf=residue_nf,
        joint_nf=8,
        hidden_nf=16,
        n_layers=1
    )
    model.initialize()
    
    # Create sampler
    ligand_sizes = [3, 4]
    pocket_sizes = [8, 6]
    
    sampler = create_molecular_sampler_from_model(
        model=model,
        ligand_sizes=ligand_sizes,
        pocket_sizes=pocket_sizes,
        num_steps=10
    )
    
    # Generate samples with try-catch
    print("Generating samples...")
    try:
        samples = sampler.generate()
        
        print(f"Generated samples:")
        print(f"  Ligand coords: {samples['ligand_coords'].shape}")
        print(f"  Ligand features: {samples['ligand_features'].shape}")
        print(f"  Pocket coords: {samples['pocket_coords'].shape}")
        print(f"  Pocket features: {samples['pocket_features'].shape}")
        
        # Verify discrete features
        lig_sums = samples['ligand_features'].sum(dim=1)
        pocket_sums = samples['pocket_features'].sum(dim=1)
        
        print(f"  Ligand feature sums (should be 1.0): {lig_sums[:5]}...")
        print(f"  Pocket feature sums (should be 1.0): {pocket_sums[:5]}...")
        print("✅ Test passed!")
        
    except Exception as e:
        print(f"❌ Error during sampling: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to debug what went wrong
        print("\nDebugging information:")
        # print(f"Model norm_values: {model.scheme.norm_values}")
        # print(f"Model norm_biases: {model.scheme.norm_biases}")

def create_fake_pocket_batch(batch_size: int, pocket_sizes: list, residue_nf: int, device: str = 'cuda') -> Dict:
    """Create realistic fake molecular data for training on `device`"""
    
    total_pocket_atoms = sum(pocket_sizes)

    # Create masks (CUDA)
    pocket_mask = torch.cat([
        torch.full((size,), i, dtype=torch.long, device=device)
        for i, size in enumerate(pocket_sizes)
    ])
    
    # Generate coordinates (CUDA)
    pocket_coords = torch.randn(total_pocket_atoms, 3, device=device) * 5.0
    
    # Generate categorical features (one-hot encoded, CUDA)
    pocket_residue_types = torch.randint(0, residue_nf, (total_pocket_atoms,), device=device)
    pocket_features = F.one_hot(pocket_residue_types, residue_nf).float()
    
    
    return {
        'pocket_coords': pocket_coords,
        'pocket_features': pocket_features,
        'pocket_mask': pocket_mask,
        'batch_size': batch_size,
        'pocket_sizes': pocket_sizes
    }

def test_conditional_molecular_samplers():
    """Test the molecular SDE sampler with better debugging"""
    
    # print("Testing Molecular SDE Sampler (CUDA)...")
    
    
    atom_nf = 4
    residue_nf = 5
    
    model = ConditionalMolecularDenoisingModel(
        atom_nf=atom_nf,
        residue_nf=residue_nf,
        joint_nf=8,
        hidden_nf=16,
        n_layers=1, 
        update_pocket_coords=False, 
        n_max_virtual_nodes=12

    )

    model.initialize()
    
    # Create sampler from dummy data
    ligand_sizes = [3, 4]
    pocket_sizes = [8, 6]
    
    sampler = create_molecular_sampler_from_model(
        model=model,
        ligand_sizes=ligand_sizes,
        pocket_sizes=pocket_sizes,
        num_steps=10
    )

    assert sampler.__class__.__name__ == "ConditionalMolecularSdeSampler", f"sampler's name is {sampler.__class__.__name__}"

    batch = create_fake_pocket_batch(len(pocket_sizes), pocket_sizes, residue_nf)

    cond = sampler._create_initial_conditioning_dict(pocket_batch = batch, guidance_scale = 1)
    
    # Generate samples with try-catch
    print("Generating samples...")
    # try:
    samples = sampler.generate(cond=cond)
    
    print(f"Generated samples:")
    print(f"  Ligand coords: {samples['ligand_coords'].shape}")
    print(f"  Ligand features: {samples['ligand_features'].shape}")
    print(f"  Pocket coords: {samples['pocket_coords'].shape}")
    print(f"  Pocket features: {samples['pocket_features'].shape}")

    # see if pocket remained unchanged:
    assert torch.allclose(samples['pocket_coords'], batch["pocket_coords"]), f"Pocket coords changed: \nSamples: {samples['pocket_coords']} \n Input {samples['pocket_coords']} "
    assert torch.allclose(samples['pocket_features'], batch["pocket_features"]), f"Pocket features changed: \nSamples: {samples['pocket_features']} \n Input {samples['pocket_features']} "

    # print(samples['pocket_mask'], batch["pocket_mask"])
    # print(samples['pocket_coords'][samples['pocket_mask']])


    
    # Verify discrete features
    lig_sums = samples['ligand_features'].sum(dim=1)
    pocket_sums = samples['pocket_features'].sum(dim=1)
    
    print(f"  Ligand feature sums (should be 1.0): {lig_sums[:5]}...")
    print(f"  Pocket feature sums (should be 1.0): {pocket_sums[:5]}...")
    print("✅ Test passed!")
        
    # except Exception as e:
    #     print(f"❌ Error during sampling: {e}")
    #     import traceback
    #     traceback.print_exc()
        
        # Try to debug what went wrong
        # print("\nDebugging information:")
        # print(f"Model norm_values: {model.scheme.norm_values}")
        # print(f"Model norm_biases: {model.scheme.norm_biases}")


if __name__ == '__main__':
    # test_molecular_samplers()
    test_conditional_molecular_samplers()
