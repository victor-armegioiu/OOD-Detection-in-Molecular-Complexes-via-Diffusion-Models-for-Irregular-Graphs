"""
Enhanced Molecular OOD Detection with Trajectory Statistics

Extends the molecular likelihood evaluator to collect rich trajectory statistics
during forward integration for improved OOD detection beyond just likelihood scores.
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
import torch.utils.checkpoint as checkpoint

from dataclasses import dataclass

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


@dataclass
class TrajectoryStatistics:
    """
    Collects and computes trajectory statistics during molecular diffusion for OOD detection.
    """
    
    def __init__(self, n_dims: int = 3):
        self.n_dims = n_dims
        self.reset()
    
    def reset(self):
        """Reset all collected statistics"""
        # Vector field magnitudes
        self.vector_field_l2_norms = []
        self.vector_field_linf_norms = []
        self.coord_field_l2_norms = []
        self.feature_field_l2_norms = []
        self.ligand_field_vars = []
        self.pocket_field_vars = []
        
        # Trajectory curvature & smoothness
        self.direction_changes = []
        self.acceleration_magnitudes = []
        self.path_segments = []
        
        # Molecular geometry
        self.com_drifts = []
        self.intermol_distance_changes = []
        self.coord_feature_correlations = []
        self.dynamic_coupling_correlations = []
        
        # Flow properties
        self.lipschitz_estimates = []
        self.flow_energy = []
        
        # State history for computing derivatives
        self.prev_state = None
        self.prev_vector_field = None
        self.initial_state = None
        self.initial_intermol_distance = None
    
    def update(self, molecular_state: MolecularState, vector_field: MolecularState, dt: Tensor):
        """Update statistics with current state and vector field"""
        
        # Store initial state
        if self.initial_state is None:
            self.initial_state = molecular_state
            self._compute_initial_geometry(molecular_state)
        
        # Vector field magnitude statistics
        self._update_vector_field_stats(vector_field)
        
        # Trajectory curvature (requires previous states)
        if self.prev_state is not None and self.prev_vector_field is not None:
            self._update_curvature_stats(molecular_state, vector_field, dt)
        
        # Molecular geometry statistics
        self._update_geometry_stats(molecular_state)
        self._update_dynamic_coupling_stats(vector_field)
        
        # Flow properties
        if self.prev_state is not None:
            self._update_flow_stats(molecular_state, vector_field, dt)
        
        # Update history
        self.prev_state = self._detach_state(molecular_state)
        self.prev_vector_field = self._detach_state(vector_field)
    
    def _detach_state(self, state: MolecularState) -> MolecularState:
        """Detach state for storage without gradients"""
        return MolecularState(
            ligand=state.ligand.detach().clone(),
            pocket=state.pocket.detach().clone(),
            ligand_mask=state.ligand_mask,
            pocket_mask=state.pocket_mask,
            batch_size=state.batch_size
        )
    
    def _update_vector_field_stats(self, vector_field: MolecularState):
        """Update vector field magnitude statistics"""
        # Full vector field norms
        full_field = torch.cat([vector_field.ligand.reshape(-1), vector_field.pocket.reshape(-1)])
        self.vector_field_l2_norms.append(torch.norm(full_field, p=2).item())
        self.vector_field_linf_norms.append(torch.norm(full_field, p=float('inf')).item())
        
        # Coordinate vs feature field norms
        coord_field = torch.cat([
            vector_field.ligand[:, :self.n_dims].reshape(-1),
            vector_field.pocket[:, :self.n_dims].reshape(-1)
        ])
        feature_field = torch.cat([
            vector_field.ligand[:, self.n_dims:].reshape(-1),
            vector_field.pocket[:, self.n_dims:].reshape(-1)
        ])
        
        self.coord_field_l2_norms.append(torch.norm(coord_field, p=2).item())
        self.feature_field_l2_norms.append(torch.norm(feature_field, p=2).item())
        
        # Component-wise variance
        self.ligand_field_vars.append(torch.var(vector_field.ligand).item())
        self.pocket_field_vars.append(torch.var(vector_field.pocket).item())
    
    def _update_curvature_stats(self, molecular_state: MolecularState, vector_field: MolecularState, dt: Tensor):
        """Update trajectory curvature and smoothness statistics"""
        # Direction changes in vector field
        prev_field_flat = torch.cat([
            self.prev_vector_field.ligand.reshape(-1),
            self.prev_vector_field.pocket.reshape(-1)
        ])
        curr_field_flat = torch.cat([
            vector_field.ligand.reshape(-1),
            vector_field.pocket.reshape(-1)
        ])
        
        # Cosine similarity between consecutive vector fields
        if torch.norm(prev_field_flat) > 1e-8 and torch.norm(curr_field_flat) > 1e-8:
            cos_sim = F.cosine_similarity(prev_field_flat.unsqueeze(0), curr_field_flat.unsqueeze(0))
            self.direction_changes.append(1.0 - cos_sim.item())  # 0 = no change, 2 = opposite direction
        
        # Acceleration magnitude (discrete second derivative)
        acceleration = torch.cat([
            vector_field.ligand.reshape(-1),
            vector_field.pocket.reshape(-1)
        ]) - torch.cat([
            self.prev_vector_field.ligand.reshape(-1),
            self.prev_vector_field.pocket.reshape(-1)
        ])
        self.acceleration_magnitudes.append(torch.norm(acceleration).item())
        
        # Path segment length for tortuosity calculation
        prev_coords = torch.cat([
            self.prev_state.ligand[:, :self.n_dims],
            self.prev_state.pocket[:, :self.n_dims]
        ], dim=0)
        curr_coords = torch.cat([
            molecular_state.ligand[:, :self.n_dims],
            molecular_state.pocket[:, :self.n_dims]
        ], dim=0)
        segment_length = torch.norm(curr_coords - prev_coords).item()
        self.path_segments.append(segment_length)
    
    def _compute_initial_geometry(self, molecular_state: MolecularState):
        """Compute initial molecular geometry references"""
        # Initial inter-molecular distance (COM to COM)
        lig_coords = molecular_state.ligand[:, :self.n_dims]
        pocket_coords = molecular_state.pocket[:, :self.n_dims]
        
        lig_com = torch.mean(lig_coords, dim=0)
        pocket_com = torch.mean(pocket_coords, dim=0)
        self.initial_intermol_distance = torch.norm(lig_com - pocket_com).item()
    
    def _update_geometry_stats(self, molecular_state: MolecularState):
        """Update molecular geometry statistics"""
        # COM drift (should be near zero due to constraints)
        all_coords = torch.cat([
            molecular_state.ligand[:, :self.n_dims],
            molecular_state.pocket[:, :self.n_dims]
        ], dim=0)
        combined_mask = torch.cat([molecular_state.ligand_mask, molecular_state.pocket_mask])
        com = scatter_mean(all_coords, combined_mask, dim=0)
        com_drift = torch.norm(com).item()
        self.com_drifts.append(com_drift)
        
        # Inter-molecular distance change
        lig_coords = molecular_state.ligand[:, :self.n_dims]
        pocket_coords = molecular_state.pocket[:, :self.n_dims]
        lig_com = torch.mean(lig_coords, dim=0)
        pocket_com = torch.mean(pocket_coords, dim=0)
        current_distance = torch.norm(lig_com - pocket_com).item()
        distance_change = abs(current_distance - self.initial_intermol_distance)
        self.intermol_distance_changes.append(distance_change)

    def _update_dynamic_coupling_stats(self, vector_field: MolecularState):
        """Compute dynamic coord-feature coupling from instantaneous drift magnitudes"""
        
        # Per-atom drift magnitudes for coordinates and features
        coord_drift_lig = torch.norm(vector_field.ligand[:, :self.n_dims], dim=1)
        feature_drift_lig = torch.norm(vector_field.ligand[:, self.n_dims:], dim=1)
        
        coord_drift_pocket = torch.norm(vector_field.pocket[:, :self.n_dims], dim=1)
        feature_drift_pocket = torch.norm(vector_field.pocket[:, self.n_dims:], dim=1)
        
        # Combine across all atoms
        all_coord_drifts = torch.cat([coord_drift_lig, coord_drift_pocket])
        all_feature_drifts = torch.cat([feature_drift_lig, feature_drift_pocket])
        
        # Correlation between drift magnitudes across atoms at this timestep
        if len(all_coord_drifts) > 1:
            correlation = torch.corrcoef(torch.stack([all_coord_drifts, all_feature_drifts]))[0, 1]
            if not torch.isnan(correlation):
                self.dynamic_coupling_correlations.append(correlation.item())
    
    def _update_flow_stats(self, molecular_state: MolecularState, vector_field: MolecularState, dt: Tensor):
        """Update flow property statistics"""
        # Lipschitz estimate: ||f(x_t) - f(x_{t-1})|| / ||x_t - x_{t-1}||
        curr_coords = torch.cat([
            molecular_state.ligand[:, :self.n_dims],
            molecular_state.pocket[:, :self.n_dims]
        ], dim=0)
        prev_coords = torch.cat([
            self.prev_state.ligand[:, :self.n_dims],
            self.prev_state.pocket[:, :self.n_dims]
        ], dim=0)
        
        dx_norm = torch.norm(curr_coords - prev_coords)
        if dx_norm > 1e-8:
            curr_field_flat = torch.cat([
                vector_field.ligand.reshape(-1),
                vector_field.pocket.reshape(-1)
            ])
            prev_field_flat = torch.cat([
                self.prev_vector_field.ligand.reshape(-1),
                self.prev_vector_field.pocket.reshape(-1)
            ])
            df_norm = torch.norm(curr_field_flat - prev_field_flat)
            lipschitz_est = (df_norm / dx_norm).item()
            self.lipschitz_estimates.append(lipschitz_est)
        
        # Flow energy: cumulative f^T * dx
        dx = curr_coords - prev_coords
        f_coords = torch.cat([
            vector_field.ligand[:, :self.n_dims],
            vector_field.pocket[:, :self.n_dims]
        ], dim=0)
        energy_increment = torch.dot(f_coords.reshape(-1), dx.reshape(-1)).item()
        self.flow_energy.append(energy_increment)
    
    def get_summary(self) -> Dict[str, float]:
        """Compute summary statistics from collected trajectory data"""
        summary = {}
        
        # Vector field magnitude summaries
        if self.vector_field_l2_norms:
            summary.update({
                'vf_l2_mean': np.mean(self.vector_field_l2_norms),
                'vf_l2_std': np.std(self.vector_field_l2_norms),
                'vf_l2_max': np.max(self.vector_field_l2_norms),
                'vf_spikiness': np.max(self.vector_field_l2_norms) / (np.mean(self.vector_field_l2_norms) + 1e-8),
                'coord_feature_ratio': np.mean(self.coord_field_l2_norms) / (np.mean(self.feature_field_l2_norms) + 1e-8)
            })
        
        # Trajectory curvature summaries
        if self.direction_changes:
            summary.update({
                'total_angular_deviation': np.sum(self.direction_changes),
                'smoothness_score': 1.0 / (np.var(self.acceleration_magnitudes) + 1e-8),
                'mean_acceleration': np.mean(self.acceleration_magnitudes)
            })
        
        if self.path_segments:
            total_path_length = np.sum(self.path_segments)
            if self.initial_state is not None:
                # Straight line distance from start to end
                initial_coords = torch.cat([
                    self.initial_state.ligand[:, :self.n_dims],
                    self.initial_state.pocket[:, :self.n_dims]
                ], dim=0)
                final_coords = torch.cat([
                    self.prev_state.ligand[:, :self.n_dims],
                    self.prev_state.pocket[:, :self.n_dims]
                ], dim=0)
                straight_line_dist = torch.norm(final_coords - initial_coords).item()
                summary['path_efficiency'] = straight_line_dist / (total_path_length + 1e-8)
                summary['path_tortuosity'] = total_path_length / (straight_line_dist + 1e-8)
        
        # Molecular geometry summaries
        if self.com_drifts:
            summary.update({
                'max_com_drift': np.max(self.com_drifts),
                'mean_com_drift': np.mean(self.com_drifts),
                'max_intermol_change': np.max(self.intermol_distance_changes) if self.intermol_distance_changes else 0.0
            })
        
        # Flow property summaries
        if self.lipschitz_estimates:
            summary.update({
                'mean_lipschitz': np.mean(self.lipschitz_estimates),
                'max_lipschitz': np.max(self.lipschitz_estimates),
                'total_flow_energy': np.sum(self.flow_energy) if self.flow_energy else 0.0
            })
        
        # Component consistency
        if self.dynamic_coupling_correlations:
            summary['dynamic_coord_feature_coupling'] = np.mean(self.dynamic_coupling_correlations)
            summary['coupling_consistency'] = 1.0 / (np.var(self.dynamic_coupling_correlations) + 1e-8)
        
        return summary


class MolecularLikelihoodEvaluator:
    """
    Enhanced molecular likelihood evaluator that collects trajectory statistics
    for improved OOD detection beyond just likelihood scores.
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
        collect_trajectory_stats: bool = True,
    ):
        self.scheme = scheme
        self.denoise_fn = denoise_fn
        self.tspan = tspan.cuda()
        self.num_hutchinson_samples = num_hutchinson_samples
        self.n_dims = n_dims
        self.atom_nf = atom_nf
        self.residue_nf = residue_nf
        self.joint_nf = joint_nf
        self.collect_trajectory_stats = collect_trajectory_stats
        
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
        
            # Project COM to maintain consistency
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
    
    def _forward_integrate_with_stats_single(
        self, 
        clean_state: MolecularState, 
        cond: TensorMapping | None = None,
        checkpoint_segments: int = 20
    ) -> Tuple[MolecularState, Tensor, Dict[str, float]]:
        """Forward integrate with trajectory statistics collection for single sample using gradient checkpointing"""
        
        assert clean_state.batch_size == 1, "This method expects single sample"
        
        # Create dynamics
        dynamics = self._create_molecular_dynamics(cond)
        params = dict(drift=dict(cond=cond), diffusion=dict())
        
        # Initialize trajectory statistics
        trajectory_stats = TrajectoryStatistics(n_dims=self.n_dims) if self.collect_trajectory_stats else None
        
        # Split integration steps into segments for checkpointing
        total_steps = len(self.tspan) - 1
        segment_size = max(1, total_steps // checkpoint_segments)
        
        current_state = clean_state
        total_divergence = torch.tensor(0.0, device=clean_state.ligand.device)
        
        for segment_start in range(0, total_steps, segment_size):
            segment_end = min(segment_start + segment_size, total_steps)
            
            # Define segment integration function for checkpointing
            def integrate_segment(state_in):
                return self._integrate_segment_with_divergence(
                    state_in, segment_start, segment_end, dynamics, params, trajectory_stats
                )
            
            # Use gradient checkpointing for this segment
            current_state, segment_divergence = checkpoint.checkpoint(
                integrate_segment, 
                current_state,
                use_reentrant=False
            )
            
            total_divergence = total_divergence + segment_divergence
        
        # Get final trajectory statistics summary
        stats_summary = trajectory_stats.get_summary() if trajectory_stats is not None else {}
        
        return current_state, total_divergence, stats_summary

    def _integrate_segment_with_divergence(
        self,
        initial_state: MolecularState,
        start_idx: int,
        end_idx: int,
        dynamics: MolecularSdeDynamics,
        params: MolecularParams,
        trajectory_stats: TrajectoryStatistics
    ) -> Tuple[MolecularState, Tensor]:
        """Integrate a segment of the trajectory with divergence computation"""
        
        current_state = initial_state
        segment_divergence = torch.tensor(0.0, device=initial_state.ligand.device)
        
        for i in range(start_idx, end_idx):
            t_curr = self.tspan[i]
            t_next = self.tspan[i + 1]
            dt = t_next - t_curr
            
            # Compute vector field for this step
            vector_field = dynamics.drift(current_state, t_curr, params["drift"])
            
            # Collect trajectory statistics (no gradients needed for this)
            if trajectory_stats is not None:
                with torch.no_grad():
                    trajectory_stats.update(current_state, vector_field, dt)
            
            # Estimate divergence at current point (this needs gradients)
            div_estimate = self._estimate_divergence_single(current_state, t_curr, params, dynamics)
            segment_divergence = segment_divergence + div_estimate * dt
            
            # Integration step
            current_state = self._heun_step(current_state, t_curr, t_next, dynamics, params)
        
        return current_state, segment_divergence
    
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
        
        # Apply COM constraint
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
        # Remove 3 DOF for COM constraint in 3D
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
    
    def evaluate_likelihood_with_stats(
        self, 
        molecular_state: MolecularState, 
        cond: TensorMapping | None = None
    ) -> Tuple[Tensor, List[Dict[str, float]]]:
        """
        Evaluate log-likelihood with trajectory statistics for each sample in batch.
        
        Args:
            molecular_state: Clean molecular state to evaluate (batch)
            cond: Optional conditioning information
            
        Returns:
            log_likelihoods: Tensor of shape (batch_size,) with log probability for each sample
            trajectory_statistics: List of trajectory statistic dictionaries for each sample
        """
        
        batch_size = molecular_state.batch_size
        log_likelihoods = torch.zeros(batch_size, device=molecular_state.ligand.device)
        trajectory_statistics = []
        
        # Process each sample in the batch separately
        for sample_idx in range(batch_size):
            # Extract single sample
            single_sample = self._extract_single_sample(molecular_state, sample_idx)
            
            # Extract conditioning for this sample if provided
            sample_cond = None
            if cond is not None:
                sample_cond = {}
                for key, value in cond.items():
                    if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                        sample_cond[key] = value[sample_idx:sample_idx+1]
                    else:
                        sample_cond[key] = value
            
            # Forward integrate from clean to noise using probability flow ODE
            terminal_state, divergence_integral, stats_summary = self._forward_integrate_with_stats_single(
                single_sample, sample_cond
            )
            
            # Evaluate terminal Gaussian likelihood
            terminal_log_prob = self._evaluate_terminal_likelihood_single(terminal_state)
            
            # Apply change of variables formula
            log_likelihood = terminal_log_prob + divergence_integral
            log_likelihoods[sample_idx] = log_likelihood
            trajectory_statistics.append(stats_summary)
        
        return log_likelihoods, trajectory_statistics
    
    def evaluate_likelihood(
        self, 
        molecular_state: MolecularState, 
        cond: TensorMapping | None = None
    ) -> Tensor:
        """
        Evaluate log-likelihood of clean molecular state for each sample in batch.
        (Backward compatibility method - returns only likelihoods)
        """
        log_likelihoods, _ = self.evaluate_likelihood_with_stats(molecular_state, cond)
        return log_likelihoods


def create_molecular_likelihood_evaluator_from_model(
    model,
    num_steps: int = 50,
    num_hutchinson_samples: int = 1,
    collect_trajectory_stats: bool = True,
    config: Dict = {'clip_max': 100.0},
) -> MolecularLikelihoodEvaluator:
    """Create an enhanced molecular likelihood evaluator from a trained model"""
    
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
        collect_trajectory_stats=collect_trajectory_stats,
    )
    evaluator._model = model
    
    return evaluator


def test_trajectory_ood_detection():
    """Test the enhanced molecular OOD detection with trajectory statistics"""
    
    print("Testing Enhanced Molecular OOD Detection with Trajectory Statistics (CUDA)...")
    
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
        
        # Create enhanced likelihood evaluator
        print("Creating enhanced likelihood evaluator with trajectory statistics...")
        evaluator = create_molecular_likelihood_evaluator_from_model(
            model=model,
            num_steps=10,  # Small for testing
            num_hutchinson_samples=3,
            collect_trajectory_stats=True,
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
        
        print("Evaluating likelihood with trajectory statistics...")
        log_likelihoods, trajectory_stats = evaluator.evaluate_likelihood_with_stats(clean_state)
        
        print(f"✅ Enhanced OOD detection successful!")
        print(f"  Batch size: {batch_size}")
        print(f"  Likelihood shape: {log_likelihoods.shape}")
        print(f"  Log-likelihoods: {log_likelihoods}")
        print(f"  Number of trajectory stat dicts: {len(trajectory_stats)}")
        
        # Display trajectory statistics for first sample
        if trajectory_stats:
            print(f"\n📊 Trajectory statistics for sample 0:")
            for key, value in trajectory_stats[0].items():
                print(f"    {key}: {value:.4f}")
        
        # Test different scenarios to show trajectory differences
        print("\n🔍 Testing OOD detection scenarios...")
        
        # Create a "corrupted" sample with larger noise
        corrupted_state = MolecularState(
            ligand=ligand_data + torch.randn_like(ligand_data) * 0.5,  # More noise
            pocket=pocket_data + torch.randn_like(pocket_data) * 0.5,
            ligand_mask=ligand_mask,
            pocket_mask=pocket_mask,
            batch_size=batch_size
        )
        
        log_liks_corrupted, stats_corrupted = evaluator.evaluate_likelihood_with_stats(corrupted_state)
        
        print(f"  Clean samples likelihood: {log_likelihoods.mean().item():.3f}")
        print(f"  Corrupted samples likelihood: {log_liks_corrupted.mean().item():.3f}")
        print(f"  Likelihood difference: {(log_likelihoods.mean() - log_liks_corrupted.mean()).item():.3f}")
        
        # Compare trajectory statistics
        if trajectory_stats and stats_corrupted:
            print(f"\n📈 Trajectory statistics comparison (clean vs corrupted):")
            for key in trajectory_stats[0].keys():
                clean_val = trajectory_stats[0][key]
                corrupt_val = stats_corrupted[0][key]
                print(f"    {key}: {clean_val:.4f} vs {corrupt_val:.4f}")
        
        # Verify we get one likelihood + stats per sample
        assert log_likelihoods.shape == (batch_size,), f"Expected shape ({batch_size},), got {log_likelihoods.shape}"
        assert len(trajectory_stats) == batch_size, f"Expected {batch_size} stat dicts, got {len(trajectory_stats)}"
        
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during enhanced OOD detection: {e}")
        import traceback
        traceback.print_exc()



def save_dict_to_json(data, output_path):
    ''' Save dictionary to JSON file. Initialize new file if it doesn't exist. Append to existing file if it does.'''
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing_data = json.load(f)
        existing_data.update(data)
        with open(output_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
    else:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
    print(f"Results saved to: {output_path}")


def process_dataset(dataset, evaluator, device, dataset_name, output_path):
    """Process a dataset and return likelihood statistics"""
    print(f"\nProcessing {dataset_name}...", flush=True)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, follow_batch=['lig_coords', 'prot_coords'])
    
    # If JSON output_path exists, load it, otherwise initialize empty dictionary
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}
        
    batch_count = 0
    for batch in dataloader:
        batch = batch.to(device)
        print(f"\nStarting batch...{batch.id}")
        print(batch)
        
        lig_features = batch.lig_features
        pocket_features = batch.prot_features

        # Convert one-hot features to embeddings
        ligand_embeddings = evaluator._model.denoiser.atom_encoder(lig_features)  # [N_lig, joint_nf]
        pocket_embeddings = evaluator._model.denoiser.residue_encoder(pocket_features)  # [N_pocket, joint_nf]

        lig_coords = batch.lig_coords
        pocket_coords = batch.prot_coords
        ligand_mask = batch.lig_coords_batch
        pocket_mask = batch.prot_coords_batch

        # Concatenate coords and embeddings
        ligand_data = torch.cat([lig_coords, ligand_embeddings], dim=1)
        pocket_data = torch.cat([pocket_coords, pocket_embeddings], dim=1)

        # Center-of-mass correction for this molecule
        combined_coords = torch.cat([lig_coords, pocket_coords], dim=0)
        combined_mask = torch.cat([ligand_mask, pocket_mask])
        combined_coords_centered = remove_mean_batch(combined_coords, combined_mask)
        ligand_data[:, :3] = combined_coords_centered[:lig_coords.shape[0]]
        pocket_data[:, :3] = combined_coords_centered[lig_coords.shape[0]:]

        # Separate the batch into individual graphs
        print(torch.bincount(ligand_mask).tolist())
        print(torch.bincount(pocket_mask).tolist())

        ligand_data = torch.split(ligand_data, torch.bincount(ligand_mask).tolist())
        pocket_data = torch.split(pocket_data, torch.bincount(pocket_mask).tolist())

        for i in range(len(ligand_data)):

            # If the molecule data already exists (with a log likelihood that is not NaN nor Infinity), skip it
            exists = batch.id[i] in all_metrics
            has_likelihood = exists and 'log_likelihood' in all_metrics[batch.id[i]]
            is_nan = has_likelihood and np.isnan(all_metrics[batch.id[i]]['log_likelihood'])
            is_inf = has_likelihood and np.isinf(all_metrics[batch.id[i]]['log_likelihood'])
            if exists and has_likelihood and not is_nan and not is_inf:
                print(f"Molecule {batch.id[i]} already processed, skipping...", flush=True)
                continue

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
            try:
                log_likelihood, trajectory_stats = evaluator.evaluate_likelihood_with_stats(state)
            except Exception as e:
                print(f"Error evaluating likelihood for molecule {batch.id[i]}: {e}")
                continue

            all_metrics[batch.id[i]] = {'log_likelihood': float(log_likelihood.item())}
            if trajectory_stats:
                trajectory_stats = {k: float(v) for k, v in trajectory_stats[0].items()}
                all_metrics[batch.id[i]].update(trajectory_stats)
        
        # Intermediate save if batch_count is a multiple of 10
        if batch_count == 0 or batch_count % 10 == 0:
            save_dict_to_json(all_metrics, output_path)
        batch_count += 1

    save_dict_to_json(all_metrics, output_path)
    return all_metrics


def main(dataset_path, checkpoint_path, results_folder, num_steps, num_hutchinson_samples, start_idx=None, stop_idx=None):
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    print(f"Dataset path: {dataset_path}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Results folder: {results_folder}")
    print(f"Number of integration steps: {num_steps}")
    print(f"Number of Hutchinson samples: {num_hutchinson_samples}")
    if start_idx is not None and stop_idx is not None:
        print(f"Processing dataset slice: indices {start_idx} to {stop_idx-1}")
    print()
    
    # Check checkpoint existence
    for path in [dataset_path, checkpoint_path]:
        name = os.path.basename(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found at {path}")
        print(f"Found {name}: {path}")
    
    # Load datasets and model
    print("\nLoading dataset and model...", flush=True)
    full_dataset = torch.load(dataset_path)
    name = os.path.basename(dataset_path)
    
    # Apply slicing if specified
    if start_idx is not None and stop_idx is not None:
        print(f"Slicing dataset: using indices {start_idx} to {stop_idx-1}")
        
        if hasattr(full_dataset, 'input_data') and hasattr(full_dataset, '__len__'):
            # This is a PDBbind_Dataset with input_data dictionary
            total_size = len(full_dataset)
            print(f"Total dataset length: {total_size}")
            
            # Clamp indices to valid range
            start_idx = max(0, start_idx)
            stop_idx = min(total_size, stop_idx)
            
            if start_idx >= stop_idx:
                raise ValueError(f"Invalid slice range: start_idx ({start_idx}) >= stop_idx ({stop_idx})")
            
            # Create sliced input_data dictionary
            sliced_input_data = {}
            slice_idx = 0
            for original_idx in range(start_idx, stop_idx):
                if original_idx in full_dataset.input_data:
                    sliced_input_data[slice_idx] = full_dataset.input_data[original_idx]
                    slice_idx += 1
            
            # Create a new dataset instance with sliced input_data
            dataset = PDBbind_Dataset.create_sliced_dataset(sliced_input_data)
            
            print(f"Sliced dataset length: {len(dataset)}")
            del full_dataset
        
        else:
            raise ValueError("Dataset does not support length operation for slicing")
    else:
        dataset = full_dataset
        print(f"Using full dataset, size: {len(dataset) if hasattr(dataset, '__len__') else 'unknown'}")


    model = load_checkpoint(checkpoint_path)
    
    # Create likelihood evaluator
    print("\nCreating likelihood evaluator...", flush=True)
    evaluator = create_molecular_likelihood_evaluator_from_model(
        model,
        num_steps=num_steps,
        num_hutchinson_samples=num_hutchinson_samples,
    )
    
    output_filename = f'{name.replace(".pt", "")}_metrics.json'
    output_path = os.path.join(results_folder, output_filename)

    # -----------------------------------------------------------
    metrics = process_dataset(dataset, evaluator, device, name, output_path)
    # -----------------------------------------------------------
    
    print(f"Done! Processed {len(metrics)} graphs.")

    

if __name__ == '__main__':
    # test_trajectory_ood_detection()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='dataset_cleansplit_train.pt', type=str)
    parser.add_argument('--checkpoint_path', default='training_runs/0725_115513_dataset_cleansplit_train/checkpoint_epoch_579.pt', type=str)
    parser.add_argument('--results_folder', default='likelihood_results', type=str)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--num_hutchinson_samples', type=int, default=20)
    parser.add_argument('--start_idx', type=int, default=None, help='Start index for dataset slicing (inclusive)')
    parser.add_argument('--stop_idx', type=int, default=None, help='Stop index for dataset slicing (exclusive)')
    args = parser.parse_args()

    main(args.dataset_path, args.checkpoint_path, args.results_folder, args.num_steps, args.num_hutchinson_samples, args.start_idx, args.stop_idx)
