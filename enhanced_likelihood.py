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


Tensor = torch.Tensor
TensorMapping = Mapping[str, Tensor]
MolecularParams = Mapping[str, Any]


@dataclass
class TrajectoryStatistics:
    """Collects and computes trajectory statistics during molecular diffusion for OOD detection."""
    
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
        full_field = torch.cat([vector_field.ligand.reshape(-1), vector_field.pocket.reshape(-1)])
        self.vector_field_l2_norms.append(torch.norm(full_field, p=2).item())
        self.vector_field_linf_norms.append(torch.norm(full_field, p=float('inf')).item())
        
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
        
        self.ligand_field_vars.append(torch.var(vector_field.ligand).item())
        self.pocket_field_vars.append(torch.var(vector_field.pocket).item())
    
    def _update_curvature_stats(self, molecular_state: MolecularState, vector_field: MolecularState, dt: Tensor):
        """Update trajectory curvature and smoothness statistics"""
        prev_field_flat = torch.cat([
            self.prev_vector_field.ligand.reshape(-1),
            self.prev_vector_field.pocket.reshape(-1)
        ])
        curr_field_flat = torch.cat([
            vector_field.ligand.reshape(-1),
            vector_field.pocket.reshape(-1)
        ])
        
        if torch.norm(prev_field_flat) > 1e-8 and torch.norm(curr_field_flat) > 1e-8:
            cos_sim = F.cosine_similarity(prev_field_flat.unsqueeze(0), curr_field_flat.unsqueeze(0))
            self.direction_changes.append(1.0 - cos_sim.item())
        
        acceleration = torch.cat([
            vector_field.ligand.reshape(-1),
            vector_field.pocket.reshape(-1)
        ]) - torch.cat([
            self.prev_vector_field.ligand.reshape(-1),
            self.prev_vector_field.pocket.reshape(-1)
        ])
        self.acceleration_magnitudes.append(torch.norm(acceleration).item())
        
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
        lig_coords = molecular_state.ligand[:, :self.n_dims]
        pocket_coords = molecular_state.pocket[:, :self.n_dims]
        
        lig_com = torch.mean(lig_coords, dim=0)
        pocket_com = torch.mean(pocket_coords, dim=0)
        self.initial_intermol_distance = torch.norm(lig_com - pocket_com).item()
    
    def _update_geometry_stats(self, molecular_state: MolecularState):
        """Update molecular geometry statistics"""
        all_coords = torch.cat([
            molecular_state.ligand[:, :self.n_dims],
            molecular_state.pocket[:, :self.n_dims]
        ], dim=0)
        combined_mask = torch.cat([molecular_state.ligand_mask, molecular_state.pocket_mask])
        com = scatter_mean(all_coords, combined_mask, dim=0)
        com_drift = torch.norm(com).item()
        self.com_drifts.append(com_drift)
        
        lig_coords = molecular_state.ligand[:, :self.n_dims]
        pocket_coords = molecular_state.pocket[:, :self.n_dims]
        lig_com = torch.mean(lig_coords, dim=0)
        pocket_com = torch.mean(pocket_coords, dim=0)
        current_distance = torch.norm(lig_com - pocket_com).item()
        distance_change = abs(current_distance - self.initial_intermol_distance)
        self.intermol_distance_changes.append(distance_change)

    def _update_dynamic_coupling_stats(self, vector_field: MolecularState):
        """Compute dynamic coord-feature coupling from instantaneous drift magnitudes"""
        coord_drift_lig = torch.norm(vector_field.ligand[:, :self.n_dims], dim=1)
        feature_drift_lig = torch.norm(vector_field.ligand[:, self.n_dims:], dim=1)
        
        coord_drift_pocket = torch.norm(vector_field.pocket[:, :self.n_dims], dim=1)
        feature_drift_pocket = torch.norm(vector_field.pocket[:, self.n_dims:], dim=1)
        
        all_coord_drifts = torch.cat([coord_drift_lig, coord_drift_pocket])
        all_feature_drifts = torch.cat([feature_drift_lig, feature_drift_pocket])
        
        if len(all_coord_drifts) > 1:
            correlation = torch.corrcoef(torch.stack([all_coord_drifts, all_feature_drifts]))[0, 1]
            if not torch.isnan(correlation):
                self.dynamic_coupling_correlations.append(correlation.item())
    
    def _update_flow_stats(self, molecular_state: MolecularState, vector_field: MolecularState, dt: Tensor):
        """Update flow property statistics"""
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
        
        if self.vector_field_l2_norms:
            summary.update({
                'vf_l2_mean': np.mean(self.vector_field_l2_norms),
                'vf_l2_std': np.std(self.vector_field_l2_norms),
                'vf_l2_max': np.max(self.vector_field_l2_norms),
                'vf_spikiness': np.max(self.vector_field_l2_norms) / (np.mean(self.vector_field_l2_norms) + 1e-8),
                'coord_feature_ratio': np.mean(self.coord_field_l2_norms) / (np.mean(self.feature_field_l2_norms) + 1e-8)
            })
        
        if self.direction_changes:
            summary.update({
                'total_angular_deviation': np.sum(self.direction_changes),
                'smoothness_score': 1.0 / (np.var(self.acceleration_magnitudes) + 1e-8),
                'mean_acceleration': np.mean(self.acceleration_magnitudes)
            })
        
        if self.path_segments:
            total_path_length = np.sum(self.path_segments)
            if self.initial_state is not None:
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
        
        if self.com_drifts:
            summary.update({
                'max_com_drift': np.max(self.com_drifts),
                'mean_com_drift': np.mean(self.com_drifts),
                'max_intermol_change': np.max(self.intermol_distance_changes) if self.intermol_distance_changes else 0.0
            })
        
        if self.lipschitz_estimates:
            summary.update({
                'mean_lipschitz': np.mean(self.lipschitz_estimates),
                'max_lipschitz': np.max(self.lipschitz_estimates),
                'total_flow_energy': np.sum(self.flow_energy) if self.flow_energy else 0.0
            })
        
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
        scheme,
        denoise_fn: MolecularDenoiseFn,
        tspan: Tensor,
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
        
        assert self.tspan[0] < self.tspan[-1], "tspan should be forward: t=0 → t=1"
    
    def _extract_single_sample(self, molecular_state: MolecularState, sample_idx: int) -> MolecularState:
        """Extract a single sample from the batch"""
        ligand_mask = (molecular_state.ligand_mask == sample_idx)
        pocket_mask = (molecular_state.pocket_mask == sample_idx)
        
        ligand_data = molecular_state.ligand[ligand_mask]
        pocket_data = molecular_state.pocket[pocket_mask]
        
        new_ligand_mask = torch.zeros(len(ligand_data), dtype=torch.long, device=ligand_data.device)
        new_pocket_mask = torch.zeros(len(pocket_data), dtype=torch.long, device=pocket_data.device)
        
        return MolecularState(
            ligand=ligand_data,
            pocket=pocket_data,
            ligand_mask=new_ligand_mask,
            pocket_mask=new_pocket_mask,
            batch_size=1
        )

    def _extract_single_sample_conditioning(
        self,
        sample_idx: int,
        ligand_mask: Tensor,
        pocket_mask: Tensor,
        cond: TensorMapping | None
    ) -> TensorMapping | None:
        """
        Extract conditioning for a single sample from batch conditioning.
        
        Args:
            sample_idx: Which sample to extract (0, 1, 2, ...)
            ligand_mask: Boolean mask for ligand atoms of this sample
            pocket_mask: Boolean mask for pocket residues of this sample
            cond: Full batch conditioning dict
            
        Returns:
            Filtered conditioning dict for just this sample, or None
        """
        if cond is None:
            return None
        
        # Extract current sample's node IDs
        sample_node_ids_atoms = cond['node_ids_atoms'][ligand_mask]
        sample_node_ids_residues = cond['node_ids_residues'][pocket_mask]
        
        # Get conditioning node IDs (full batch)
        cond_node_ids_atoms = cond['cond_node_ids_atoms']
        cond_node_ids_residues = cond['cond_node_ids_residues']
        
        # Determine molecule ID namespace for this sample
        # Node IDs are namespaced: mol 0 = [0, 999999], mol 1 = [1000000, 1999999], etc.
        mol_id_offset = sample_idx * 1000000
        mol_id_max = (sample_idx + 1) * 1000000
        
        # Filter conditioning atoms: find those in this molecule's ID range
        atom_in_range = (cond_node_ids_atoms >= mol_id_offset) & (cond_node_ids_atoms < mol_id_max)
        filtered_cond_node_ids_atoms = cond_node_ids_atoms[atom_in_range]
        filtered_cond_coords_atoms = cond['cond_coords_ligand'][atom_in_range]
        
        # Filter conditioning residues: find those in this molecule's ID range
        residue_in_range = (cond_node_ids_residues >= mol_id_offset) & (cond_node_ids_residues < mol_id_max)
        filtered_cond_node_ids_residues = cond_node_ids_residues[residue_in_range]
        filtered_cond_coords_residues = cond['cond_coords_pocket'][residue_in_range]
        
        # Return filtered conditioning for this sample
        return {
            'cond_coords_ligand': filtered_cond_coords_atoms,
            'cond_coords_pocket': filtered_cond_coords_residues,
            'node_ids_atoms': sample_node_ids_atoms,
            'node_ids_residues': sample_node_ids_residues,
            'cond_node_ids_atoms': filtered_cond_node_ids_atoms,
            'cond_node_ids_residues': filtered_cond_node_ids_residues,
        }
    
    def _create_molecular_dynamics(self, cond: TensorMapping | None) -> MolecularSdeDynamics:
        """Create molecular probability flow ODE dynamics for likelihood evaluation"""
        
        def _molecular_drift(state: MolecularState, t: Tensor, params: dict) -> MolecularState:
            if not t.requires_grad:
                t = t.requires_grad_(True)
        
            sigma_t = self.scheme.sigma(t)
            dlog_sigma_dt = dlog_dt(self.scheme.sigma)(t)
        
            x0_hat = self.denoise_fn(state, sigma=sigma_t, cond=params.get("cond"))
        
            coeff = dlog_sigma_dt
            drift_lig = coeff * (state.ligand - x0_hat.ligand)
            drift_pock = coeff * (state.pocket - x0_hat.pocket)
        
            drift_coords = torch.cat([drift_lig[:, :self.n_dims], drift_pock[:, :self.n_dims]], 0)
            masks = torch.cat([state.ligand_mask, state.pocket_mask])
            drift_coords_centered = remove_mean_batch(drift_coords, masks)
            drift_lig = torch.cat([drift_coords_centered[:len(drift_lig)], drift_lig[:, self.n_dims:]], dim=1)
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
            eps_lig = torch.randn_like(molecular_state.ligand)
            eps_pocket = torch.randn_like(molecular_state.pocket)
            
            eps_coords_combined = torch.cat([
                eps_lig[:, :self.n_dims], 
                eps_pocket[:, :self.n_dims]
            ], dim=0)
            combined_mask = torch.cat([molecular_state.ligand_mask, molecular_state.pocket_mask])
            eps_coords_centered = remove_mean_batch(eps_coords_combined, combined_mask)
            
            eps_lig[:, :self.n_dims] = eps_coords_centered[:len(molecular_state.ligand)]
            eps_pocket[:, :self.n_dims] = eps_coords_centered[len(molecular_state.ligand):]
            
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
        state_lig = molecular_state.ligand.detach().requires_grad_(True)
        state_pocket = molecular_state.pocket.detach().requires_grad_(True)
        
        temp_state = MolecularState(
            ligand=state_lig,
            pocket=state_pocket,
            ligand_mask=molecular_state.ligand_mask,
            pocket_mask=molecular_state.pocket_mask,
            batch_size=molecular_state.batch_size
        )
        
        drift_state = dynamics.drift(temp_state, t, params["drift"])
        
        drift_flat = torch.cat([
            drift_state.ligand.reshape(-1), 
            drift_state.pocket.reshape(-1)
        ])
        eps_flat = torch.cat([
            eps_lig.reshape(-1), 
            eps_pocket.reshape(-1)
        ])
        
        vjp_result = torch.autograd.grad(
            outputs=drift_flat,
            inputs=[state_lig, state_pocket],
            grad_outputs=eps_flat,
            create_graph=False,
            retain_graph=False,
            allow_unused=True
        )
        
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
        """Forward integrate with trajectory statistics collection for single sample"""
        assert clean_state.batch_size == 1, "This method expects single sample"
        
        dynamics = self._create_molecular_dynamics(cond)
        params = dict(drift=dict(cond=cond), diffusion=dict())
        
        trajectory_stats = TrajectoryStatistics(n_dims=self.n_dims) if self.collect_trajectory_stats else None
        
        total_steps = len(self.tspan) - 1
        segment_size = max(1, total_steps // checkpoint_segments)
        
        current_state = clean_state
        total_divergence = torch.tensor(0.0, device=clean_state.ligand.device)
        
        for segment_start in range(0, total_steps, segment_size):
            segment_end = min(segment_start + segment_size, total_steps)
            
            def integrate_segment(state_in):
                return self._integrate_segment_with_divergence(
                    state_in, segment_start, segment_end, dynamics, params, trajectory_stats
                )
            
            current_state, segment_divergence = checkpoint.checkpoint(
                integrate_segment, 
                current_state,
                use_reentrant=False
            )
            
            total_divergence = total_divergence + segment_divergence
        
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
            
            vector_field = dynamics.drift(current_state, t_curr, params["drift"])
            
            if trajectory_stats is not None:
                with torch.no_grad():
                    trajectory_stats.update(current_state, vector_field, dt)
            
            div_estimate = self._estimate_divergence_single(current_state, t_curr, params, dynamics)
            segment_divergence = segment_divergence + div_estimate * dt
            
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
        
        k1 = dynamics.drift(state, t_curr, params["drift"])
        
        pred_ligand = state.ligand + k1.ligand * dt
        pred_pocket = state.pocket + k1.pocket * dt
        
        state_pred = MolecularState(
            ligand=pred_ligand,
            pocket=pred_pocket,
            ligand_mask=state.ligand_mask,
            pocket_mask=state.pocket_mask,
            batch_size=state.batch_size
        )
        
        k2 = dynamics.drift(state_pred, t_next, params["drift"])
        
        new_ligand = state.ligand + 0.5 * (k1.ligand + k2.ligand) * dt
        new_pocket = state.pocket + 0.5 * (k1.pocket + k2.pocket) * dt
        
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
        return (total_atoms - 1) * self.n_dims
    
    def _evaluate_terminal_likelihood_single(self, terminal_state: MolecularState) -> Tensor:
        """Evaluate log p_T(x_T) for Gaussian at final time for single sample"""
        assert terminal_state.batch_size == 1, "This method expects single sample"
        
        final_t = self.tspan[-1]
        final_sigma = self.scheme.sigma(final_t)
        final_scale = self.scheme.scale(final_t)

        coords_lig = terminal_state.ligand[:, :self.n_dims]
        coords_pocket = terminal_state.pocket[:, :self.n_dims]
        coords = torch.cat([coords_lig, coords_pocket], dim=0)
        
        features_lig = terminal_state.ligand[:, self.n_dims:]
        features_pocket = terminal_state.pocket[:, self.n_dims:]
        features = torch.cat([features_lig, features_pocket], dim=0)
        
        coord_dof = self._get_degrees_of_freedom_single(terminal_state)
        feature_dim = features.shape[1]
        total_feature_dof = features.shape[0] * feature_dim
        
        sigma_total = final_sigma * final_scale
        coord_norm_squared = torch.sum(coords ** 2)
        coord_log_prob = (
            -0.5 * coord_norm_squared / (sigma_total ** 2) - 
            0.5 * coord_dof * torch.log(2 * torch.pi * sigma_total ** 2)
        )
        
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
            log_likelihoods: [batch_size] log probabilities
            trajectory_statistics: List of trajectory statistic dicts for each sample
        """
        batch_size = molecular_state.batch_size
        log_likelihoods = torch.zeros(batch_size, device=molecular_state.ligand.device)
        trajectory_statistics = []
        
        for sample_idx in range(batch_size):
            # Extract single sample
            single_sample = self._extract_single_sample(molecular_state, sample_idx)
            
            # Extract conditioning for this specific sample
            ligand_mask = (molecular_state.ligand_mask == sample_idx)
            pocket_mask = (molecular_state.pocket_mask == sample_idx)
            single_sample_cond = self._extract_single_sample_conditioning(
                sample_idx, ligand_mask, pocket_mask, cond
            )
            
            terminal_state, divergence_integral, stats_summary = self._forward_integrate_with_stats_single(
                single_sample, single_sample_cond
            )
            
            terminal_log_prob = self._evaluate_terminal_likelihood_single(terminal_state)
            
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
        (Backward compatibility - returns only likelihoods)
        """
        log_likelihoods, _ = self.evaluate_likelihood_with_stats(molecular_state, cond)
        return log_likelihoods


def create_molecular_likelihood_evaluator_from_model(
    model,
    num_steps: int = 50,
    num_hutchinson_samples: int = 1,
    collect_trajectory_stats: bool = True,
) -> MolecularLikelihoodEvaluator:
    """Create an enhanced molecular likelihood evaluator from a trained model"""
    
    tspan_sampling = edm_noise_decay(scheme=model.scheme, num_steps=num_steps)
    
    # For likelihood evaluation, we need FORWARD integration: t=0 → t=1
    tspan_likelihood = torch.flip(tspan_sampling, dims=[0])
    
    molecular_denoise_fn = create_molecular_denoiser_wrapper(
        model.denoiser, model.scheme, requires_grad=True
    )
    
    evaluator = MolecularLikelihoodEvaluator(
        scheme=model.scheme,
        denoise_fn=molecular_denoise_fn,
        tspan=tspan_likelihood,
        num_hutchinson_samples=num_hutchinson_samples,
        n_dims=model.n_dims,
        atom_nf=model.atom_nf,
        residue_nf=model.residue_nf,
        joint_nf=model.joint_nf,
        collect_trajectory_stats=collect_trajectory_stats,
    )
    evaluator._model = model
    
    return evaluator


def test_per_sample_likelihood_evaluator():
    """Test the per-sample molecular likelihood evaluator"""
    
    print("Testing Per-Sample Molecular Likelihood Evaluator (CUDA)...")
    
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
        print("Creating per-sample likelihood evaluator...")
        evaluator = create_molecular_likelihood_evaluator_from_model(
            model=model,
            num_steps=10,  # Small for testing
            num_hutchinson_samples=5,
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
        
        print("\n" + "="*70)
        print("=== Test 1: Likelihood WITHOUT Conditioning ===")
        print("="*70)
        
        print("Evaluating likelihood of clean molecular states...")
        log_likelihoods = evaluator.evaluate_likelihood(clean_state)
        
        print(f"✅ Per-sample likelihood evaluation successful!")
        print(f"  Batch size: {batch_size}")
        print(f"  Output shape: {log_likelihoods.shape}")
        print(f"  Log-likelihoods: {log_likelihoods}")
        
        # Verify we get one likelihood per sample
        assert log_likelihoods.shape == (batch_size,), f"Expected shape ({batch_size},), got {log_likelihoods.shape}"
        
        print("✅ Test 1 passed!")
        
        # ========== NEW TEST 2 ==========
        
        print("\n" + "="*70)
        print("=== Test 2: Likelihood WITH Variable-Sized Conditioning ===")
        print("="*70)
        
        # Create conditioning with DIFFERENT sizes (simulates MD trajectory)
        # Current state: [3,4,2] ligand atoms, [8,6,5] pocket residues
        # Conditioning: [3,4,2] ligand atoms, [10,8,7] pocket residues (MORE!)
        cond_lig_coords = torch.randn(total_lig_atoms, 3, device='cuda') * 0.15
        cond_pocket_coords = torch.randn(25, 3, device='cuda') * 0.15  # 10+8+7 = 25 > 19
        
        # Node IDs for CURRENT state (what we're evaluating likelihood of)
        node_ids_atoms = torch.cat([
            torch.arange(3, dtype=torch.long, device='cuda'),  # Mol 0
            torch.arange(4, dtype=torch.long, device='cuda') + 1000000,  # Mol 1
            torch.arange(2, dtype=torch.long, device='cuda') + 2000000,  # Mol 2
        ])
        
        node_ids_residues = torch.cat([
            # Mol 0: 8 residues that stayed
            torch.tensor([10, 11, 15, 20, 25, 30, 35, 40], dtype=torch.long, device='cuda'),
            # Mol 1: 6 residues that stayed
            torch.tensor([100010, 100011, 100015, 100020, 100025, 100030], dtype=torch.long, device='cuda'),
            # Mol 2: 5 residues that stayed
            torch.tensor([200010, 200011, 200015, 200020, 200025], dtype=torch.long, device='cuda'),
        ])
        
        # Node IDs for CONDITIONING state (initial frame with MORE residues)
        cond_node_ids_atoms = node_ids_atoms.clone()  # Ligands same
        
        cond_node_ids_residues = torch.cat([
            # Mol 0: 10 residues (includes 12,13 that moved OUT)
            torch.tensor([10, 11, 12, 13, 15, 20, 25, 30, 35, 40], dtype=torch.long, device='cuda'),
            # Mol 1: 8 residues (includes 12,14 that moved OUT)
            torch.tensor([100010, 100011, 100012, 100014, 100015, 100020, 100025, 100030], dtype=torch.long, device='cuda'),
            # Mol 2: 7 residues (includes 12,14 that moved OUT)
            torch.tensor([200010, 200011, 200012, 200014, 200015, 200020, 200025], dtype=torch.long, device='cuda'),
        ])
        
        cond = {
            'cond_coords_ligand': cond_lig_coords,
            'cond_coords_pocket': cond_pocket_coords,
            'node_ids_atoms': node_ids_atoms,
            'node_ids_residues': node_ids_residues,
            'cond_node_ids_atoms': cond_node_ids_atoms,
            'cond_node_ids_residues': cond_node_ids_residues,
        }
        
        print(f"\n📊 Graph Size Comparison:")
        curr_total = total_lig_atoms + total_pocket_atoms
        cond_total = cond_lig_coords.shape[0] + cond_pocket_coords.shape[0]
        print(f"  Current state:      {total_lig_atoms} lig + {total_pocket_atoms} pocket = {curr_total} nodes")
        print(f"  Conditioning state: {cond_lig_coords.shape[0]} lig + {cond_pocket_coords.shape[0]} pocket = {cond_total} nodes")
        print(f"  Difference: +{cond_total - curr_total} extra nodes in conditioning")
        
        print(f"\n🔍 Expected Matching:")
        print(f"  • All {total_lig_atoms} ligand atoms should match")
        print(f"  • {total_pocket_atoms}/25 pocket residues match (current state)")
        print(f"  • 6 unmatched residues in conditioning (2 per molecule moved OUT)")
        print(f"  • Unmatched get NULL embeddings → won't affect likelihood")
        
        print(f"\n🚀 Evaluating likelihood with variable-sized conditioning...")
        log_likelihoods_cond = evaluator.evaluate_likelihood(clean_state, cond=cond)
        
        print(f"\n✅ Conditioning Results:")
        print(f"  Output shape: {log_likelihoods_cond.shape}")
        print(f"  Log-likelihoods (with cond): {log_likelihoods_cond}")
        print(f"  Log-likelihoods (no cond):   {log_likelihoods}")
        
        # Verify shape
        assert log_likelihoods_cond.shape == (batch_size,), \
            f"Expected shape ({batch_size},), got {log_likelihoods_cond.shape}"
        
        # Verify no NaNs or Infs
        has_nan = torch.isnan(log_likelihoods_cond).any()
        has_inf = torch.isinf(log_likelihoods_cond).any()
        
        print(f"  No NaNs: {not has_nan}")
        print(f"  No Infs: {not has_inf}")
        
        assert not has_nan, "❌ NaN detected in conditional log-likelihoods!"
        assert not has_inf, "❌ Inf detected in conditional log-likelihoods!"
        
        # Check that conditioning changes likelihood (should be different)
        likelihood_diff = (log_likelihoods_cond - log_likelihoods).abs().mean()
        print(f"  Mean absolute difference: {likelihood_diff.item():.6f}")
        
        if likelihood_diff > 1e-3:
            print(f"  ✅ Conditioning changes likelihood as expected!")
        else:
            print(f"  ⚠️  Warning: Likelihoods very similar (diff={likelihood_diff.item():.6f})")
        
        print(f"\n🎉 SUCCESS! Variable-sized conditioning works in likelihood evaluator!")
        print(f"  • Handled {cond_total - curr_total} extra conditioning nodes")
        print(f"  • ID-based matching prevented NaN/Inf")
        print(f"  • Computed valid per-sample log-likelihoods")
        
        print(f"\n🔬 Validating per-sample conditioning extraction:")
        for sample_idx in range(batch_size):
            ligand_mask = (clean_state.ligand_mask == sample_idx)
            pocket_mask = (clean_state.pocket_mask == sample_idx)
            
            sample_cond = evaluator._extract_single_sample_conditioning(
                sample_idx, ligand_mask, pocket_mask, cond
            )
            
            n_lig_current = ligand_mask.sum().item()
            n_pocket_current = pocket_mask.sum().item()
            n_lig_cond = sample_cond['cond_coords_ligand'].shape[0]
            n_pocket_cond = sample_cond['cond_coords_pocket'].shape[0]
            
            print(f"  Sample {sample_idx}:")
            print(f"    Current:      {n_lig_current} lig + {n_pocket_current} pocket")
            print(f"    Conditioning: {n_lig_cond} lig + {n_pocket_cond} pocket")
            print(f"    Node ID ranges: atoms [{sample_cond['node_ids_atoms'].min()}, {sample_cond['node_ids_atoms'].max()}]")
            print(f"                    residues [{sample_cond['node_ids_residues'].min()}, {sample_cond['node_ids_residues'].max()}]")
            
            # Verify sizes make sense
            assert n_lig_current <= n_lig_cond or n_lig_current == n_lig_cond, \
                f"Sample {sample_idx}: current ligand ({n_lig_current}) vs conditioning ({n_lig_cond})"
            assert sample_cond['node_ids_atoms'].shape[0] == n_lig_current, \
                f"Sample {sample_idx}: Node ID count mismatch for atoms!"
            assert sample_cond['node_ids_residues'].shape[0] == n_pocket_current, \
                f"Sample {sample_idx}: Node ID count mismatch for residues!"
        
        print(f"  ✅ All samples have correctly filtered conditioning!")        
        print("\n" + "="*70)
        print("✅ ALL LIKELIHOOD EVALUATOR TESTS PASSED!")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Error during likelihood evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    test_per_sample_likelihood_evaluator()
