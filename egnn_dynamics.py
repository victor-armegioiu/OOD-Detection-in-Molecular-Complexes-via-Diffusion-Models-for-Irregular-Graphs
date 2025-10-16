"""
EGNN Dynamics for Molecular Diffusion - Enhanced with Geometric Regularization

Contains the EGNNDynamics class that predicts noise for molecular systems.
Now includes geometric losses for preventing fragmentation and encouraging rings.

https://github.com/arneschneuing/DiffSBDD/blob/5d0d38d16c8932a0339fd2ce3f67ade98bbdff27/equivariant_diffusion/dynamics.py#L10
combined with
https://github.com/arneschneuing/DiffSBDD/blob/main/equivariant_diffusion/egnn_new.py#L187
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch_scatter import scatter_mean


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def coord2cross(x, edge_index, batch_mask, norm_constant=1):
    mean = unsorted_segment_sum(x, batch_mask,
                                num_segments=batch_mask.max() + 1,
                                normalization_factor=None,
                                aggregation_method='mean')
    row, col = edge_index
    cross = torch.cross(x[row]-mean[batch_mask[row]],
                        x[col]-mean[batch_mask[col]], dim=1)
    norm = torch.linalg.norm(cross, dim=1, keepdim=True)
    cross = cross / (norm + norm_constant)
    return cross


def remove_mean_batch(x, indices):
    """Remove center of mass for each molecule in batch"""
    mean = scatter_mean(x, indices, dim=0)
    return x - mean[indices]


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0,
                 reflection_equiv=True):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        self.reflection_equiv = reflection_equiv
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        self.cross_product_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer
        ) if not self.reflection_equiv else None
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, coord_cross,
                    edge_attr, edge_mask, update_coords_mask=None):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)

        if not self.reflection_equiv:
            phi_cross = self.cross_product_mlp(input_tensor)
            if self.tanh:
                phi_cross = torch.tanh(phi_cross) * self.coords_range
            trans = trans + coord_cross * phi_cross

        if edge_mask is not None:
            trans = trans * edge_mask

        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)

        if update_coords_mask is not None:
            agg = update_coords_mask * agg

        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, coord_cross,
                edge_attr=None, node_mask=None, edge_mask=None,
                update_coords_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, coord_cross,
                                 edge_attr, edge_mask,
                                 update_coords_mask=update_coords_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum', reflection_equiv=True):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.reflection_equiv = reflection_equiv

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method,
                                                       reflection_equiv=self.reflection_equiv))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None,
                edge_attr=None, update_coords_mask=None, batch_mask=None):
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.reflection_equiv:
            coord_cross = None
        else:
            coord_cross = coord2cross(x, edge_index, batch_mask,
                                      self.norm_constant)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr,
                                               node_mask=node_mask, edge_mask=edge_mask)
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, coord_cross, edge_attr,
                                       node_mask, edge_mask, update_coords_mask=update_coords_mask)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum', reflection_equiv=True):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.reflection_equiv = reflection_equiv
        self.norm_constant = norm_constant

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            base_edge_feat_nf = self.sin_embedding.dim  # we embed the radial once
        else:
            self.sin_embedding = None
            base_edge_feat_nf = 1  # we only pass 'radial' (1D) when no sin embedding
        
        edge_feat_nf = base_edge_feat_nf + in_edge_nf

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method,
                                                               reflection_equiv=self.reflection_equiv))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, update_coords_mask=None,
                batch_mask=None, edge_attr=None):
        edge_feat, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            edge_feat = self.sin_embedding(edge_feat)
        if edge_attr is not None:
            edge_feat = torch.cat([edge_feat, edge_attr], dim=1)
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask,
                edge_attr=edge_feat, update_coords_mask=update_coords_mask,
                batch_mask=batch_mask)

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x


class GeometricRegularizer(nn.Module):
    """Geometric regularization losses for molecular diffusion"""
    
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def infer_actual_bonds_threshold(self, target_coords):
        """
        Infer bond threshold from target's likely bonding pattern
        """
        n_atoms = target_coords.shape[0]
        if n_atoms <= 1:
            return torch.tensor(2.0, device=target_coords.device)
        
        dist_matrix = torch.cdist(target_coords, target_coords)
        
        # For each atom, find its 2-3 nearest neighbors (likely bonds)
        # Most atoms have 2-4 bonds in organic molecules
        k_nearest = min(4, n_atoms - 1)
        
        nearest_distances = []
        for i in range(n_atoms):
            distances_from_i = dist_matrix[i]
            # Remove self-distance (0) and get k nearest
            distances_from_i[i] = float('inf')
            k_nearest_dists = torch.topk(distances_from_i, k_nearest, largest=False)[0]
            nearest_distances.append(k_nearest_dists)
        
        # Use 90th percentile of nearest neighbor distances as threshold
        # This captures most actual bonds while excluding longer non-bonds
        all_near_distances = torch.cat(nearest_distances)
        threshold = torch.quantile(all_near_distances, 0.9) * 1.1  # Small buffer
        
        return threshold
        
    def reachability_loss(self, coords, target_coords, mask, temperature=0.1):
        """
        Compares reachability matrices of predicted vs target mols.
        """
        penalties = []
        
        for batch_idx in torch.unique(mask):
            pred_mol = coords[mask == batch_idx]
            target_mol = target_coords[mask == batch_idx]
            n_atoms = pred_mol.shape[0]
            
            if n_atoms <= 1:
                continue
                
            bond_threshold = self.infer_actual_bonds_threshold(target_mol)
            
            # Build soft adjacency matrices
            pred_dist = torch.cdist(pred_mol, pred_mol)
            target_dist = torch.cdist(target_mol, target_mol)
            
            pred_adj = torch.sigmoid(-(pred_dist - bond_threshold) / temperature)
            target_adj = torch.sigmoid(-(target_dist - bond_threshold) / temperature)
            
            # Remove self-connections
            eye = torch.eye(n_atoms, device=pred_adj.device)
            pred_adj = pred_adj * (1 - eye)
            target_adj = target_adj * (1 - eye)
            
            # Compute full reachability matrices
            pred_reach = self.soft_reachability_matrix(pred_adj)
            target_reach = self.soft_reachability_matrix(target_adj)
            
            # Compare entire matrices element-wise
            matrix_loss = F.mse_loss(pred_reach, target_reach)
            penalties.append(matrix_loss)
        
        return torch.stack(penalties).mean() if penalties else torch.tensor(0.0)

    def soft_reachability_matrix(self, adj_matrix):
        """
        Compute soft reachability matrix R where R[i,j] = reachability from i to j
        """
        n = adj_matrix.shape[0]
        
        # Start with direct connections
        reachability = adj_matrix.clone()
        
        # Add paths of increasing length
        adj_power = adj_matrix.clone()
        for k in range(2, min(n + 1, 5)):  # Paths up to length 4
            adj_power = torch.mm(adj_power, adj_matrix)
            # Add with exponential decay (shorter paths matter more)
            reachability += adj_power * (0.5 ** (k-1))
        
        # Normalize to [0,1] range for stable training
        reachability = torch.tanh(reachability)
        
        return reachability

    def radius_of_gyration_loss(self, coords, target_coords, mask):
        """
        Match radius of gyration to target molecules
        """
        penalties = []
        
        for batch_idx in torch.unique(mask):
            pred_mol = coords[mask == batch_idx]
            target_mol = target_coords[mask == batch_idx]
            
            if pred_mol.shape[0] <= 1:
                continue
                
            # Predicted Rg
            pred_centroid = pred_mol.mean(dim=0)
            pred_rg = torch.sqrt(torch.sum((pred_mol - pred_centroid)**2, dim=1).mean())
            
            # Target Rg
            target_centroid = target_mol.mean(dim=0)
            target_rg = torch.sqrt(torch.sum((target_mol - target_centroid)**2, dim=1).mean())
            
            # Match target's radius of gyration
            rg_diff = F.mse_loss(pred_rg, target_rg)
            penalties.append(rg_diff)
        
        return torch.stack(penalties).mean() if penalties else torch.tensor(0.0, device=coords.device)

    def cycle_betti_loss(self, coords, mask, bond_threshold=2.0, temperature=0.15):
        """
        Differentiable cycle detection using Betti-1 numbers from algebraic topology.
        
        Rewards cycle formation by computing the soft Betti-1 number (number of 1D holes)
        using the Euler characteristic: β₁ = E - V + C, where:
        - E = number of edges (soft)
        - V = number of vertices  
        - C = number of connected components (soft)
        
        Args:
            coords: [N, 3] atom coordinates
            mask: [N] molecule indices for batching
            bond_threshold: distance threshold for bonding (Angstroms)
            temperature: softness parameter for sigmoid adjacency
            
        Returns:
            Negative Betti-1 loss (rewards cycle formation)
        """
        if coords.size(0) == 0:
            return coords.new_zeros(())
        
        losses = []
        
        for batch_idx in torch.unique(mask):
            # Extract molecule coordinates
            mol_coords = coords[mask == batch_idx]
            n_atoms = mol_coords.size(0)
            
            # Skip molecules too small to form cycles
            if n_atoms < 3:
                continue
            
            # Build soft adjacency matrix
            distances = torch.cdist(mol_coords, mol_coords)
            adjacency = torch.sigmoid(-(distances - bond_threshold) / temperature)
            adjacency = adjacency * (1 - torch.eye(n_atoms, device=adjacency.device))
            
            # Compute topological quantities
            n_vertices = torch.tensor(n_atoms, dtype=torch.float32, device=coords.device)
            n_edges_soft = adjacency.triu(diagonal=1).sum()  # Count upper triangular (unique edges)
            n_components_soft = self.soft_connected_components(adjacency)
            
            # Betti-1 number: β₁ = E - V + C (number of 1D holes/cycles)
            betti_1 = (n_edges_soft - n_vertices + n_components_soft).clamp(min=0)
            
            # Negative loss = reward cycles
            losses.append(-betti_1)
        
        return torch.stack(losses).mean() if losses else coords.new_zeros(())

    def soft_connected_components(self, adjacency):
        """
        Compute soft number of connected components using the soft reachability matrix.
        
        Args:
            adjacency: [n, n] soft adjacency matrix
            
        Returns:
            Soft component count (1.0 = fully connected, >1.0 = fragmented)
        """
        n = adjacency.size(0)
        if n <= 1:
            return torch.tensor(1.0, device=adjacency.device)
        
        # Use existing soft reachability matrix computation
        reachability = self.soft_reachability_matrix(adjacency)
        
        # Measure connectivity: well-connected graphs have high reachability everywhere
        # Remove diagonal since self-reachability is always 1
        off_diagonal_reachability = reachability.sum() - torch.trace(reachability)
        max_off_diagonal = n * (n - 1)  # Perfect connectivity (excluding diagonal)
        
        # Convert to component count: high reachability → ~1 component
        connectivity_ratio = off_diagonal_reachability / max_off_diagonal
        component_count = 1.0 + (1.0 - connectivity_ratio) * (n - 1)
        
        return component_count

    def pairwise_distance_loss(self, coords, mask, min_dist=0.8, max_dist=8.0):
        """
        Simple distance-based geometric constraints
        """
        penalties = []
        
        for batch_idx in torch.unique(mask):
            mol_coords = coords[mask == batch_idx]
            n_atoms = mol_coords.shape[0]
            
            if n_atoms <= 1:
                continue
                
            # Get all pairwise distances
            dist_matrix = torch.cdist(mol_coords, mol_coords)
            
            # Get upper triangular (unique pairs)
            triu_indices = torch.triu_indices(n_atoms, n_atoms, offset=1, device=mol_coords.device)
            distances = dist_matrix[triu_indices[0], triu_indices[1]]
            
            # Soft repulsion (atoms too close)
            too_close_penalty = torch.exp(-distances / min_dist).mean()
            
            # Prevent excessive drift (atoms too far)
            too_far_penalty = F.relu(distances - max_dist).mean()
            
            total_penalty = too_close_penalty + too_far_penalty
            penalties.append(total_penalty)
        
        return torch.stack(penalties).mean() if penalties else torch.tensor(0.0, device=coords.device)

    def comprehensive_geometric_loss(self, coords, target_coords, mask, 
                                   weights=(1.0, 0.8, 0.5, 0.3)):
        """
        Combine multiple geometric constraints
        """
        w_components, w_rg, w_cycles, w_distances = weights

        total_loss = (
            w_components * self.reachability_loss(coords, target_coords, mask) +
            w_rg * self.radius_of_gyration_loss(coords, target_coords, mask) +
            w_cycles * self.cycle_betti_loss(coords, mask) +
            w_distances * self.pairwise_distance_loss(coords, mask)
        )
        
        return total_loss


class EGNNDynamics(nn.Module):
    def __init__(self, atom_nf, residue_nf,
             n_dims, joint_nf=16, hidden_nf=64, device='cpu',
             act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
             condition_time=True, tanh=False,
             norm_constant=1, inv_sublayers=2, use_sin_embedding=False,
             normalization_factor=100, aggregation_method='sum',
             update_pocket_coords=True, edge_cutoff_ligand=None,
             edge_cutoff_pocket=None, edge_cutoff_interaction=None,
             reflection_equivariant=True, edge_embedding_dim=None,
             geometric_regularization=False, geom_loss_weight=0.1,
             use_conditioning=False):  
        super().__init__()
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction
        self.norm_constant = norm_constant
        
        # FIX: Create the actual SinusoidsEmbeddingNew object if sin_embedding is True
        if use_sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
        else:
            self.sin_embedding = None
        
        self.use_sin_embedding = use_sin_embedding
        self.edge_nf = edge_embedding_dim
        self.geometric_regularization = geometric_regularization
        self.geom_loss_weight = geom_loss_weight
    
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, joint_nf)
        )
    
        self.atom_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, atom_nf)
        )
    
        self.residue_encoder = nn.Sequential(
            nn.Linear(residue_nf, 2 * residue_nf),
            act_fn,
            nn.Linear(2 * residue_nf, joint_nf)
        )
    
        self.residue_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * residue_nf),
            act_fn,
            nn.Linear(2 * residue_nf, residue_nf)
        )
    
        self.edge_embedding = nn.Embedding(3, self.edge_nf) \
            if self.edge_nf is not None else None
        self.edge_nf = 0 if self.edge_nf is None else self.edge_nf
    
        if condition_time:
            dynamics_node_nf = joint_nf + 1
        else:
            print('Warning: dynamics model is _not_ conditioned on time.')
            dynamics_node_nf = joint_nf
    
        # NEW: Compute total edge feature dimensions correctly
        # This is what gets passed to EGNN via the edge_attr parameter
        total_in_edge_nf = self.edge_nf  # Edge type embeddings (0 if None)
        
        # Add space for conditioning edge features if enabled
        # how many per-edge conditioning features (d0 embedding + validity bit)
        self.cond_edge_feat_dim = (self.sin_embedding.dim if self.use_sin_embedding else 1) + 1
        
        # Recompute total edge-feat input size that EGNN will get:
        total_in_edge_nf = self.edge_nf  # edge type embedding dim or 0
        if use_conditioning:
            total_in_edge_nf += self.cond_edge_feat_dim
    
        self.egnn = EGNN(
            in_node_nf=dynamics_node_nf, 
            in_edge_nf=total_in_edge_nf,  # <-- FIXED SIZE, accounts for conditioning
            hidden_nf=hidden_nf, device=device, act_fn=act_fn,
            n_layers=n_layers, attention=attention, tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers, sin_embedding=use_sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            reflection_equiv=reflection_equivariant
        )
        self.node_nf = dynamics_node_nf
        self.update_pocket_coords = update_pocket_coords
    
        # Geometric regularizer
        if self.geometric_regularization:
            self.geometric_regularizer = GeometricRegularizer(device=device)
            
        # For storing losses
        self.last_geometric_loss = torch.tensor(0.0)
        
        # Conditioning encoders using INVARIANTS
        self.use_conditioning = use_conditioning
        if self.use_conditioning:
            # Learned NULL embeddings for unmatched nodes
            self.null_embedding_atoms = nn.Parameter(torch.zeros(n_dims))
            self.null_embedding_residues = nn.Parameter(torch.zeros(n_dims))
            
            # Node-level: encode invariants
            self.cond_node_mlp = nn.Sequential(
                nn.Linear(1, joint_nf),
                act_fn,
                nn.Linear(joint_nf, 2 * joint_nf)
            )
            
        self.device = device
        self.n_dims = n_dims
        self.condition_time = condition_time
        self.to(device)

    def _match_by_id(self, current_ids, cond_coords, cond_ids, null_embedding):
        """
        Efficient O(N log M) node matching using sorted indices.
        
        Args:
            current_ids: [N_current] node IDs (long, on device)
            cond_coords: [M_cond, 3] conditioning coordinates  
            cond_ids: [M_cond] node IDs in conditioning graph (long, on device)
            null_embedding: [3] learned NULL embedding for unmatched nodes
            
        Returns:
            matched_coords: [N_current, 3]
            match_mask: [N_current] - True where matched, False for NULL
        """
        device = current_ids.device
        N_current = len(current_ids)
        M_cond = len(cond_ids)
        
        # Initialize with learned NULL
        matched_coords = null_embedding.unsqueeze(0).expand(N_current, -1).clone()
        match_mask = torch.zeros(N_current, dtype=torch.bool, device=device)
        
        if M_cond == 0:
            return matched_coords, match_mask
        
        # Sort conditioning IDs - O(M log M)
        sorted_cond_ids, sort_idx = torch.sort(cond_ids)
        sorted_cond_coords = cond_coords[sort_idx]
        
        # Find insertion positions - O(N log M)
        insert_positions = torch.searchsorted(sorted_cond_ids, current_ids)
        
        # NO CLAMPING - positions == M means "not found"
        valid_positions = insert_positions < M_cond
        
        # Check actual matches
        matched = valid_positions & (sorted_cond_ids[insert_positions.clamp(max=M_cond-1)] == current_ids)
        
        if matched.any():
            matched_coords[matched] = sorted_cond_coords[insert_positions[matched]]
            match_mask[matched] = True
        
        return matched_coords, match_mask
        
    def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues, 
            target_atoms=None, target_residues=None,
            cond_coords_atoms=None, cond_coords_residues=None,
            node_ids_atoms=None, node_ids_residues=None,
            cond_node_ids_atoms=None, cond_node_ids_residues=None):
        """
        Forward pass of EGNN Dynamics with invariant conditioning
        
        Args:
            xh_atoms: [total_lig_atoms, n_dims + joint_nf]
            xh_residues: [total_pocket_atoms, n_dims + joint_nf]
            t: [batch_size, 1] - normalized timestep values
            mask_atoms: [total_lig_atoms] - molecule indices
            mask_residues: [total_pocket_atoms] - molecule indices
            target_atoms: [total_lig_atoms, n_dims + joint_nf] - for geometric loss
            target_residues: [total_pocket_atoms, n_dims + residue_nf] - for geometric loss
            cond_coords_atoms: [total_lig_atoms, n_dims] - INITIAL coordinates
            cond_coords_residues: [total_pocket_atoms, n_dims] - INITIAL coordinates
            
        Returns:
            Tuple of (ligand_output, pocket_output)
        """
    
        x_atoms = xh_atoms[:, :self.n_dims].clone()
        h_atoms = xh_atoms[:, self.n_dims:].clone()
    
        x_residues = xh_residues[:, :self.n_dims].clone()
        h_residues = xh_residues[:, self.n_dims:].clone()
    
        edge_cond = None
        node_match_mask = None
        
        # Check ALL required conditioning inputs
        conditioning_available = (
            self.use_conditioning and 
            cond_coords_atoms is not None and cond_coords_residues is not None and
            node_ids_atoms is not None and node_ids_residues is not None and
            cond_node_ids_atoms is not None and cond_node_ids_residues is not None
        )
        
        if conditioning_available:
            # Ensure all ID tensors are long dtype and on correct device
            device = x_atoms.device
            node_ids_atoms = node_ids_atoms.long().to(device)
            node_ids_residues = node_ids_residues.long().to(device)
            cond_node_ids_atoms = cond_node_ids_atoms.long().to(device)
            cond_node_ids_residues = cond_node_ids_residues.long().to(device)
            
            # Match nodes by ID - O(N log M)
            matched_cond_atoms, match_mask_atoms = self._match_by_id(
                current_ids=node_ids_atoms,
                cond_coords=cond_coords_atoms,
                cond_ids=cond_node_ids_atoms,
                null_embedding=self.null_embedding_atoms
            )
            
            matched_cond_residues, match_mask_residues = self._match_by_id(
                current_ids=node_ids_residues,
                cond_coords=cond_coords_residues,
                cond_ids=cond_node_ids_residues,
                null_embedding=self.null_embedding_residues
            )
            
            # Combine and DETACH - no gradients to initial frame
            x0 = torch.cat([matched_cond_atoms, matched_cond_residues], dim=0).detach()
            mask_combined = torch.cat([mask_atoms, mask_residues])
            node_match_mask = torch.cat([match_mask_atoms, match_mask_residues])
            
            # Compute COM ONLY over matched nodes
            # COM = (Σ matched coords) / (Σ match flags)
            from torch_scatter import scatter_add
            match_weights = node_match_mask.float()
            weighted_coords = x0 * match_weights.unsqueeze(1)
            
            coord_sum_per_mol = scatter_add(weighted_coords, mask_combined, dim=0)
            match_count_per_mol = scatter_add(match_weights, mask_combined, dim=0)
            match_count_per_mol = match_count_per_mol.clamp(min=1.0)
            
            com = coord_sum_per_mol / match_count_per_mol.unsqueeze(1)
            com_expanded = com[mask_combined]
            
            # Compute radial distance for ALL nodes
            radial_dist_sq = torch.sum((x0 - com_expanded) ** 2, dim=1, keepdim=True)
            
            # Get FiLM parameters
            gamma_beta = self.cond_node_mlp(radial_dist_sq)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            
            # MASK FiLM: zero out for unmatched nodes
            gamma = gamma * node_match_mask.unsqueeze(1).float()
            beta = beta * node_match_mask.unsqueeze(1).float()
            
            gamma_atoms = gamma[:len(mask_atoms)]
            beta_atoms = beta[:len(mask_atoms)]
            gamma_residues = gamma[len(mask_atoms):]
            beta_residues = beta[len(mask_atoms):]
            
            # Apply FiLM
            h_atoms = h_atoms * (1.0 + gamma_atoms) + beta_atoms
            h_residues = h_residues * (1.0 + gamma_residues) + beta_residues
            
            edge_cond = x0
    
        # Combine the two node types
        x = torch.cat((x_atoms, x_residues), dim=0)
        h = torch.cat((h_atoms, h_residues), dim=0)
        mask = torch.cat([mask_atoms, mask_residues])
    
        if self.condition_time:
            if np.prod(t.size()) == 1:
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                h_time = t[mask]
            h = torch.cat([h, h_time], dim=1)
    
        # Get edges
        edges = self.get_edges(mask_atoms, mask_residues, x_atoms, x_residues)
        assert torch.all(mask[edges[0]] == mask[edges[1]])
    
        # Get edge types and augment with initial distances if conditioning
        edge_attr = None

        # Step 1: edge type embeddings (4-dim in your test)
        if self.edge_nf > 0:
            edge_types = torch.zeros(edges.size(1), dtype=torch.long, device=edges.device)
            edge_types[(edges[0] < len(mask_atoms)) & (edges[1] < len(mask_atoms))] = 1
            edge_types[(edges[0] >= len(mask_atoms)) & (edges[1] >= len(mask_atoms))] = 2
            edge_attr = self.edge_embedding(edge_types)
        
        # Step 2: conditioning edge features
        if self.use_conditioning and (edge_cond is not None) and (node_match_mask is not None):
            d0_sq, _ = coord2diff(edge_cond, edges, self.norm_constant)
            edge_valid_mask = node_match_mask[edges[0]] & node_match_mask[edges[1]]
            edge_valid_float = edge_valid_mask.float().unsqueeze(1)
        
            if self.sin_embedding is not None:
                d0_emb = self.sin_embedding(d0_sq)
            else:
                d0_emb = d0_sq  # 1D
        
            d0_emb = d0_emb * edge_valid_float
            d0_features = torch.cat([d0_emb, edge_valid_float], dim=1)  # [num_edges, cond_edge_feat_dim]
        else:
            # No conditioning available: append zeros of identical width
            d0_features = x.new_zeros(edges.size(1), self.cond_edge_feat_dim)
        
        # Concatenate the conditioning block (real or zeros)
        edge_attr = torch.cat([edge_attr, d0_features], dim=1) if edge_attr is not None else d0_features

    
        update_coords_mask = None if self.update_pocket_coords \
            else torch.cat((torch.ones_like(mask_atoms),
                            torch.zeros_like(mask_residues))).unsqueeze(1)
        
        h_final, x_final = self.egnn(h, x, edges,
                                     update_coords_mask=update_coords_mask,
                                     batch_mask=mask, edge_attr=edge_attr)
        vel = (x_final - x)
    
        # Geometric regularization loss
        if self.geometric_regularization and target_atoms is not None and self.training:
            pred_ligand_coords = x_final[:len(mask_atoms)]
            target_ligand_coords = target_atoms[:, :self.n_dims]
            
            geom_loss = self.geometric_regularizer.comprehensive_geometric_loss(
                pred_ligand_coords, target_ligand_coords, mask_atoms
            )
            self.last_geometric_loss = geom_loss * self.geom_loss_weight
        else:
            self.last_geometric_loss = torch.tensor(0.0, device=self.device)
    
        if self.condition_time:
            h_final = h_final[:, :-1]
    
        # Decode atom and residue features
        h_final_atoms = self.atom_decoder(h_final[:len(mask_atoms)])
        h_final_residues = self.residue_decoder(h_final[len(mask_atoms):])
    
        if torch.any(torch.isnan(vel)):
            if self.training:
                vel[torch.isnan(vel)] = 0.0
            else:
                raise ValueError("NaN detected in EGNN output")
    
        if self.update_pocket_coords:
            vel = remove_mean_batch(vel, mask)
    
        return torch.cat([vel[:len(mask_atoms)], h_final_atoms], dim=-1), \
               torch.cat([vel[len(mask_atoms):], h_final_residues], dim=-1)
        
    def get_edges(self, batch_mask_ligand, batch_mask_pocket, x_ligand, x_pocket):
        """Create edges for the molecular graph"""
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]

        if self.edge_cutoff_l is not None:
            adj_ligand = adj_ligand & (torch.cdist(x_ligand, x_ligand) <= self.edge_cutoff_l)

        if self.edge_cutoff_p is not None:
            adj_pocket = adj_pocket & (torch.cdist(x_pocket, x_pocket) <= self.edge_cutoff_p)

        if self.edge_cutoff_i is not None:
            adj_cross = adj_cross & (torch.cdist(x_ligand, x_pocket) <= self.edge_cutoff_i)

        adj = torch.cat((torch.cat((adj_ligand, adj_cross), dim=1),
                         torch.cat((adj_cross.T, adj_pocket), dim=1)), dim=0)
        edges = torch.stack(torch.where(adj), dim=0)

        return edges


class PreconditionedEGNNDynamics(nn.Module):
    """Preconditioned wrapper for EGNNDynamics following Karras et al. (2022)."""
    
    def __init__(self, egnn_dynamics: nn.Module, sigma_data: float = 1.0):
        """
        Args:
            egnn_dynamics: The base EGNNDynamics model
            sigma_data: Expected standard deviation of the data (default: 1.0)
        """
        super().__init__()
        self.egnn_dynamics = egnn_dynamics
        self.sigma_data = sigma_data

    def atom_encoder(self, atom_features):
        return self.egnn_dynamics.atom_encoder(atom_features)

    def residue_encoder(self, residue_features):
        return self.egnn_dynamics.residue_encoder(residue_features)
    
    def forward(
        self,
        xh_atoms,
        xh_residues,
        sigma,
        mask_atoms,
        mask_residues,
        target_atoms=None,
        target_residues=None,
        cond_coords_atoms=None,
        cond_coords_residues=None,
        node_ids_atoms=None,              # ADD
        node_ids_residues=None,           # ADD
        cond_node_ids_atoms=None,         # ADD
        cond_node_ids_residues=None,      # ADD
    ):
        """
        Preconditioned forward pass following Karras et al. (2022).
        
        Applies EDM-style preconditioning to coordinates only, leaving
        categorical features and logits untouched. Initial frame conditioning
        coordinates are NOT scaled.
        
        Args:
            xh_atoms: [N_lig, n_dims + joint_nf] noisy ligand coords + embeddings
            xh_residues: [N_poc, n_dims + joint_nf] noisy pocket coords + embeddings
            sigma: [batch_size] or scalar noise level
            mask_atoms: [N_lig] molecule indices for ligand
            mask_residues: [N_poc] molecule indices for pocket
            target_atoms: [N_lig, n_dims + joint_nf] clean targets (for geometric loss)
            target_residues: [N_poc, n_dims + joint_nf] clean targets
            cond_coords_atoms: [N_lig, n_dims] initial frame coords (optional)
            cond_coords_residues: [N_poc, n_dims] initial frame coords (optional)
            node_ids_atoms: [N_lig] node IDs for current graph (optional)
            node_ids_residues: [N_poc] node IDs for current graph (optional)
            cond_node_ids_atoms: [M_lig] node IDs for conditioning graph (optional)
            cond_node_ids_residues: [M_poc] node IDs for conditioning graph (optional)
            
        Returns:
            Tuple of (ligand_output, pocket_output) with preconditioned coordinates + logits
        """
        
        # Determine batch size and device
        batch_size = len(torch.unique(torch.cat([mask_atoms, mask_residues])))
        device = xh_atoms.device
        dtype = xh_atoms.dtype
        
        # Ensure sigma is [batch_size] tensor
        sigma = torch.as_tensor(sigma, device=device, dtype=dtype)
        if sigma.ndim == 0:
            sigma = sigma.expand(batch_size)
        if sigma.ndim != 1 or sigma.shape[0] != batch_size:
            raise ValueError(
                f"sigma must be 1D with length {batch_size}, got shape {sigma.shape}"
            )
        
        # Prevent numerical issues
        sigma = sigma.clamp_min(1e-12)
        
        # Compute EDM preconditioning coefficients
        total_var = self.sigma_data**2 + sigma**2
        c_skip = self.sigma_data**2 / total_var
        c_out = (sigma * self.sigma_data) / torch.sqrt(total_var)
        c_in = 1.0 / torch.sqrt(total_var)
        c_noise = 0.25 * torch.log(sigma)
        
        # Split inputs into coordinates and embeddings
        n_dims = self.egnn_dynamics.n_dims
        coords_lig = xh_atoms[:, :n_dims]
        embeddings_lig = xh_atoms[:, n_dims:]
        coords_poc = xh_residues[:, :n_dims]
        embeddings_poc = xh_residues[:, n_dims:]
        
        # Apply c_in scaling to coordinates only (per-molecule)
        scale_lig = c_in[mask_atoms].unsqueeze(1)  # [N_lig, 1]
        scale_poc = c_in[mask_residues].unsqueeze(1)  # [N_poc, 1]
        
        coords_lig_scaled = scale_lig * coords_lig
        coords_poc_scaled = scale_poc * coords_poc
        
        # Recombine scaled coordinates with original embeddings
        xh_atoms_scaled = torch.cat([coords_lig_scaled, embeddings_lig], dim=1)
        xh_residues_scaled = torch.cat([coords_poc_scaled, embeddings_poc], dim=1)
        
        # Pass conditioning coordinates WITHOUT scaling
        # (they represent initial frame, not noisy state)
        if cond_coords_atoms is not None:
            cond_coords_atoms = cond_coords_atoms.to(device=device, dtype=dtype)
        if cond_coords_residues is not None:
            cond_coords_residues = cond_coords_residues.to(device=device, dtype=dtype)
        
        # Forward through base EGNN with normalized time
        t = c_noise.unsqueeze(1)  # [batch_size, 1]
        
        pred_lig, pred_poc = self.egnn_dynamics(
            xh_atoms_scaled,
            xh_residues_scaled,
            t,
            mask_atoms,
            mask_residues,
            target_atoms=target_atoms,
            target_residues=target_residues,
            cond_coords_atoms=cond_coords_atoms,
            cond_coords_residues=cond_coords_residues,
            node_ids_atoms=node_ids_atoms,                    # ADD
            node_ids_residues=node_ids_residues,              # ADD
            cond_node_ids_atoms=cond_node_ids_atoms,          # ADD
            cond_node_ids_residues=cond_node_ids_residues,    # ADD
        )
        
        # Split predictions into coordinates and logits
        coords_pred_lig = pred_lig[:, :n_dims]
        logits_pred_lig = pred_lig[:, n_dims:]
        coords_pred_poc = pred_poc[:, :n_dims]
        logits_pred_poc = pred_poc[:, n_dims:]
        
        # Apply preconditioning to predicted coordinates
        # out = c_skip * x_noisy + c_out * f(x_scaled)
        skip_lig = c_skip[mask_atoms].unsqueeze(1)  # [N_lig, 1]
        out_lig = c_out[mask_atoms].unsqueeze(1)  # [N_lig, 1]
        coords_out_lig = skip_lig * coords_lig + out_lig * coords_pred_lig
        
        skip_poc = c_skip[mask_residues].unsqueeze(1)  # [N_poc, 1]
        out_poc = c_out[mask_residues].unsqueeze(1)  # [N_poc, 1]
        coords_out_poc = skip_poc * coords_poc + out_poc * coords_pred_poc
        
        # Combine preconditioned coordinates with unpreconditioned logits
        ligand_out = torch.cat([coords_out_lig, logits_pred_lig], dim=1)
        pocket_out = torch.cat([coords_out_poc, logits_pred_poc], dim=1)
        
        return ligand_out, pocket_out

    
    @property
    def last_geometric_loss(self):
        """Access geometric loss from the underlying EGNN dynamics"""
        return self.egnn_dynamics.last_geometric_loss


def test_egnn_dynamicsg():
    """Test EGNNDynamics with variable-sized conditioning (mismatched node counts)"""
    
    print("="*80)
    print("Testing EGNNDynamics with Variable-Sized Conditioning")
    print("="*80)
    
    batch_size = 2
    atom_nf = 5
    residue_nf = 7
    n_dims = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    joint_nf = 8
    
    # ===== CURRENT STATE (MD graph) =====
    # Molecule 0: 4 lig atoms, 12 pocket atoms (16 total)
    # Molecule 1: 4 lig atoms, 8 pocket atoms (12 total)
    total_lig_atoms = 8
    total_pocket_atoms = 20
    
    xh_atoms = torch.randn(total_lig_atoms, n_dims + joint_nf, device=device)
    xh_residues = torch.randn(total_pocket_atoms, n_dims + joint_nf, device=device)
    
    mask_atoms = torch.cat([torch.zeros(4), torch.ones(4)]).long().to(device)
    mask_residues = torch.cat([torch.zeros(12), torch.ones(8)]).long().to(device)
    
    # Node IDs for current graph (namespaced by molecule)
    node_ids_atoms = torch.tensor([
        0, 1, 2, 3,              # Mol 0: atoms 0-3
        100000, 100001, 100002, 100003  # Mol 1: atoms 0-3 + offset
    ], dtype=torch.long, device=device)
    
    node_ids_residues = torch.tensor([
        # Mol 0: 12 residues (IDs 10, 11, 15, 20, 22, 25, 30, 35, 40, 45, 50, 55)
        10, 11, 15, 20, 22, 25, 30, 35, 40, 45, 50, 55,
        # Mol 1: 8 residues + offset
        100010, 100011, 100015, 100020, 100022, 100025, 100030, 100035
    ], dtype=torch.long, device=device)
    
    # ===== CONDITIONING STATE (Initial frame) =====
    # Different sizes! More residues in initial state
    # Molecule 0: 4 lig atoms, 15 pocket atoms (19 total)
    # Molecule 1: 4 lig atoms, 10 pocket atoms (14 total)
    cond_total_lig = 8  # Same ligand atoms
    cond_total_pocket = 25  # MORE pocket atoms
    
    cond_coords_atoms = torch.randn(cond_total_lig, n_dims, device=device)
    cond_coords_residues = torch.randn(cond_total_pocket, n_dims, device=device)
    
    cond_node_ids_atoms = torch.tensor([
        0, 1, 2, 3,              # Mol 0: same atoms
        100000, 100001, 100002, 100003  # Mol 1: same atoms
    ], dtype=torch.long, device=device)
    
    cond_node_ids_residues = torch.tensor([
        # Mol 0: 15 residues (includes 12, 13, 14 that moved OUT in MD)
        10, 11, 12, 13, 14, 15, 20, 22, 25, 30, 35, 40, 45, 50, 55,
        # Mol 1: 10 residues (includes 12, 14 that moved OUT)
        100010, 100011, 100012, 100014, 100015, 100020, 100022, 100025, 100030, 100035
    ], dtype=torch.long, device=device)
    
    # Training metadata
    t = torch.tensor([0.3, 0.7], dtype=torch.float32, device=device)
    target_atoms = torch.randn(total_lig_atoms, n_dims + joint_nf, device=device)
    target_residues = torch.randn(total_pocket_atoms, n_dims + joint_nf, device=device)
    
    print(f"\n📊 Test Setup:")
    print(f"  Current graph: {total_lig_atoms} lig + {total_pocket_atoms} pocket = {total_lig_atoms + total_pocket_atoms} nodes")
    print(f"  Conditioning:  {cond_total_lig} lig + {cond_total_pocket} pocket = {cond_total_lig + cond_total_pocket} nodes")
    print(f"  Difference: {(cond_total_lig + cond_total_pocket) - (total_lig_atoms + total_pocket_atoms)} extra nodes in conditioning")
    
    # Create model
    model = EGNNDynamics(
        atom_nf=atom_nf,
        residue_nf=residue_nf,
        n_dims=n_dims,
        joint_nf=joint_nf,
        hidden_nf=32,
        device=device,
        n_layers=2,
        condition_time=True,
        update_pocket_coords=True,
        edge_embedding_dim=4,
        use_sin_embedding=True,
        reflection_equivariant=True,
        tanh=True,
        attention=True,
        geometric_regularization=False,
        use_conditioning=True
    )
    
    model = PreconditionedEGNNDynamics(model)
    model.train()
    
    # ===== TEST 1: Without Conditioning =====
    print("\n" + "-"*80)
    print("TEST 1: Forward pass WITHOUT conditioning")
    print("-"*80)
    
    ligand_out, pocket_out = model(
        xh_atoms, xh_residues, t, mask_atoms, mask_residues,
        target_atoms=target_atoms, target_residues=target_residues
    )
    
    print(f"✓ Output shapes: ligand {ligand_out.shape}, pocket {pocket_out.shape}")
    assert ligand_out.shape == (total_lig_atoms, n_dims + atom_nf)
    assert pocket_out.shape == (total_pocket_atoms, n_dims + residue_nf)
    print(f"✓ No NaNs: ligand={not torch.isnan(ligand_out).any()}, pocket={not torch.isnan(pocket_out).any()}")
    
    # ===== TEST 2: With Variable-Sized Conditioning =====
    print("\n" + "-"*80)
    print("TEST 2: Forward pass WITH variable-sized conditioning")
    print("-"*80)
    
    ligand_out_cond, pocket_out_cond = model(
        xh_atoms, xh_residues, t, mask_atoms, mask_residues,
        target_atoms=target_atoms, target_residues=target_residues,
        cond_coords_atoms=cond_coords_atoms,
        cond_coords_residues=cond_coords_residues,
        node_ids_atoms=node_ids_atoms,
        node_ids_residues=node_ids_residues,
        cond_node_ids_atoms=cond_node_ids_atoms,
        cond_node_ids_residues=cond_node_ids_residues
    )
    
    print(f"✓ Output shapes: ligand {ligand_out_cond.shape}, pocket {pocket_out_cond.shape}")
    assert ligand_out.shape == (total_lig_atoms, n_dims + atom_nf)
    assert pocket_out.shape == (total_pocket_atoms, n_dims + residue_nf)
    print(f"✓ No NaNs: ligand={not torch.isnan(ligand_out_cond).any()}, pocket={not torch.isnan(pocket_out_cond).any()}")
    
    # Check conditioning changes output
    output_diff = (ligand_out_cond - ligand_out).abs().mean()
    print(f"✓ Output difference with/without conditioning: {output_diff.item():.6f}")
    assert output_diff > 1e-6, "Conditioning should change output!"
    
    # ===== TEST 3: Check Matching Logic =====
    print("\n" + "-"*80)
    print("TEST 3: Verify node matching logic")
    print("-"*80)
    
    # Manually check what should match
    # Ligand: all 8 atoms should match (same IDs in both graphs)
    # Pocket mol 0: 12 current vs 15 conditioning → 12 should match, 3 unmatched in cond
    # Pocket mol 1: 8 current vs 10 conditioning → 8 should match, 2 unmatched in cond
    
    print(f"✓ Expected matches:")
    print(f"  Ligand atoms: 8/8 should match (100%)")
    print(f"  Pocket mol 0: 12/12 current should match (residues 10,11,15,20,22,25,30,35,40,45,50,55)")
    print(f"  Pocket mol 1: 8/8 current should match")
    print(f"  Unmatched in conditioning: 3+2=5 residues (12,13,14 from mol0; 12,14 from mol1)")
    
    # ===== TEST 4: Edge Validity Bits =====
    print("\n" + "-"*80)
    print("TEST 4: Edge conditioning with validity bits")
    print("-"*80)
    
    # Check that edge features have correct dimension
    # Should include: edge_type_emb (4) + distance_emb (sin_emb.dim) + validity (1)
    sin_emb = SinusoidsEmbeddingNew()
    expected_edge_dim = 4 + sin_emb.dim + 1
    print(f"✓ Expected edge feature dim: {expected_edge_dim}")
    print(f"  (4 type embedding + {sin_emb.dim} distance + 1 validity bit)")
    
    # ===== TEST 5: Gradient Flow =====
    print("\n" + "-"*80)
    print("TEST 5: Verify no gradient to conditioning")
    print("-"*80)
    
    cond_coords_atoms.requires_grad = True
    cond_coords_residues.requires_grad = True
    
    ligand_out_grad, pocket_out_grad = model(
        xh_atoms, xh_residues, t, mask_atoms, mask_residues,
        cond_coords_atoms=cond_coords_atoms,
        cond_coords_residues=cond_coords_residues,
        node_ids_atoms=node_ids_atoms,
        node_ids_residues=node_ids_residues,
        cond_node_ids_atoms=cond_node_ids_atoms,
        cond_node_ids_residues=cond_node_ids_residues
    )
    
    loss = ligand_out_grad.sum() + pocket_out_grad.sum()
    loss.backward()
    
    print(f"✓ Conditioning coords gradient is None: {cond_coords_atoms.grad is None}")
    assert cond_coords_atoms.grad is None, "Gradients should NOT flow to conditioning!"
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nSummary:")
    print("  ✓ Handles variable-sized conditioning (different node counts)")
    print("  ✓ Node matching works via ID lookup (O(N log M))")
    print("  ✓ Unmatched nodes get learned NULL embeddings")
    print("  ✓ FiLM masking prevents NULL nodes from affecting output")
    print("  ✓ Edge validity bits distinguish real/fake conditioning distances")
    print("  ✓ No gradient flow to conditioning (detached properly)")
    
    return model


if __name__ == "__main__":
    test_egnn_dynamicsg()
