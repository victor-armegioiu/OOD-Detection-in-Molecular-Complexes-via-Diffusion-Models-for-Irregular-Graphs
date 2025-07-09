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
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
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
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        edge_feat_nf = edge_feat_nf + in_edge_nf

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
                 norm_constant=0, inv_sublayers=2, sin_embedding=False,
                 normalization_factor=100, aggregation_method='sum',
                 update_pocket_coords=True, edge_cutoff_ligand=None,
                 edge_cutoff_pocket=None, edge_cutoff_interaction=None,
                 reflection_equivariant=True, edge_embedding_dim=None,
                 geometric_regularization=False, geom_loss_weight=0.1):
        super().__init__()
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction
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

        self.egnn = EGNN(
            in_node_nf=dynamics_node_nf, in_edge_nf=self.edge_nf,
            hidden_nf=hidden_nf, device=device, act_fn=act_fn,
            n_layers=n_layers, attention=attention, tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
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

        self.device = device
        self.n_dims = n_dims
        self.condition_time = condition_time
        self.to(device)

    def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues, 
                target_atoms=None, target_residues=None):
        """
        Forward pass of EGNN Dynamics
        
        Args:
            xh_atoms: [total_lig_atoms, n_dims + joint_nf] - ligand coords + embedding_size
            xh_residues: [total_pocket_atoms, n_dims + residue_nf] - pocket coords + embedding_size  
            t: [batch_size, 1] - normalized timestep values
            mask_atoms: [total_lig_atoms] - molecule indices for ligand atoms
            mask_residues: [total_pocket_atoms] - molecule indices for pocket residues
            target_atoms: [total_lig_atoms, n_dims + joint_nf] - target ligand coords (for geometric loss)
            target_residues: [total_pocket_atoms, n_dims + residue_nf] - target pocket coords
            
        Returns:
            Tuple of (ligand_output, pocket_output) representing predicted noise
        """

        x_atoms = xh_atoms[:, :self.n_dims].clone()
        h_atoms = xh_atoms[:, self.n_dims:].clone()

        x_residues = xh_residues[:, :self.n_dims].clone()
        h_residues = xh_residues[:, self.n_dims:].clone()

        # embed atom features and residue features in a shared space
        # h_atoms = self.atom_encoder(h_atoms)
        # h_residues = self.residue_encoder(h_residues)

        # combine the two node types
        x = torch.cat((x_atoms, x_residues), dim=0)
        h = torch.cat((h_atoms, h_residues), dim=0)
        mask = torch.cat([mask_atoms, mask_residues])

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t[mask]
            h = torch.cat([h, h_time], dim=1)

        # get edges of a complete graph
        edges = self.get_edges(mask_atoms, mask_residues, x_atoms, x_residues)
        assert torch.all(mask[edges[0]] == mask[edges[1]])

        # Get edge types
        if self.edge_nf > 0:
            # 0: ligand-pocket, 1: ligand-ligand, 2: pocket-pocket
            edge_types = torch.zeros(edges.size(1), dtype=int, device=edges.device)
            edge_types[(edges[0] < len(mask_atoms)) & (edges[1] < len(mask_atoms))] = 1
            edge_types[(edges[0] >= len(mask_atoms)) & (edges[1] >= len(mask_atoms))] = 2

            # Learnable embedding
            edge_types = self.edge_embedding(edge_types)
        else:
            edge_types = None

        update_coords_mask = None if self.update_pocket_coords \
            else torch.cat((torch.ones_like(mask_atoms),
                            torch.zeros_like(mask_residues))).unsqueeze(1)
        h_final, x_final = self.egnn(h, x, edges,
                                     update_coords_mask=update_coords_mask,
                                     batch_mask=mask, edge_attr=edge_types)
        vel = (x_final - x)

        # Geometric regularization loss (computed on predicted coordinates)
        if self.geometric_regularization and target_atoms is not None and self.training:
            # Extract predicted and target coordinates for ligands only
            pred_ligand_coords = x_final[:len(mask_atoms)]
            target_ligand_coords = target_atoms[:, :self.n_dims]
            
            # Compute geometric loss
            geom_loss = self.geometric_regularizer.comprehensive_geometric_loss(
                pred_ligand_coords, target_ligand_coords, mask_atoms
            )
            self.last_geometric_loss = geom_loss * self.geom_loss_weight
        else:
            self.last_geometric_loss = torch.tensor(0.0, device=self.device)

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        # decode atom and residue features
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
        
    def forward(self, xh_atoms, xh_residues, sigma, mask_atoms, mask_residues,
                target_atoms=None, target_residues=None):
        """
        Preconditioned forward pass with separate handling for coordinates and categorical features.
        
        Args:
            xh_atoms: [total_lig_atoms, n_dims + joint_nf] - ligand coords + embeddings
            xh_residues: [total_pocket_atoms, n_dims + joint_nf] - pocket coords + embeddings  
            sigma: [batch_size] or scalar - noise level (not normalized time!)
            mask_atoms: [total_lig_atoms] - molecule indices for ligand atoms
            mask_residues: [total_pocket_atoms] - molecule indices for pocket residues
            target_atoms: [total_lig_atoms, n_dims + joint_nf] - target ligand coords (for geometric loss)
            target_residues: [total_pocket_atoms, n_dims + residue_nf] - target pocket coords
            
        Returns:
            Tuple of (ligand_output, pocket_output) with preconditioned coordinates + logits
        """
        # Ensure sigma is 1D tensor with batch dimension
        batch_size = len(torch.unique(torch.cat([mask_atoms, mask_residues])))
        if sigma.dim() < 1:
            sigma = sigma.expand(batch_size)
    
        if sigma.dim() != 1 or batch_size != sigma.shape[0]:
            print(sigma.shape, batch_size)

            raise ValueError(
                "sigma must be 1D and have the same leading (batch) dim as x"
                f" ({batch_size})"
            )
            
        # Compute preconditioning coefficients
        total_var = self.sigma_data**2 + sigma**2
        c_skip = self.sigma_data**2 / total_var
        c_out = sigma * self.sigma_data / torch.sqrt(total_var)
        c_in = 1 / torch.sqrt(total_var)
        c_noise = 0.25 * torch.log(sigma)
        
        # Split input into coordinates and embeddings
        coords_lig = xh_atoms[:, :self.egnn_dynamics.n_dims]  # [N_lig, 3]
        embeddings_lig = xh_atoms[:, self.egnn_dynamics.n_dims:]  # [N_lig, joint_nf]
        coords_pocket = xh_residues[:, :self.egnn_dynamics.n_dims]  # [N_pocket, 3]
        embeddings_pocket = xh_residues[:, self.egnn_dynamics.n_dims:]  # [N_pocket, joint_nf]
        
        # Apply c_in scaling to coordinates only (per molecule)
        coords_lig_scaled = coords_lig.clone()
        coords_pocket_scaled = coords_pocket.clone()
        
        for i, batch_idx in enumerate(torch.unique(mask_atoms)):
            atom_mask = mask_atoms == batch_idx
            coords_lig_scaled[atom_mask] = c_in[i] * coords_lig[atom_mask]
            
        for i, batch_idx in enumerate(torch.unique(mask_residues)):
            residue_mask = mask_residues == batch_idx  
            coords_pocket_scaled[residue_mask] = c_in[i] * coords_pocket[residue_mask]
        
        # Recombine scaled coordinates with original embeddings
        xh_atoms_scaled = torch.cat([coords_lig_scaled, embeddings_lig], dim=1)
        xh_residues_scaled = torch.cat([coords_pocket_scaled, embeddings_pocket], dim=1)
        
        # Convert sigma to normalized time for the base model
        t = c_noise.unsqueeze(1)  # [batch_size, 1]
        
        # Forward through base EGNN (pass through target data for geometric loss)
        f_ligand, f_pocket = self.egnn_dynamics(
            xh_atoms_scaled, xh_residues_scaled, t, mask_atoms, mask_residues,
            target_atoms=target_atoms, target_residues=target_residues
        )
        
        # Split EGNN output into coordinates and logits
        coords_pred_lig = f_ligand[:, :self.egnn_dynamics.n_dims]  # [N_lig, 3]
        logits_pred_lig = f_ligand[:, self.egnn_dynamics.n_dims:]  # [N_lig, atom_nf]
        coords_pred_pocket = f_pocket[:, :self.egnn_dynamics.n_dims]  # [N_pocket, 3]
        logits_pred_pocket = f_pocket[:, self.egnn_dynamics.n_dims:]  # [N_pocket, residue_nf]
        
        # Apply preconditioning to coordinates only (skip connection + output scaling)
        coords_out_lig = coords_lig.clone()
        coords_out_pocket = coords_pocket.clone()
        
        for i, batch_idx in enumerate(torch.unique(mask_atoms)):
            atom_mask = mask_atoms == batch_idx
            coords_out_lig[atom_mask] = c_skip[i] * coords_lig[atom_mask] + c_out[i] * coords_pred_lig[atom_mask]
            
        for i, batch_idx in enumerate(torch.unique(mask_residues)):
            residue_mask = mask_residues == batch_idx
            coords_out_pocket[residue_mask] = c_skip[i] * coords_pocket[residue_mask] + c_out[i] * coords_pred_pocket[residue_mask]
        
        # Combine preconditioned coordinates with raw logits (no preconditioning for classification)
        ligand_out = torch.cat([coords_out_lig, logits_pred_lig], dim=1)
        pocket_out = torch.cat([coords_out_pocket, logits_pred_pocket], dim=1)
        
        return ligand_out, pocket_out

    @property
    def last_geometric_loss(self):
        """Access geometric loss from the underlying EGNN dynamics"""
        return self.egnn_dynamics.last_geometric_loss


def test_egnn_dynamics():
    """Test the enhanced EGNNDynamics implementation"""
    
    print("Testing Enhanced EGNNDynamics with Geometric Regularization...")
    
    # Configuration
    batch_size = 2
    atom_nf = 5
    residue_nf = 7
    n_dims = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    joint_nf = 8
    
    # Create test data
    total_lig_atoms = 8
    total_pocket_atoms = 20
    
    xh_atoms = torch.randn(total_lig_atoms, n_dims + joint_nf, device=device)
    xh_residues = torch.randn(total_pocket_atoms, n_dims + joint_nf, device=device)
    
    # Create target data (same structure as inputs for testing)
    target_atoms = torch.randn(total_lig_atoms, n_dims + joint_nf, device=device)
    target_residues = torch.randn(total_pocket_atoms, n_dims + joint_nf, device=device)
    
    mask_atoms = torch.cat([torch.zeros(4), torch.ones(4)]).long().to(device)
    mask_residues = torch.cat([torch.zeros(12), torch.ones(8)]).long().to(device)
    
    t = torch.tensor([0.3, 0.7], dtype=torch.float32, device=device)
    print(f"Time tensor shape: {t.shape}")
    
    # Create model with geometric regularization
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
        sin_embedding=True,
        reflection_equivariant=True,
        tanh=True,
        attention=True,
        geometric_regularization=True,
        geom_loss_weight=0.1
    )

    model = PreconditionedEGNNDynamics(model)
    
    # Forward pass with geometric regularization
    model.train()  # Set to training mode to enable geometric loss
    ligand_out, pocket_out = model(
        xh_atoms, xh_residues, t, mask_atoms, mask_residues,
        target_atoms=target_atoms, target_residues=target_residues
    )
    
    print(f"Input shapes: ligand {xh_atoms.shape}, pocket {xh_residues.shape}")
    print(f"Output shapes: ligand {ligand_out.shape}, pocket {pocket_out.shape}")
    print(f"Geometric loss: {model.last_geometric_loss.item():.6f}")
    print(f"✅ Enhanced EGNNDynamics test passed!")
    
    # Test without targets (should have zero geometric loss)
    model.eval()
    ligand_out_eval, pocket_out_eval = model(xh_atoms, xh_residues, t, mask_atoms, mask_residues)
    print(f"Geometric loss (eval mode): {model.last_geometric_loss.item():.6f}")
    
    return model


if __name__ == "__main__":
    test_egnn_dynamics()
