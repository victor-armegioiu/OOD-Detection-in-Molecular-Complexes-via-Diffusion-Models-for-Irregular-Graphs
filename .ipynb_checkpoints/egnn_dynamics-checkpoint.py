"""
EGNN Dynamics for Molecular Diffusion

Contains the EGNNDynamics class that predicts noise for molecular systems.
This is the core neural network that learns to denoise molecular structures.
"""

import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_mean


def remove_mean_batch(x, indices):
    """Remove center of mass for each molecule in batch"""
    mean = scatter_mean(x, indices, dim=0)
    return x - mean[indices]


class EGNNDynamics(nn.Module):
    def __init__(self, atom_nf, residue_nf,
                 n_dims, joint_nf=16, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics',
                 norm_constant=0, inv_sublayers=2, sin_embedding=False,
                 normalization_factor=100, aggregation_method='sum',
                 update_pocket_coords=True, edge_cutoff_ligand=None,
                 edge_cutoff_pocket=None, edge_cutoff_interaction=None,
                 reflection_equivariant=True, edge_embedding_dim=None):
        super().__init__()
        self.mode = mode
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction
        self.edge_nf = edge_embedding_dim

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

        # Simplified EGNN for this demo (avoiding complex EGNN imports)
        self.egnn = self._create_simple_egnn(
            dynamics_node_nf, self.edge_nf, hidden_nf, device, act_fn, n_layers
        )
        self.node_nf = dynamics_node_nf
        self.update_pocket_coords = update_pocket_coords

        self.device = device
        self.n_dims = n_dims
        self.condition_time = condition_time
        
        # Move model to device
        self.to(device)

    def _create_simple_egnn(self, in_node_nf, in_edge_nf, hidden_nf, device, act_fn, n_layers):
        """Create a simplified EGNN for testing purposes"""
        class SimpleEGNN(nn.Module):
            def __init__(self, in_node_nf, in_edge_nf, hidden_nf, act_fn, n_layers):
                super().__init__()
                self.n_layers = n_layers
                
                # Node update layers
                self.node_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(in_node_nf, hidden_nf),
                        act_fn,
                        nn.Linear(hidden_nf, in_node_nf)
                    ) for _ in range(n_layers)
                ])
                
                # Coordinate update layers
                self.coord_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(in_node_nf * 2 + in_edge_nf, hidden_nf),
                        act_fn,
                        nn.Linear(hidden_nf, 1)
                    ) for _ in range(n_layers)
                ])

            def forward(self, h, x, edges, update_coords_mask=None, batch_mask=None, edge_attr=None):
                row, col = edges
                
                for layer_idx in range(self.n_layers):
                    # Node update
                    h_residual = h
                    h = h + self.node_layers[layer_idx](h)
                    
                    # Coordinate update
                    edge_input = torch.cat([h[row], h[col]], dim=1)
                    if edge_attr is not None:
                        edge_input = torch.cat([edge_input, edge_attr], dim=1)

                    # Coordinate differences
                    coord_diff = x[row] - x[col]
                    weights = self.coord_layers[layer_idx](edge_input)
                    coord_updates = coord_diff * weights

                    # Aggregate coordinate updates using scatter_add
                    x_updates = torch.zeros_like(x)
                    x_updates.scatter_add_(0, row.unsqueeze(1).expand(-1, x.size(1)), coord_updates)
                    
                    x = x + x_updates * 0.1  # Small learning rate for stability

                    # Apply update mask if provided
                    if update_coords_mask is not None:
                        x = x * update_coords_mask + x * (1 - update_coords_mask)

                return h, x

        return SimpleEGNN(in_node_nf, in_edge_nf, hidden_nf, act_fn, n_layers)

    def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues):
        """
        Forward pass of EGNN Dynamics
        
        Args:
            xh_atoms: [total_lig_atoms, n_dims + atom_nf] - ligand coords + features
            xh_residues: [total_pocket_atoms, n_dims + residue_nf] - pocket coords + features  
            t: [batch_size, 1] - normalized timestep values
            mask_atoms: [total_lig_atoms] - molecule indices for ligand atoms
            mask_residues: [total_pocket_atoms] - molecule indices for pocket residues
            
        Returns:
            Tuple of (ligand_output, pocket_output) representing predicted noise
        """
        
        x_atoms = xh_atoms[:, :self.n_dims].clone()
        h_atoms = xh_atoms[:, self.n_dims:].clone()

        x_residues = xh_residues[:, :self.n_dims].clone()
        h_residues = xh_residues[:, self.n_dims:].clone()

        # embed atom features and residue features in a shared space
        h_atoms = self.atom_encoder(h_atoms)
        h_residues = self.residue_encoder(h_residues)

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
            # in case of unconditional joint distribution, include this as in
            # the original code
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


def test_egnn_dynamics():
    """Test the EGNNDynamics implementation"""
    
    print("Testing EGNNDynamics...")
    
    # Configuration
    batch_size = 2
    atom_nf = 5
    residue_nf = 7
    n_dims = 3
    device = 'cuda'
    
    # Create test data
    total_lig_atoms = 8
    total_pocket_atoms = 20
    
    xh_atoms = torch.randn(total_lig_atoms, n_dims + atom_nf, device=device)
    xh_residues = torch.randn(total_pocket_atoms, n_dims + residue_nf, device=device)
    
    mask_atoms = torch.cat([torch.zeros(4), torch.ones(4)]).long().to(device)
    mask_residues = torch.cat([torch.zeros(12), torch.ones(8)]).long().to(device)
    
    t = torch.tensor([[0.3], [0.7]], dtype=torch.float32, device=device)
    
    # Create model
    model = EGNNDynamics(
        atom_nf=atom_nf,
        residue_nf=residue_nf,
        n_dims=n_dims,
        joint_nf=8,
        hidden_nf=32,
        device='cuda',
        n_layers=2,
        condition_time=True,
        update_pocket_coords=True,
        edge_embedding_dim=4
    )
    
    # Forward pass
    ligand_out, pocket_out = model(xh_atoms, xh_residues, t, mask_atoms, mask_residues)
    
    print(f"Input shapes: ligand {xh_atoms.shape}, pocket {xh_residues.shape}")
    print(f"Output shapes: ligand {ligand_out.shape}, pocket {pocket_out.shape}")
    print(f"✅ EGNNDynamics test passed!")
    
    return model


if __name__ == "__main__":
    test_egnn_dynamics()