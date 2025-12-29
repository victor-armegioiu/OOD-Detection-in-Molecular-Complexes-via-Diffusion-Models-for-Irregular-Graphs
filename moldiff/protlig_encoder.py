"""
Protein-Ligand Graph Encoder

This script processes protein PDB files and their corresponding ligand files to create
graph representations suitable for machine learning applications. It supports multiple
file formats and can handle both single files and directories with subdirectories.

Features:
- Recursively processes protein PDB files from directories and subdirectories
- Supports multiple ligand formats: SDF, MOL2, and PDB
- Creates graph representations at atom-level or residue-level granularity
- Handles ligand bond detection and inclusion
- Flexible output organization

File Organization:
- Protein files: Must end with '.pdb' (excluding '_lig.pdb' files)
- Ligand files: Can be .mol2, .sdf, or _lig.pdb
- Ligand discovery: First searches same directory as protein, then ligand_source if specified

Output Behavior:
- With --output_dir: All graphs saved directly in the specified directory
- Without --output_dir: Graphs saved next to their respective protein files

Usage Examples:
    # Process a single protein-ligand pair
    python protlig_encoder.py --protein_source protein.pdb --ligand_source ligand.sdf

    # Process a directory with subdirectories, save all graphs to output_dir
    python protlig_encoder.py --protein_source /path/to/proteins --output_dir /path/to/graphs

    # Process with global ligand for all proteins
    python protlig_encoder.py --protein_source /path/to/proteins --ligand_source /path/to/global_ligand.sdf

    # Process with custom granularity and bond detection
    python protlig_encoder.py --protein_source /path/to/proteins --granularity atom-level --detect_lig_bonds_by_distance true

Directory Structure Example:
    proteins/
    ├── subdir1/
    │   ├── protein1.pdb
    │   └── protein1_lig.pdb
    ├── subdir2/
    │   ├── protein2.pdb
    │   └── protein2.sdf
    └── protein3.pdb

Arguments:
    --protein_source: Path to protein file or directory (required)
    --ligand_source: Path to ligand file or directory (optional)
    --output_dir: Directory to save all graphs (optional)
    --granularity: Graph granularity: "atom-level" or "residue-level-fully-connected"
    --include_lig_bonds: Include ligand bonds in graph (default: true)
    --detect_lig_bonds_by_distance: Detect bonds by distance if not in file (default: false)

Output:
    - PyTorch Geometric Data objects saved as .pt files
    - Each graph contains: node features, edge indices, edge attributes, positions, masks
    - Progress logging and summary statistics
"""

import os
import gemmi
import torch
import argparse
from torch_geometric.utils import to_undirected, add_self_loops
from torch_geometric.data import Data, Batch
from typing import List, Dict, Any, Optional, NamedTuple, Tuple
import traceback
import numpy as np
from moldiff.constants import (
    bond_decoder, 
    atom_decoder, 
    aa_decoder3, 
    metals, 
    protein_letters_1to3, 
    protein_letters_3to1_extended, 
    aa_atom_index, 
    aa_atom_mask, 
    aa_nerf_indices, 
    aa_chi_indices, 
    aa_chi_anchor_atom
    )                 

from moldiff.parsing import Protein, Ligand
from moldiff.nerf import ic_to_coords, get_nerf_params_wrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Protein-Ligand Graph Encoder")
    parser.add_argument("--protein_source", required=True, help="Protein file or directory with input PDB files")
    parser.add_argument("--ligand_source", required=False, help="Ligand file or directory of ligand files")
    parser.add_argument("--output_dir", required=False, help="Optional output directory for saving graphs")
    parser.add_argument("--granularity", choices=["atom-level", "residue-level-fully-connected"], default="residue-level-fully-connected")
    parser.add_argument("--include_lig_bonds", type=lambda x: x.lower() == 'true', default=True, help="Include ligand bonds in the graph")
    parser.add_argument("--detect_lig_bonds_by_distance", type=lambda x: x.lower() == 'true', default=False, help="Detect ligand bonds by distance")
    args = parser.parse_args()
    return args


def find_protein_files_recursive(directory: str) -> List[str]:
    """
    Recursively find all protein files in a directory and its subdirectories.
    Excludes ligand files (ending with '_lig.pdb').
    """
    protein_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdb') and not file.endswith('_lig.pdb'):
                protein_files.append(os.path.join(root, file))
    return sorted(protein_files)


def one_hot_encoding(x, allowable_set, padding_length=0, padding_position=1):
    """
    One-hot encode a value from an allowable set. If the value is not in the allowable set, return None.
    
    Args:
        x: Value to encode
        allowable_set: Set of allowable values
        padding_length: Length of padding
        padding_position: Position of padding (0 for start, 1 for end)
    
    Returns:
        One-hot encoded value
    """
    if x not in allowable_set:
        return None
    if padding_length > 0 and padding_position not in [0, 1]:
        raise ValueError("padding_position must be 0 (start) or 1 (end)")
    if padding_position == 1: 
        return list(map(lambda s: x == s, allowable_set)) + [False] * padding_length
    if padding_position == 0: 
        return [False] * padding_length + list(map(lambda s: x == s, allowable_set))


def padded_concat(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Concatenate two arrays vertically with appropriate zero padding.
    First array is padded at the end, second array is padded at the beginning.
    
    Args:
        arr1: First array to concatenate, will be padded at the end
        arr2: Second array to concatenate, will be padded at the beginning
    
    Returns:
        Concatenated array with shape (arr1.shape[0] + arr2.shape[0], arr1.shape[1] + arr2.shape[1])
    """
    n1, m1 = arr1.shape
    n2, m2 = arr2.shape
    
    # Create padded versions of both arrays
    arr1_padded = np.pad(arr1, ((0, 0), (0, m2)), mode='constant', constant_values=0)
    arr2_padded = np.pad(arr2, ((0, 0), (m1, 0)), mode='constant', constant_values=0)
    
    # Concatenate vertically
    return np.concatenate([arr1_padded, arr2_padded], axis=0)



class ProteinGraphEncoder:
    def __init__(self, granularity: str = "atom-level"):
        self.granularity = granularity

    def _ligand_graph(self, ligand: Ligand, remove_hydrogens: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process ligand into graph components with node and edge features.
        Returns:
            Tuple containing:
            - coords: (N, 3) array of atom coordinates
            - node_features: (N, F) array of atom features
            - edge_index: (2, E) array of edge indices
            - edge_attr: (E, F) array of edge features including:
                - one-hot encoded bond type (4 features: single, double, triple, aromatic)
                - bond length
                - spatial features (optional)
        """
        if remove_hydrogens:
            valid_atoms = [i for i, atom in enumerate(ligand.atoms) if atom.atom_type != 'H']
            atom_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_atoms)}
        else:
            valid_atoms = range(len(ligand.atoms))
            atom_idx_map = {idx: idx for idx in valid_atoms}

        # Process nodes
        ligand_coords = []
        ligand_features = []
        for idx in valid_atoms:
            atom = ligand.atoms[idx]
            ligand_coords.append([atom.x, atom.y, atom.z])
            ligand_features.append(one_hot_encoding(atom.atom_type, atom_decoder))

        # Process edges
        edge_list = []
        edge_features = []
        bond_types = {'SINGLE': [1,0,0,0], 'DOUBLE': [0,1,0,0], 'TRIPLE': [0,0,1,0], 'AROMATIC': [0,0,0,1]}
        
        for bond in ligand.bonds:
            
            # Skip if either atom was removed (hydrogen)
            if bond.atom1_idx not in atom_idx_map or bond.atom2_idx not in atom_idx_map: 
                continue
                
            new_idx1 = atom_idx_map[bond.atom1_idx]
            new_idx2 = atom_idx_map[bond.atom2_idx]
            edge_list.extend([[new_idx1, new_idx2], [new_idx2, new_idx1]])
            
            # Calculate bond length
            atom1 = ligand.atoms[bond.atom1_idx]
            atom2 = ligand.atoms[bond.atom2_idx]
            bond_length = np.sqrt((atom1.x - atom2.x)**2 + (atom1.y - atom2.y)**2 + (atom1.z - atom2.z)**2)
            
            # Edge features: [bond_type_onehot, bond_length]
            edge_feat = bond_types[bond.bond_type] + [bond_length]
            edge_features.extend([edge_feat, edge_feat])
        
        if edge_list:
            edge_index = np.array(edge_list).T  # Shape: (2, num_edges)
            edge_attr = np.array(edge_features)  # Shape: (num_edges, num_features)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 5), dtype=np.float32)  # 4 bond types + 1 length
        
        return np.array(ligand_coords), np.array(ligand_features), edge_index, edge_attr

    def _get_interacting_residue_indices(self, residues, ligand_coords, interaction_distance: float) -> list:
        """Return indices of protein residues that have *any atom* within interaction_distance of any ligand atom.
        This logic is directly reusable for atom-level encoding (swap residues -> atoms).
        """
        interacting = []
        for i, residue in enumerate(residues):
            if not residue.get('atoms'):  # skip empty
                continue
            res_coords = np.array([[a['pos_x'], a['pos_y'], a['pos_z']] for a in residue['atoms']], dtype=float)
            # pairwise distances to ligand atoms
            diff = res_coords[:, None, :] - ligand_coords[None, :, :]
            d2 = (diff * diff).sum(-1)
            if (d2 < (interaction_distance ** 2)).any():
                interacting.append(i)
        return interacting

    def _standardize_res_name(self, res_name: str):
        """Normalize residue names; collapse metals to 'METAL' like current pipeline does."""
        # metals handling
        if res_name.strip('0123456789').upper() in [m.upper() for m in metals]:
            return 'METAL'
        # normalize to 3-letter uppercase if possible
        try:
            std3 = protein_letters_1to3[protein_letters_3to1_extended[res_name]]
            return std3.upper()
        except Exception:
            return res_name.upper()

    def _residue_rep_coord(self, residue):
        """Pick a single representative coordinate for the residue.
        Current behavior: use CA if available; else use first atom.
        For METAL (pseudo-residue), use first atom.
        This encapsulation makes it trivial to swap in centroid/CB later.
        """
        atoms = residue.get('atoms', [])
        if not atoms:
            return None
        # CA preference
        for a in atoms:
            if a.get('name') == 'CA':
                return [a['pos_x'], a['pos_y'], a['pos_z']]
        # default: first atom
        a0 = atoms[0]
        return [a0['pos_x'], a0['pos_y'], a0['pos_z']]

    def _stack_nodes_and_masks(self, ligand_coords, ligand_feats, prot_coords, prot_feats, ca_mask = None):
        """Concatenate ligand and protein nodes and build masks. Returns (x, pos, lig_mask, prot_mask)."""
        # x: padded concat (ligand feat dim may differ from residue feat dim)
        x = padded_concat(ligand_feats, prot_feats)
        pos = np.concatenate([ligand_coords, prot_coords], axis=0)
        lig_mask = np.concatenate([np.ones(len(ligand_coords), dtype=bool), np.zeros(len(prot_coords), dtype=bool)])
        prot_mask = ~lig_mask
        if ca_mask: 
            ca_mask = np.concatenate([np.zeros(len(ligand_coords), dtype=bool), ca_mask])
            return x, pos, lig_mask, prot_mask, ca_mask
        else:
            return x, pos, lig_mask, prot_mask

    def _fully_connected_edges_with_dist(self, pos):
        """Build fully connected directed edges and encode Euclidean distance per edge."""
        n = len(pos)
        # all pairs including i->i (keep behavior identical to original)
        src = np.repeat(np.arange(n), n)
        dst = np.tile(np.arange(n), n)
        edge_index = np.vstack([src, dst])
        # distances
        diff = pos[src] - pos[dst]
        d = np.sqrt((diff * diff).sum(-1))[:, None]
        return edge_index, d

    def _build_edges_with_protlig_and_cutoff(
            self, 
            pos: torch.Tensor,
            prot_mask: torch.Tensor,
            lig_mask: torch.Tensor,
            cutoff: float = 6.0
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges for protein-ligand graphs:
        1. Add all prot–lig edges (both directions).
        2. Add cutoff-based edges between all nodes (prot-prot, lig-lig, prot-lig)
            if they are not already included.
        
        Args:
            pos (Tensor): (N,3) coordinates
            prot_mask (Tensor): (N,) bool mask for protein nodes
            lig_mask (Tensor): (N,) bool mask for ligand nodes
            cutoff (float): distance cutoff in Å
        
        Returns:
            edge_index (LongTensor): shape (2,E), edges
            edge_attr  (FloatTensor): shape (E,1), distances
        """
        pos_np = pos
        N = len(pos_np)

        prot_idx = np.where(prot_mask)[0]
        lig_idx  = np.where(lig_mask)[0]

        edge_list = []
        edge_attr = []

        # --- 1. Add all prot-lig edges ---
        for p in prot_idx:
            for l in lig_idx:
                d = np.linalg.norm(pos_np[p] - pos_np[l])
                edge_list.extend([[p, l], [l, p]])
                edge_attr.extend([[d], [d]])

        # store set of already added edges for fast lookup
        existing_edges = set((u, v) for u, v in edge_list)

        # --- 2. Add cutoff edges ---
        diff = pos_np[:, None, :] - pos_np[None, :, :]
        d2 = (diff * diff).sum(-1)
        mask = (d2 > 0) & (d2 <= cutoff * cutoff)
        src, dst = np.nonzero(mask)

        for s, t in zip(src, dst):
            if (s, t) not in existing_edges:
                d = np.sqrt(d2[s, t])
                edge_list.append([s, t])
                edge_attr.append([d])
                existing_edges.add((s, t))

        # --- Convert to torch ---
        edge_index = torch.LongTensor(edge_list).T
        edge_attr  = torch.FloatTensor(edge_attr)

        return edge_index, edge_attr

 
    def encode(self, protein: Protein, ligand: Optional[Ligand] = None) -> Dict[str, Any]:
        if self.granularity == "atom-level":
            return self._encode_atom_level(protein, ligand)
        elif self.granularity == "residue-level-fully-connected":
            return self._encode_residue_level_fully_connected(protein, ligand)
        else:
            raise ValueError(f"Unknown featurization type: {self.granularity}")


    def _encode_residue_level_fully_connected(self, 
                              protein: Protein, 
                              ligand: Optional[Ligand], 
                              interaction_distance: float = 5.0,
                              remove_hydrogens: bool = True) -> Data:

        if not ligand:
            print("No ligand provided to _encode_residue_level_fully_connected, returning None")
            return None

        # 1) ligand graph (reused for atom-level)
        ligand_coords, ligand_features, lig_edge_index, lig_edge_attr = self._ligand_graph(ligand, remove_hydrogens)  

        # 2) identify interacting residues (reusable for atom-level by swapping residues->atoms)
        residues = protein.get_residues()
        interacting_residues = self._get_interacting_residue_indices(residues, ligand_coords, interaction_distance)
        print(f"Found {len(interacting_residues)} residues within {interaction_distance}Å of ligand")

        # 3) featurize residues + choose representative coordinates
        protein_coords = []
        protein_features = []

        for idx in interacting_residues:
            res = residues[idx]
            res_name = self._standardize_res_name(res['name'])
            res_feat = one_hot_encoding(res_name, aa_decoder3)
            if res_feat is None:
                print(f"Warning: {res['name']} not found in aa_decoder3 or standard mappings, returning None")
                return None
            protein_features.append(res_feat)
            
            # Representative coordinate
            rep = self._residue_rep_coord(res)
            if rep is None:
                print(f"Warning: residue at index {idx} has no atoms, skipping")
                continue
            protein_coords.append(rep)

        protein_coords = np.array(protein_coords, dtype=float)
        protein_features = np.array(protein_features, dtype=float)

        # 4) concatenate nodes and build masks
        x_np, pos_np, lig_mask_np, prot_mask_np = self._stack_nodes_and_masks(
            np.array(ligand_coords, dtype=float),
            np.array(ligand_features, dtype=float),
            protein_coords,
            protein_features
        )

        # 5) fully connected edges with distances
        edge_index_np, edge_attr_np = self._fully_connected_edges_with_dist(pos_np)

        # 6) torchify
        x = torch.FloatTensor(x_np)
        edge_index = torch.LongTensor(edge_index_np)
        edge_attr = torch.FloatTensor(edge_attr_np)
        pos = torch.FloatTensor(pos_np)
        lig_mask = torch.BoolTensor(lig_mask_np)
        prot_mask = torch.BoolTensor(prot_mask_np)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, lig_mask=lig_mask, prot_mask=prot_mask)


    def _encode_atom_level(self, 
                           protein: Protein, 
                           ligand: Optional[Ligand],
                           interaction_distance: float = 5.0,
                           remove_hydrogens: bool = True) -> Data:
        
        """Encode *side-chain* atoms at full atom level and ligand atoms, using a cutoff graph.
        Returns the same tuple as residue-level: (x, edge_index, edge_attr, pos, lig_mask, prot_mask)
        where edge_attr is a single distance feature.
        """


        if ligand is None:
            raise ValueError("No ligand provided to _encode_atom_level, returning None")

        # 1) Ligand nodes (coords + features); we will only reuse coords/features here.
        lig_coords, lig_node_feats, _, _ = self._ligand_graph(ligand, remove_hydrogens=remove_hydrogens)

        # 2) Determine interacting residues to limit the protein search space.
        residues = protein.get_residues()
        interacting_res_idxs = self._get_interacting_residue_indices(
            residues,
            np.array(lig_coords, dtype=float),
            interaction_distance=interaction_distance
        )
        interacting_res = [residues[i] for i in interacting_res_idxs]


        # 4) Protein atom coords + features (one-hot by element using atom_decoder)
        
        # get_nerf_params
        get_nerf_params = get_nerf_params_wrapper("dict")
        prot_coords = []
        prot_feats = []
        ca_mask = []
        unique_interacting_residue_identifiers = []
        skipped = 0
        for res in interacting_res:
            res_name = res.get('name', 'UNK')
            for atm in res.get('atoms', []):
                elem = atm.get('element')
                if remove_hydrogens and elem == 'H':
                    continue
                if elem not in atom_decoder:
                    skipped += 1
                    continue
                
                # NOTE: here would be the place to extend feature space, e.g. bond angles hardcoded to 90 here
                chis = list(get_nerf_params(res)["chi"])
                
                prot_feats.append([elem == s for s in atom_decoder] + [res_name == s for s in aa_decoder3])# + chis)
                prot_coords.append([atm['pos_x'], atm['pos_y'], atm['pos_z']])
                ca_mask.append(atm.get("name") == "CA")
            unique_interacting_residue_identifiers.append(res) # chain, num, icode to uniquely identify

        if skipped:
            print(f"Skipped {skipped} side-chain atoms with unknown element symbol.")

        prot_coords = np.array(prot_coords, dtype=float) if prot_coords else np.zeros((0,3), dtype=float)
        prot_feats = np.array(prot_feats, dtype=float) if prot_feats else np.zeros((0, len(atom_decoder)), dtype=float)


        # 5) Stack nodes and build masks
        x_np, pos_np, lig_mask_np, prot_mask_np, ca_mask_np = self._stack_nodes_and_masks(
            np.array(lig_coords, dtype=float),
            np.array(lig_node_feats, dtype=float),
            prot_coords,
            prot_feats, 
            ca_mask
        )

        # 6) Build cutoff graph (across ligand + protein nodes) with 6.0Å default TODO probably a hyperparameter
        edge_index_np, edge_attr_np = self._build_edges_with_protlig_and_cutoff(pos_np, lig_mask_np, prot_mask_np, cutoff=6.0)


        # 7) Torch tensors
        x = torch.FloatTensor(x_np)
        edge_index = torch.LongTensor(edge_index_np)
        edge_attr = torch.FloatTensor(edge_attr_np)
        pos = torch.FloatTensor(pos_np)
        lig_mask = torch.BoolTensor(lig_mask_np)
        prot_mask = torch.BoolTensor(prot_mask_np)
        ca_mask = torch.BoolTensor(ca_mask_np)

        return Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            pos=pos, 
            lig_mask=lig_mask, 
            prot_mask=prot_mask, 
            ca_mask=ca_mask, 
            interacting_res_ids = unique_interacting_residue_identifiers
            )





def find_protein_ligand_pairs(protein_source: str, ligand_source: Optional[str] = None) -> List[Tuple[str, str]]:

    """
    Find all protein-ligand pairs in the given protein_source and ligand_source.
    - If protein_source is a file, it is assumed to be a global protein and is combined with all ligands found in 
            1) ligand_source if given
            2) the same directory as the protein_source
    - If protein_source is a directory, all protein files in the directory and subdirectories are used. Each is combined with a ligand with the same name
      and one of the extensions .mol2, .sdf, or _lig.pdb found in
            1) ligand_source if given
            2) the same directory as the protein_source

    Args:
        protein_source: Path to protein file or directory with input protein structure files
        ligand_source: Path to ligand file or directory of ligand files

    Returns:
        List[Tuple[str, str]]: List of protein-ligand pairs
    """

    def find_ligands_in_dir(dir_path, pattern=None):
        if pattern:
            for ext in ['.mol2', '.sdf', '_lig.pdb']:
                ligand_path = os.path.join(dir_path, pattern + ext)
                if os.path.exists(ligand_path):
                    return ligand_path
            return None
        else: 
            ligands = []
            for ext in ['.mol2', '.sdf', '_lig.pdb']:
                ligands.extend([f.path for f in os.scandir(dir_path) if f.name.endswith(ext)])
            return ligands

    protein_ligand_pairs = []

    # Protein Source is a file: GLOBAL PROTEIN: 
    # Find all ligands 1) in the ligand_source directory or 2) in the same directory as the protein file
    # and combine them with the global protein
    if os.path.isfile(protein_source):
        print(f"Using global protein from {protein_source}")
        global_protein_path = protein_source

        if ligand_source and os.path.isdir(ligand_source):
            ligand_paths = find_ligands_in_dir(ligand_source)
        else:
            ligand_paths = find_ligands_in_dir(os.path.dirname(protein_source))
        for ligand_path in ligand_paths:
            protein_ligand_pairs.append((global_protein_path, ligand_path))
    
    # Protein Source is a directory: NOT GLOBAL PROTEIN: 
    # For each protein, find the corresponding ligand with the same name
    elif os.path.isdir(protein_source):
        protein_paths = find_protein_files_recursive(protein_source)
        print(f"Found {len(protein_paths)} protein files in {protein_source} and subdirectories")
        for protein_path in protein_paths:
            protein_file = os.path.basename(protein_path)
            base = os.path.splitext(protein_file)[0]            
            ligand_found = False

            # --- FIND THE LIGAND for this protein ---
            # If ligand_source is a directory, search for ligand files in the directory
            if ligand_source and os.path.isdir(ligand_source):
                ligand_path = find_ligands_in_dir(ligand_source, base)
                if ligand_path:
                    ligand_found = True      
            
            # If no ligand_source is given or nothing was found, search in the same directory as the protein file
            if not ligand_source or not ligand_found:
                protein_dir = os.path.dirname(protein_path)
                ligand_path = find_ligands_in_dir(protein_dir, base)
                if ligand_path:
                    ligand_found = True
                    
            if not ligand_found:
                print(f"Warning: No ligand found for {protein_file}, skipping.")
                continue
            else:
                protein_ligand_pairs.append((protein_path, ligand_path))
    else:
        raise ValueError(f"Invalid protein source: {protein_source}")
    
    if not protein_ligand_pairs:
        raise ValueError(f"No protein-ligand pairs found for {protein_source} and {ligand_source}")
    
    return protein_ligand_pairs



def encode_protein_ligand_pairs(protein_ligand_pairs: List[Tuple[str, str]], 
                                encoder: ProteinGraphEncoder, 
                                output_dir: Optional[str] = None,
                                include_lig_bonds: bool = True,
                                detect_lig_bonds_by_distance: bool = False
                                ):
    """
    Encode the protein-ligand pairs into graphs using the given encoder.

    Args:
        protein_ligand_pairs: List of tuples (str, str) of protein-ligand pair file paths
        encoder: ProteinGraphEncoder instance
        output_dir: Optional output directory for saving graphs
        include_lig_bonds: Include ligand bonds in the graph
        detect_lig_bonds_by_distance: Detect ligand bonds by distance

    Saves graphs (.pt files) to output_dir or next to protein files
    """
    
    for i, (protein_path, ligand_path) in enumerate(protein_ligand_pairs):
        
        protein_file = os.path.basename(protein_path)
        ligand_file = os.path.basename(ligand_path)
        print(f"\nProcessing {protein_file} and {ligand_file} ({i+1}/{len(protein_ligand_pairs)})")
        
        # Parse the ligand and the protein
        # ---------------------------------------------------------------------------------
        ligand = Ligand(ligand_path, 
                        include_bonds=include_lig_bonds, 
                        detect_bonds_by_distance=detect_lig_bonds_by_distance)
        
        protein = Protein(protein_path)

        if not (ligand and protein):
            print(f"Warning: No ligand or protein parsed for {protein_file}+{ligand_file} - skipping.")
            continue
        # ---------------------------------------------------------------------------------


        # Encode the protein and ligand pairs into graphs
        # ---------------------------------------------------------------------------------
        try:
            graph = encoder.encode(protein, ligand)
            if not graph:
                    print(f"Warning: No graph returned from encoder for {protein_file}+{ligand_file} - skipping.")
                    continue

        except Exception as e:
            print(f"Error encoding {protein_file}+{ligand_file}: {e}, skipping.")
            traceback.print_exc()  # <-- prints full stack trace to stderr
            continue
        # ---------------------------------------------------------------------------------
        
        
        # If output_dir is provided, save all graphs directly in the output directory, else save next to protein files
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, protein_file.replace('.pdb', '.pt'))
        else:
            protein_dir = os.path.dirname(protein_path)
            output_file = os.path.join(protein_dir, protein_file.replace('.pdb', '.pt'))
        
        torch.save(graph, output_file)
        print(f"Saved graph to {output_file}")


    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total protein-ligand pairs found: {len(protein_ligand_pairs)}")
    if output_dir:
        print(f"All graphs saved to: {output_dir}")
    else:
        print(f"Graphs saved next to their respective protein files")
    print(f"{'='*60}")






if __name__ == "__main__":

    args = parse_args()
    print(f"Using args: {args}")

    encoder = ProteinGraphEncoder(granularity=args.granularity)

    protein_ligand_pairs = find_protein_ligand_pairs(args.protein_source, args.ligand_source)
    
    encode_protein_ligand_pairs(protein_ligand_pairs, 
                        encoder, 
                        args.output_dir, 
                        args.include_lig_bonds, 
                        args.detect_lig_bonds_by_distance)