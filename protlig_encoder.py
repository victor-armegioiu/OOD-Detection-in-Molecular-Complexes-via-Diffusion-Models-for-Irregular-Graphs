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
import numpy as np
from constants import atom_decoder, aa_decoder3, metals, protein_letters_1to3, protein_letters_3to1_extended
from parsing import Protein, Ligand

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


def find_pdb_files_recursive(directory: str) -> List[str]:
    """
    Recursively find all PDB files in a directory and its subdirectories.
    Excludes ligand files (ending with '_lig.pdb').
    """
    pdb_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdb') and not file.endswith('_lig.pdb'):
                pdb_files.append(os.path.join(root, file))
    return sorted(pdb_files)


def one_of_k_encoding(x, allowable_set, padding_length=0, padding_position=1):
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


    def encode(self, protein: Protein, ligand: Optional[Ligand] = None) -> Dict[str, Any]:
        if self.granularity == "atom-level":
            return self._encode_atom_level(protein, ligand)
        elif self.granularity == "residue-level-fully-connected":
            return self._encode_residue_level_fully_connected(protein, ligand)
        else:
            raise ValueError(f"Unknown featurization type: {self.granularity}")


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
            ligand_features.append(one_of_k_encoding(atom.atom_type, atom_decoder))

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



    def _encode_residue_level_fully_connected(self, 
                              protein: Protein, 
                              ligand: Optional[Ligand], 
                              interaction_distance: float = 5.0,
                              remove_hydrogens: bool = True):
        
        if not ligand:
            print("No ligand provided to _encode_residue_level_fully_connected, returning None")
            return None
        
        ligand_coords, ligand_features, lig_edge_index, lig_edge_attr = self._ligand_graph(ligand, remove_hydrogens)  

        residues = protein.get_residues()
        
        # Find interacting residues
        interacting_residues = []
        for i, residue in enumerate(residues):
            residue_coords = np.array([[atom['pos_x'], atom['pos_y'], atom['pos_z']] for atom in residue['atoms']])
            
            # Compute all pairwise distances between residue atoms and ligand atoms Shape: (n_residue_atoms, n_ligand_atoms)
            distances = np.sqrt(np.sum((residue_coords[:, np.newaxis, :] - ligand_coords[np.newaxis, :, :]) ** 2, axis=2))
            if np.any(distances < interaction_distance):
                interacting_residues.append(i)
        
        print(f"Found {len(interacting_residues)} residues within {interaction_distance}Å of ligand")

        # Protein Features and Coordinates for interacting residues
        protein_coords = []
        protein_features = []
        for idx in interacting_residues:
            res = residues[idx]
            
            if res['name'].strip('0123456789').upper() in [m.upper() for m in metals]:
                res['name'] = 'METAL'
            
            # One-hot encode the residue name
            res_features = one_of_k_encoding(res['name'], aa_decoder3)
            if res_features is None:
                # If not found, see if there is a standard version of the residue name
                try:
                    standardized = protein_letters_1to3[protein_letters_3to1_extended[res['name']]]
                except:
                    print(f"Warning: {res['name']} not found in aa_decoder3 or protein_letters_3to1_extended, returning None")
                    return None
                res_features = one_of_k_encoding(standardized.upper(), aa_decoder3)
            
            if res_features is None:
                print(f"Warning: {res['name']} not found in aa_decoder3, returning None")
                return None

            protein_features.append(res_features)
            
            if res['name'] == 'METAL':
                protein_coords.append([res['atoms'][0]['pos_x'], res['atoms'][0]['pos_y'], res['atoms'][0]['pos_z']])
            else:
                added_ca = False
                for atom in res['atoms']:
                    if atom['name'] == 'CA':
                        protein_coords.append([atom['pos_x'], atom['pos_y'], atom['pos_z']])
                        added_ca = True
                if not added_ca:
                    protein_coords.append([res['atoms'][0]['pos_x'], res['atoms'][0]['pos_y'], res['atoms'][0]['pos_z']])

        protein_coords = np.array(protein_coords)
        protein_features = np.array(protein_features)            
        
        # Transform into fully connected graph with distance as edge attribute
        pos = np.concatenate([ligand_coords, protein_coords], axis=0)
        x = padded_concat(ligand_features, protein_features)
        lig_mask = np.concatenate([np.ones(len(ligand_coords)), np.zeros(len(protein_coords))])
        prot_mask = np.concatenate([np.zeros(len(ligand_coords)), np.ones(len(protein_coords))])
        num_nodes = len(pos)
        
        # Create all possible pairs of nodes for a fully connected graph
        edge_index = [[i, j] for i in range(num_nodes) for j in range(num_nodes)]
        edge_index = np.array(edge_index).T  # Shape: (2, num_edges)
        
        # Calculate distances between all pairs of nodes
        source_coords = pos[edge_index[0]]  # Shape: (num_edges, 3)
        target_coords = pos[edge_index[1]]  # Shape: (num_edges, 3)
        
        # Calculate Euclidean distances
        edge_attr = np.sqrt(np.sum((source_coords - target_coords) ** 2, axis=1))
        edge_attr = edge_attr.reshape(-1, 1)

        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.FloatTensor(edge_attr)
        pos = torch.FloatTensor(pos)
        return x, edge_index, edge_attr, pos, torch.BoolTensor(lig_mask), torch.BoolTensor(prot_mask)



    def _encode_atom_level(self, protein: Protein, ligand: Optional[Ligand]) -> Dict[str, Any]:
        atoms = protein.get_atoms()

        print("\n=== Protein Atom-Level Information ===")
        print(f"Total number of atoms: {len(atoms)}")
        # Print first 3 atoms as examples
        for i, atom in enumerate(atoms[:3]):
            print(f"\nAtom {i + 1}:")
            print(f"  Name: {atom['name']}")
            print(f"  Element: {atom['element']}")
            print(f"  Position: ({atom['pos_x']:.3f}, {atom['pos_y']:.3f}, {atom['pos_z']:.3f})")

        if ligand:
            print("\n=== Ligand Information ===")
            print(f"Total number of atoms in ligand: {len(ligand.atoms)}")
            # Print first 3 ligand atoms as examples
            for i, atom in enumerate(ligand.atoms[:3]):
                print(f"\nLigand Atom {i + 1}:")
                print(f"  Type: {atom.atom_type}")
                print(f"  Position: ({atom.x:.3f}, {atom.y:.3f}, {atom.z:.3f})")
            
            print(f"\nTotal number of bonds in ligand: {len(ligand.bonds)}")
            # Print first 3 bonds as examples
            for i, bond in enumerate(ligand.bonds[:3]):
                print(f"\nLigand Bond {i + 1}:")
                print(f"  Between atoms: {bond.atom1_idx + 1} and {bond.atom2_idx + 1}")
                print(f"  Bond type: {bond.bond_type}")
                print(f"  Atoms involved: {ligand.atoms[bond.atom1_idx].atom_type} - {ligand.atoms[bond.atom2_idx].atom_type}")

        return {"nodes": [], "edges": [], "features": []}



def process_folder(protein_source: str, 
                   encoder: ProteinGraphEncoder, 
                   ligand_source: Optional[str] = None, 
                   output_dir: Optional[str] = None, 
                   include_lig_bonds: bool = True, 
                   detect_lig_bonds_by_distance: bool = False):

    # Detect global protein, else find all PDB files in protein_source and subdirectories
    if is_global_protein := protein_source and os.path.isfile(protein_source):
        print(f"Using global protein from {protein_source}")
        pdb_paths = [protein_source]
    else:
        pdb_paths = find_pdb_files_recursive(protein_source)
        print(f"Found {len(pdb_paths)} PDB files in {protein_source} and subdirectories")

    # Detect global ligand if ligand_source is a file, else set ligand to None for the moment
    if is_global_ligand := ligand_source and os.path.isfile(ligand_source):
        print(f"Using global ligand from {ligand_source}")
        ligand = Ligand(ligand_source,
                include_bonds=include_lig_bonds, 
                detect_bonds_by_distance=detect_lig_bonds_by_distance)
        if not ligand.atoms:
            print(f"Warning: Global ligand could not be parsed. Exiting.")
            exit(1)
    else:
        ligand = None

    # For each PDB file, find the corresponding ligand (if not global) and encode the complex
    for i, pdb_path in enumerate(pdb_paths, 1):
        protein = None
        pdb_file = os.path.basename(pdb_path)
        rel_path = os.path.relpath(pdb_path, protein_source)
        print(f"\nProcessing [{i}/{len(pdb_paths)}] {rel_path}")
        protein = Protein(pdb_path)

        # --- FIND THE LIGANDS for this protein ---
        if ligand_source and not is_global_ligand:
            ligand = None
            base = os.path.splitext(pdb_file)[0]            
            
            # Search for ligand files in the same directory as the protein file first
            protein_dir = os.path.dirname(pdb_path)
            ligand_found = False
            for ext in ['.mol2', '.sdf', '_lig.pdb']:
                ligand_path = os.path.join(protein_dir, base + ext)
                if os.path.exists(ligand_path):
                    print(f"Processing {os.path.basename(ligand_path)}")
                    ligand = Ligand(ligand_path, 
                                    include_bonds=include_lig_bonds, 
                                    detect_bonds_by_distance=detect_lig_bonds_by_distance)
                    ligand_found = True
                    break
            
            # If not found in same directory, try the ligand_source directory
            if not ligand_found and ligand_source:
                for ext in ['.mol2', '.sdf', '_lig.pdb']:
                    ligand_path = os.path.join(ligand_source, base + ext)
                    if os.path.exists(ligand_path):
                        print(f"Processing {os.path.basename(ligand_path)} from ligand_source")
                        ligand = Ligand(ligand_path, 
                                        include_bonds=include_lig_bonds, 
                                        detect_bonds_by_distance=detect_lig_bonds_by_distance)
                        ligand_found = True
                        break

            if not ligand_found:
                print(f"Warning: No ligand found for {pdb_file}, skipping.")
                continue

        if not (ligand and protein):
            print(f"Warning: No ligand or protein generated for {pdb_file}, skipping.")
            continue


        # Encode the protein and ligand as a graph
        # ---------------------------------------------------------------------------------
        try:
            graph = encoder.encode(protein, ligand)

            if graph: 
                x, edge_index, edge_attr, pos, lig_mask, prot_mask = graph
            else:
                print(f"Warning: No graph returned from encoder for {pdb_file}, skipping.")
                continue
        except Exception as e:
            print(f"Error encoding {pdb_file}: {e}, skipping.")
            continue
        # ---------------------------------------------------------------------------------
        
        # Save graph data
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, lig_mask=lig_mask, prot_mask=prot_mask)
        
        # If output_dir is provided, save all graphs directly in the output directory, else save next to protein files
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, pdb_file.replace('.pdb', '.pt'))
        else:
            protein_dir = os.path.dirname(pdb_path)
            output_file = os.path.join(protein_dir, pdb_file.replace('.pdb', '.pt'))
        
        torch.save(graph, output_file)
        print(f"Saved graph to {output_file}")


    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total PDB files found: {len(pdb_paths)}")
    if output_dir:
        print(f"All graphs saved to: {output_dir}")
    else:
        print(f"Graphs saved next to their respective protein files")
    print(f"{'='*60}")


if __name__ == "__main__":

    args = parse_args()
    print(f"Using args: {args}")

    encoder = ProteinGraphEncoder(granularity=args.granularity)
    process_folder(args.protein_source, 
                        encoder, 
                        args.ligand_source, 
                        args.output_dir, 
                        args.include_lig_bonds, 
                        args.detect_lig_bonds_by_distance)