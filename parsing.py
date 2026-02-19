import os
from typing import List, Dict, Any, NamedTuple
import gemmi
import numpy as np
from constants import COVALENT_RADII


class AtomInfo(NamedTuple):
    atom_type: str
    x: float
    y: float
    z: float

class Bond(NamedTuple):
    atom1_idx: int  # Index in the atoms list
    atom2_idx: int
    bond_type: str  # 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'


class Protein:
    def __init__(self, pdb_path: str):
        self.pdb_path = pdb_path
        structure = self._load_structure()  # Don't store the structure as instance variable
        self._atoms = self._extract_atoms(structure)
        self._residues = self._extract_residues(structure)
        structure = None # Let the structure object be garbage collected after we're done with it

    def _load_structure(self) -> gemmi.Structure:
        structure = gemmi.read_structure(self.pdb_path)
        structure.setup_entities()
        return structure

    def _extract_atoms(self, structure: gemmi.Structure) -> List[Dict[str, Any]]:
        atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atom_data = {
                            'name': atom.name,
                            'element': atom.element.name,
                            'pos_x': atom.pos.x,
                            'pos_y': atom.pos.y,
                            'pos_z': atom.pos.z,
                        }
                        atoms.append(atom_data)
        return atoms

    def _extract_residues(self, structure: gemmi.Structure) -> List[Dict[str, Any]]:
        residues = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.name in ['HOH', 'WAT', 'H2O', 'DOD']: continue
                    res_data = {
                        'name': residue.name,
                        'seqid': residue.seqid.num,
                        'chain_name': chain.name,
                        'atoms': []
                    }
                    for atom in residue:
                        atom_data = {
                            'name': atom.name,
                            'element': atom.element.name,
                            'pos_x': atom.pos.x,
                            'pos_y': atom.pos.y,
                            'pos_z': atom.pos.z
                        }
                        res_data['atoms'].append(atom_data)
                    residues.append(res_data)
        return residues

    def get_atoms(self) -> List[Dict[str, Any]]:
        return self._atoms

    def get_residues(self) -> List[Dict[str, Any]]:
        return self._residues



class Ligand:
    def __init__(self, file_path: str, include_bonds: bool = True, detect_bonds_by_distance: bool = False):
        self.file_path = file_path
        self.atoms = []
        self.bonds = []
        self.include_bonds = include_bonds
        self.detect_bonds_by_distance = detect_bonds_by_distance
        self._load_ligand()


    def _load_ligand(self):
            
        # Parse the ligand file
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext == '.sdf':
            self._parse_sdf()
        elif ext == '.mol2':
            self._parse_mol2()
        elif ext == '.pdb':
            self._parse_pdb()
        else:
            print(f"Warning: Unsupported ligand format: {self.file_path}")
        
        # DISTANCE-BASED DETECTION: If include_bonds is True and we have no bonds (either because 
        # detect_bonds_by_distance=True or bond parsing failed), use distance-based detection
        if self.include_bonds and not self.bonds:
            self._detect_bonds_by_distance()


    def _parse_mol2(self):
        '''
        Parse MOL2 format ligand file
        '''
        in_atom_section = False
        in_bond_section = False
        parse_bonds = self.include_bonds and not self.detect_bonds_by_distance
        
        with open(self.file_path, 'r') as f:
            for line in f:
                if "@<TRIPOS>ATOM" in line:
                    in_atom_section = True
                    in_bond_section = False
                    continue
                elif "@<TRIPOS>BOND" in line:
                    if not parse_bonds:
                        break
                    in_atom_section = False
                    in_bond_section = True
                    continue
                elif "@<TRIPOS>" in line:
                    in_atom_section = False
                    in_bond_section = False
                    continue
                
                if in_atom_section:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        try:
                            atom_info = AtomInfo(
                                atom_type=parts[5].split('.')[0],  # Remove atom type modifiers
                                x=float(parts[2]),
                                y=float(parts[3]),
                                z=float(parts[4])
                            )
                            self.atoms.append(atom_info)
                            print(f"Found atom: {line.strip()}")
                            continue
                        except (ValueError, IndexError):
                            continue
                
                elif in_bond_section and parse_bonds:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        try:
                            bond = Bond(
                                atom1_idx=int(parts[1]) - 1, # Convert to 0-based indexing
                                atom2_idx=int(parts[2]) - 1,
                                bond_type=self._convert_bond_type(parts[3])
                            )
                            # Validate bond indices
                            if not (bond.atom1_idx >= len(self.atoms) or bond.atom2_idx >= len(self.atoms)):
                                self.bonds.append(bond)
                                print(f"Found bond: {line.strip()}")
                                continue

                        except (ValueError, IndexError):
                            pass
                print(f"Found nothing: {line.strip()}")


    def _parse_sdf(self):
        '''
        Parse SDF format ligand file
        '''
        parse_bonds = self.include_bonds and not self.detect_bonds_by_distance
        
        with open(self.file_path, 'r') as f:
            lines = f.readlines()

            try:
                for line in lines:
                    parts = line.strip().split()
                    if not parts: continue
                        
                    # Detect atom lines by checking if first 3 elements are coordinates and 4th element is a valid
                    # atom type (uppercase letter followed by optional lowercase)
                    try:
                        if len(parts) >= 4:
                            _ = float(parts[0])
                            _ = float(parts[1])
                            _ = float(parts[2])

                            if parts[3].isalpha() and parts[3][0].isupper():
                                atom_info = AtomInfo(
                                    atom_type=parts[3],
                                    x=float(parts[0]),
                                    y=float(parts[1]),
                                    z=float(parts[2])
                                )
                                self.atoms.append(atom_info)
                                # print(f"Found atom: {atom_info}")
                                continue
                    except (ValueError, IndexError):
                        pass

                    # Detect bond lines by checking if the first 2 elements are atom indices (positive integers) and the 
                    # third element is a valid bond type 
                    if parse_bonds:
                        try:
                            if len(parts) >= 3:
                                atom1 = int(parts[0])
                                atom2 = int(parts[1])
                                bond_type = int(parts[2])
                                if atom1 > 0 and atom2 > 0 and bond_type in [1, 2, 3, 4]:
                                    bond = Bond(
                                        atom1_idx=atom1 - 1,  # Convert to 0-based indexing
                                        atom2_idx=atom2 - 1,
                                        bond_type=self._convert_sdf_bond_type(bond_type)
                                        )

                                    # Validate bond indices
                                    if not (bond.atom1_idx >= len(self.atoms) or bond.atom2_idx >= len(self.atoms)):
                                        self.bonds.append(bond)
                                        # print(f"Found bond: {bond}")
                                        continue
                        except (ValueError, IndexError):
                            pass
                    
                print(f"Found {len(self.atoms)} atoms and {len(self.bonds)} bonds in {self.file_path}")
    
            except Exception as e:
                print(f"Warning: Error parsing SDF file {self.file_path}: {str(e)}")
        
        
    def _parse_pdb(self):
        """
        Parse PDB format ligand file    
        Note: This method does not support bonds, but only returns atom information. 
        Bonds can be detected by distance between atoms, see _detect_bonds_by_distance
        """
        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
            
                # PDB ATOM/HETATM records
                if line.startswith(('ATOM', 'HETATM')):
                    try:
                        parts = line.split()
                        if not parts or parts[0] not in ('ATOM', 'HETATM'): continue

                        # Prefer atom name from the canonical 3rd token if available
                        atom_name = parts[2] if len(parts) >= 3 else None

                        # Find coordinates as the first trio of decimal-like floats (contain '.')
                        x, y, z = None, None, None
                        def looks_decimal(token: str) -> bool:
                            return ('.' in token)
                        
                        for i in range(1, len(parts) - 2):
                            p0, p1, p2 = parts[i], parts[i+1], parts[i+2]
                            if looks_decimal(p0) and looks_decimal(p1) and looks_decimal(p2):
                                try:
                                    x, y, z = float(p0), float(p1), float(p2)
                                    break
                                except ValueError:
                                    continue
                        
                        if x is None or y is None or z is None:
                            # Fallback: try fixed positions 6-8 (0-based 5-7) commonly used in split PDB
                            try:
                                x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
                            except Exception:
                                print(f"Could not find valid coordinates in line: {line}")
                                continue
                        
                        # Determine element: prefer last short alphabetic token, else derive from atom_name letters
                        element = None
                        if parts:
                            last = parts[-1]
                            if last.isalpha() and 1 <= len(last) <= 2:
                                element = last.capitalize()
                        if element is None and atom_name:
                            letters = ''.join(c for c in atom_name if c.isalpha())[:2]
                            if letters:
                                element = letters.capitalize()
                        if not element:
                            element = 'C'
                        
                        atom_info = AtomInfo(
                            atom_type=element,
                            x=x,
                            y=y,
                            z=z
                        )
                        self.atoms.append(atom_info)
                        # print(f"Found atom: {element} at ({x:.3f}, {y:.3f}, {z:.3f})")
                        
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing PDB line: {line} - {e}")
                        continue
                
                # Stop parsing at END record
                elif line.startswith('END'):
                    break
        
        print(f"Found {len(self.atoms)} atoms in {self.file_path}")



    def _detect_bonds_by_distance(self, tolerance: float = 0.45):
        """Detect bonds based on distance between atoms and their covalent radii."""
        if not self.atoms:
            return

        # Create coordinate array
        coords = np.array([[atom.x, atom.y, atom.z] for atom in self.atoms])
        
        # Calculate all pairwise distances
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=-1))
        
        # Calculate expected bond distances based on covalent radii
        n_atoms = len(self.atoms)
        expected_distances = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r1 = COVALENT_RADII.get(self.atoms[i].atom_type, 1.0)
                r2 = COVALENT_RADII.get(self.atoms[j].atom_type, 1.0)
                expected_distances[i, j] = expected_distances[j, i] = r1 + r2

        # Find bonds where distance is less than expected + tolerance
        bonds = []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if distances[i, j] <= expected_distances[i, j] + tolerance:
                    bonds.append(Bond(i, j, 'SINGLE'))  # Default to single bonds
        self.bonds = bonds


    @staticmethod
    def _convert_bond_type(mol2_type: str) -> str:
        """Convert MOL2 bond type to standard representation."""
        type_map = {
            '1': 'SINGLE', 'ar': 'AROMATIC', '2': 'DOUBLE',
            '3': 'TRIPLE', 'am': 'SINGLE',  # Amide bonds as single
            'du': 'SINGLE', 'un': 'SINGLE'   # Default unknown to single
        }
        return type_map.get(mol2_type.lower(), 'SINGLE')

    @staticmethod
    def _convert_sdf_bond_type(sdf_type: int) -> str:
        """Convert SDF bond type to standard representation."""
        type_map = {
            1: 'SINGLE', 2: 'DOUBLE', 3: 'TRIPLE',
            4: 'AROMATIC'
        }
        return type_map.get(sdf_type, 'SINGLE')