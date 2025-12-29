import os
from typing import List, Dict, Any, NamedTuple
import gemmi
import numpy as np
import torch
import warnings
from pathlib import Path
import math

from moldiff.constants import (COVALENT_RADII, 
                       bond_decoder, 
                       atom_encoder,
                       atom_decoder, 
                       aa_decoder3, 
                       metals, 
                       protein_letters_1to3, 
                       protein_letters_3to1,
                       protein_letters_3to1_extended, 
                       aa_atom_index, 
                       aa_atom_mask, 
                       aa_nerf_indices, 
                       aa_chi_indices, 
                       aa_chi_anchor_atom)  

# from nerf import ic_to_coords, get_nerf_params_wrapper


class AtomInfo(NamedTuple):
    atom_type: str
    x: float
    y: float
    z: float

class Bond(NamedTuple):
    atom1_idx: int  # Index in the atoms list
    atom2_idx: int
    bond_type: str  # 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'

# https://github.com/EleutherAI/mp_nerf/blob/master/mp_nerf/utils.py
def get_dihedral(c1, c2, c3, c4):
    """ Returns the dihedral angle ϕ=∠(plane (c1​,c2​,c3​),plane (c2​,c3​,c4​)) around the bond c2-c3 in radians.
        Will use atan2 formula from:
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        Inputs:
        * c1: (batch, 3) or (3,)
        * c2: (batch, 3) or (3,)
        * c3: (batch, 3) or (3,)
        * c4: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3

    return torch.atan2( ( (torch.norm(u2, dim=-1, keepdim=True) * u1) * torch.cross(u2,u3, dim=-1) ).sum(dim=-1) ,
                        (  torch.cross(u1,u2, dim=-1) * torch.cross(u2, u3, dim=-1) ).sum(dim=-1) )


class Protein:
    def __init__(
            self, 
            pdb_path: str, 
            interacting_residues: List[Dict[str, Any]] | None = None
            ):
        self.pdb_path = pdb_path
        self.interacting_residues = interacting_residues
        self._structure = gemmi.read_structure(pdb_path)
        assert len(self._structure) == 1, "PDB file contains two distinct protein models"
        self._structure.setup_entities()

        # check residues
        if self.interacting_residues is not None:
            self.parser_type = "gemmi" if isinstance(interacting_residues[0], gemmi.Residue) else "dict"
            if self.parser_type == "gemmi":
                # NOTE: main barrier is that chain name is not stored in gemmi.Residue objects
                raise NotImplementedError("Side-chain rotation for list gemmi residues not implemented yet: Chain issue")
            else:
                # check if elements have the necessary keys: 
                assert all([key in interacting_residues[0].keys() for key in ["chain_name", "seqid_num", "seqid_icode"]]), (
                    "A key of "  + ",".join(["chain_name", "seqid_num", "seqid_icode"]) +f" is missing: {interacting_residues[0].keys()}")
        

    @property
    def structure(self) -> gemmi.Structure:
        """Access the underlying gemmi.Structure object."""
        return self._structure

    @property
    def residues(self):
        return [
            res for model in self._structure
            for chain in model
            for res in chain
        ]

    @property
    def atoms(self):
        return [
            atom for model in self._structure
            for chain in model
            for res in chain
            for atom in res
        ]

    # method to return list of dicts for old setup
    def get_atoms(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': atom.name,
                'element': atom.element.name,
                'pos_x': atom.pos.x,
                'pos_y': atom.pos.y,
                'pos_z': atom.pos.z
            } for model in self._structure
            for chain in model
            for res in chain
            for atom in res
        ]

    # method to return list of dicts for old setup
    def get_residues(self) -> List[Dict[str, Any]]:
        return [
            {   'chain_name': chain.name,
                'name': residue.name,
                'seqid_num': residue.seqid.num,
                'chain_name': chain.name,
                'seqid_icode': residue.seqid.icode,
                'atoms': [
                    {
                        'name': atom.name,
                        'element': atom.element.name,
                        'pos_x': atom.pos.x,
                        'pos_y': atom.pos.y,
                        'pos_z': atom.pos.z
                    } for atom in residue
                ]
            } for model in self._structure
            for chain in model
            for residue in chain
            if residue.name not in ['HOH', 'WAT', 'H2O', 'DOD'] # water exclusion filter
        ]

    

    def rotate_side_chains(
            self, 
            new_chi
           ):

        if self.interacting_residues is None:
            raise ValueError("Pass interacting_residue list upon Protein object instantiation")

        get_nerf_params = get_nerf_params_wrapper(self.parser_type)
        nerf_params = [get_nerf_params(residue) for residue in self.interacting_residues]
        stacked_nerf_params = {k: torch.stack([d[k] for d in nerf_params]) for k in nerf_params[0].keys()}

        # calculate_new_coordinates
        updated_coords = ic_to_coords(
            fixed_coord = stacked_nerf_params["fixed_coord"], 
            atom_mask = stacked_nerf_params["atom_mask"], 
            nerf_indices = stacked_nerf_params["nerf_indices"], 
            length = stacked_nerf_params["length"], 
            theta = stacked_nerf_params["theta"], 
            chi = new_chi, 
            ddihedral = stacked_nerf_params["ddihedral"], 
            chi_indices = stacked_nerf_params["chi_indices"]
            )
        

        # extract the model
        model = self._structure[0]

        # order of updated_coords and interacting_res_ids matched: assumes that interacting residue
        for res_idx, residue_ in enumerate(self.interacting_residues):

            chain_name = residue_["chain_name"]
            res_seqid_num = residue_["seqid_num"]
            res_seqid_icode = residue_["seqid_icode"]

            # find the interacting residue in the model object and check if it is unique
            residue_group = model.find_residue_group(chain = chain_name, seqid = gemmi.SeqId(res_seqid_num, res_seqid_icode))
            assert len(residue_group) == 1, "Multiple residues with same seqid found"
            residue = residue_group[0]

            # sanity check if CA position matches
            ca = next((atom for atom in residue if atom.name == "CA"), None)
            assert ca, f"No CA found in residue {residue.seqid.num}"
            assert np.allclose(
                np.array(ca.pos.tolist()),
                np.array(stacked_nerf_params["fixed_coord"][res_idx, 1, :].tolist())
            ), (
                f"CA position mismatch at residue index {res_idx}:\n"
                f"  gemmi: {np.array(ca.pos.tolist())}\n"
                f"  tensor: {np.array(stacked_nerf_params['fixed_coord'][res_idx, 1, :].tolist())}\n"
                f"  Δ = {np.array(ca.pos.tolist()) - np.array(stacked_nerf_params['fixed_coord'][res_idx, 1, :].tolist())}"
            )

            # update atom positions in the structure object
            atom_idxs = aa_atom_index[protein_letters_3to1_extended[residue.name]]

            for atom in residue: 
                if atom.name not in atom_idxs:
                    continue 
                try: 
                    atom_idx = atom_idxs[atom.name]
                    atom.pos =  gemmi.Position(*map(float, updated_coords[res_idx, atom_idx, :]))
                except KeyError:
                    # if atom.element != "H": # skip warning for hydrogens
                    warnings.warn(f"{atom.name} in residue {residue.seqid.num} could not be updated")
                    continue
    
    def detect_steric_clashes(
            self, 
            ligand_coords, 
            ligand_atom_types,
            clash_margin: float = 0., 
            pymol_save_path: str|None = None
            ):
        """Detect steric clashes between protein and ligand atoms.
        
        Args:
            ligand_coords: tensor of shape (N, 3) with xyz coordinates 
            ligand_atom_types: tensor of shape (N, 10) with one-hot encoded atom types
            clash_margin: Additional distance buffer for clash detection

        Returns:
            dict: Contains number of clashes and details of each clash
        """
        # Input validation
        assert ligand_coords.shape[1] == 3, f"Expected ligand coords shape (N,3), got {ligand_coords.shape}"
        assert ligand_atom_types.shape[1] == 10, f"Expected ligand atom types shape (N,10), got {ligand_atom_types.shape}"
        assert ligand_coords.shape[0] == ligand_atom_types.shape[0], "Mismatch between coords and atom types length"

        # Decode ligand atom types
        ligand_elements = [atom_decoder[i] for i in torch.argmax(ligand_atom_types, dim=1)]
        h_mask = torch.sum(ligand_atom_types, dim=1) == 0  # identify missing atom types
        ligand_elements = [
            "H" if h_mask[i] else ligand_elements[i]
            for i in range(len(ligand_elements))
        ] # fill Hydrogens

        # Debug
        # print("Ligand coords:")
        # for i in range(len(ligand_elements)):
        #     print(i+1, ligand_elements[i], ligand_coords[i])

        # Get protein atom coordinates and types
        protein_coords = []
        protein_elements = []
        protein_residues = []
        protein_atom_names = []
        # protein_atom_altlocs = []
        print(self._structure)
        # NOTE: can be optimized if interacting residues are passed
        for model in self._structure:
            for chain in model:
                for residue in chain:
                    if residue.name in ['HOH', 'WAT', 'H2O', 'DOD']:
                        continue
                    for atom in residue:
                        protein_coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                        protein_elements.append(atom.element.name)
                        protein_residues.append(f"{chain.name}_{residue.name}_{residue.seqid.num}")
                        protein_atom_names.append(f"{atom.name}")


        protein_coords = torch.tensor(protein_coords)
        
        # Debug
        print("Protein coords")
        for i in range(len(protein_coords)):
            print(protein_residues[i], protein_elements[i], protein_atom_names[i], protein_coords[i])

        # Calculate all pairwise distances
        diffs = protein_coords.unsqueeze(1) - ligand_coords.unsqueeze(0)
        distances = torch.norm(diffs, dim=2)

        assert distances.shape == (len(protein_coords), len(ligand_coords)), (
            f"Unexpected dist matrix shape of {distances.shape}, expected {(len(protein_coords), len(ligand_coords))}"
        )

        # Check for clashes and record details
        clashes = []
        for i, (prot_element, residue, atom_name) in enumerate(zip(protein_elements, protein_residues, protein_atom_names)):
            for j, lig_element in enumerate(ligand_elements):
                # detect minimal possible distance
                min_dist = (COVALENT_RADII[prot_element] + COVALENT_RADII[lig_element] + clash_margin)
                # check for a clash
                if distances[i,j] < min_dist:
                    clashes.append({
                    'protein_residue': residue,
                    'protein_atom': atom_name,
                    'protein_element': prot_element,
                    'protein_coord': protein_coords[i],
                    'ligand_idx': j,
                    'ligand_element': lig_element,
                    'ligand_coord': ligand_coords[j],
                    'distance': distances[i,j].item(),
                    'min_allowed': min_dist
                    })

        results =  {
            'num_clashes': len(clashes),
            'clashes': clashes
        }

        if pymol_save_path is not None:
            self.write_clash_pml(results, pymol_save_path)

        return results 

    
    @staticmethod
    def write_clash_pml(results, output_pml="show_clashes.pml"):
        """
        Write a minimal PyMOL .pml script to visualize steric clashes
        from a preloaded session with 'pocket' and 'ligand' objects.

        Parameters
        ----------
        results : dict
            Output of clash detection:
            {'num_clashes': int, 'clashes': list of dicts with
            'protein_residue', 'protein_atom', 'ligand_idx', 'distance', ...}
        output_pml : str
            Output .pml file name.
        """
        clashes = results.get("clashes", [])
        with open(output_pml, "w") as f:
            f.write("# Auto-generated PyMOL script for clash visualization\n")
            f.write("set dash_color, red\n")
            f.write("set dash_width, 2.0\n")
            f.write("set dash_gap, 0.1\n\n")

            for k, clash in enumerate(clashes):
                lig_idx = clash["ligand_idx"] + 1  # PyMOL atom indices start at 1
                residue_chain, residue_name, residue_number = clash["protein_residue"].split("_")
                atom_name = clash["protein_atom"]
                dist = clash["distance"]
                f.write(
                    f"distance clash_{k}, pocket and chain {residue_chain} and resi {residue_number} "
                    f"and resn {residue_name} and name {atom_name}, "
                    f"ligand and index {lig_idx}\n"
                )   
                f.write(f"color red, clash_{k}\n")
                f.write(f"hide labels, clash_{k}\n")
                f.write(
                    f"label clash_{k}, "
                    f"\"{clash['protein_residue']}:{atom_name} - Lig{lig_idx} ({dist:.2f} A)\"\n\n"
                )

        print(f"PyMOL clash script written to: {output_pml}")

        
    
    def write_structure_to_pdb(self, save_dir = "."):
        
        protein_string = self.pdb_path.split("/")[-1].strip(".pdb")
        assert len(protein_string) == 4, f"Invalid protein_string: {protein_string}" 

        self._structure.write_minimal_pdb(f"{save_dir}/{protein_string}_updated.pdb")

    



# test
def test_side_chain_rotation_and_steric_clashes(
        protein_string = "1a1e", 
        graph_root = Path("data_extracted_graphs_atom_level"),
        pdb_root = Path("mini_example_dataset"), 
        ):

    torch.manual_seed(41)
    max_forgiving_margin = 0.0

    # Ligand
    ligand_path = str(pdb_root / f"{protein_string}.sdf")
    ligand = Ligand(ligand_path)
    
    # convert ligand atoms to tensors
    ligand_coords = torch.tensor([[atom.x, atom.y, atom.z] for atom in ligand.atoms])
    
    # NOTE: Ligand H's are not considered!!
    # create one-hot encoded atom types
    atom_types = torch.zeros((len(ligand.atoms), 10))
    for i, atom in enumerate(ligand.atoms):
        if atom.atom_type == "H": continue
        type_idx = atom_encoder[atom.atom_type]
        atom_types[i, type_idx] = 1

    # Pocket: load the interacting residues and angles features
    graph = torch.load(graph_root / f"{protein_string}.pt")
    interacting_res_ids = graph.interacting_res_ids
    pocket = Protein(str(pdb_root / f"{protein_string}.pdb"), interacting_res_ids)

    # TEST1: steric clashes between loaded objects:
    clash_results_initial = pocket.detect_steric_clashes(
        ligand_coords,
        atom_types,
        clash_margin=max_forgiving_margin,
        pymol_save_path=f"pml/{protein_string}__initial_clashes.pml"
    )

    print(f"Detected {clash_results_initial['num_clashes']} steric clashes for no side chain rotation and zero translation")
    if clash_results_initial['num_clashes'] > 0: 
        print("Clashes:")
        for clash in clash_results_initial["clashes"]: print(clash)
        print("\n")


    # TEST2: test side chain rotation: noise the angles and rotate the side chains
    old_chi = graph.x[graph.prot_mask & graph.ca_mask][:, -6:]
    new_chi = old_chi + torch.tensor([2] * 5 + [0]) # large 180 rotation for all sidechains to simulate steric clashes # (torch.rand(old_chi.shape) - 0.5) # add random noise in interval [-0.5, +0.5]
    pocket.rotate_side_chains(new_chi)
    # pocket.write_structure_to_pdb(pdb_root)

    print("\nSide chain rotation pass successful!")
            
    # TEST3: check steric clashes after rotation
    clash_results_zero = pocket.detect_steric_clashes(
        ligand_coords,
        atom_types,
        clash_margin=max_forgiving_margin,
        pymol_save_path=f"pml/{protein_string}__rotated_chains_zero_trans.pml"
    )

    print(f"Detected {clash_results_zero['num_clashes']} steric clashes for zero translation")
    if clash_results_zero['num_clashes'] > 0: 
        print("Clashes:")
        for clash in clash_results_zero["clashes"]: print(clash)
        print("\n")

    # translate ligand by 4A in x direction
    translated_coords = ligand_coords.clone()
    translation = np.array([2.0, 0., 0.])
    translated_coords += translation

    # detect clashes
    clash_results = pocket.detect_steric_clashes(
        translated_coords,
        atom_types,
        clash_margin=max_forgiving_margin,
        pymol_save_path=f"pml/{protein_string}_translation_{'_'.join(f'{x:.2f}' for x in translation)}.pml"
    )
    print(f"Detected {clash_results['num_clashes']} steric clashes for translation {translation}")
    if clash_results['num_clashes'] > 0: 

        print("Clashes:")
        for clash in clash_results["clashes"]: print(clash)
        print("\nSteric clash detection pass successful!")











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
                            # print(f"Found atom: {line.strip()}")
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
                                # print(f"Found bond: {line.strip()}")
                                continue

                        except (ValueError, IndexError):
                            pass
        print(f"--- Found {len(self.atoms)} atoms and {len(self.bonds)} bonds in {self.file_path}")


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
                    
                print(f"--- Found {len(self.atoms)} atoms and {len(self.bonds)} bonds in {self.file_path}")
    
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
    

    
if __name__ == "__main__":
    test_side_chain_rotation_and_steric_clashes()



    def pdb_coords_equal(file1: str, file2: str, tol: float = 1e-4) -> bool:
        """
        Compare two PDB files atom-by-atom.
        Print all atoms whose coordinates differ by more than `tol`.
        Returns True if all coordinates are identical within tolerance.
        """

        s1 = gemmi.read_structure(file1)
        s2 = gemmi.read_structure(file2)

        diffs_found = False
        total_atoms_1 = total_atoms_2 = 0

        for model1, model2 in zip(s1, s2):
            for chain1, chain2 in zip(model1, model2):
                for res1, res2 in zip(chain1, chain2):
                    for atom1, atom2 in zip(res1, res2):
                        total_atoms_1 += 1
                        total_atoms_2 += 1

                        # Convert gemmi.Position → numpy array of floats
                        pos1 = np.array([atom1.pos.x, atom1.pos.y, atom1.pos.z], dtype=float)
                        pos2 = np.array([atom2.pos.x, atom2.pos.y, atom2.pos.z], dtype=float)

                        diff = np.linalg.norm(pos1 - pos2)
                        if diff > tol:
                            diffs_found = True
                            print(
                                f"Diff {diff:.4f} Å at "
                                f"{chain1.name}_{res1.name}_{res1.seqid.num}{res1.seqid.icode.strip()} "
                                f"{atom1.name} "
                                f"{pos1} vs {pos2}"
                            )

        if total_atoms_1 != total_atoms_2:
            print(f"Different atom counts: {total_atoms_1} vs {total_atoms_2}")
            diffs_found = True

        if not diffs_found:
            print(f"✅ All coordinates identical within tolerance {tol} Å.")
            return True
        else:
            print(f"⚠️ Coordinate differences found (tolerance {tol} Å).")
            return False


# Example usage:
# pdb_coords_equal("1a1e_updated.pdb", "1a1e.pdb", tol=1e-3)


    # Example usage
    # pdb_coords_equal("/cluster/work/math/pbaertschi/bioinformatics/mini_example_dataset/1a1e_updated.pdb", "/cluster/work/math/pbaertschi/bioinformatics/mini_example_dataset/1a1e.pdb")

