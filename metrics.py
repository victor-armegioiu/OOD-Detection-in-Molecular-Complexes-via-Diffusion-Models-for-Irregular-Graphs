import torch
import json
import warnings
import tempfile
from openbabel import openbabel
from Dataset import PDBbind_Dataset

from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, UFFHasAllMoleculeParams
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import inchi
from collections import Counter
import numpy as np
from torch_geometric.data import Data
from molecular_diffusion import MolecularDenoisingModel
import argparse
from molecular_samplers import create_molecular_sampler_from_model
import utils
from molecular_diffusion import (
    log_uniform_sampling,
    edm_weighting,
    MolecularDiffusion,
    exponential_noise_schedule
)
from constants import (
    ligand_size_distribution, 
    ligand_to_pocket_size_mapping, 
    atom_decoder, 
    aa_decoder3, 
    atom_freq_dist, 
    aa_freq_dist,
    ring_size_dist)


def load_checkpoint(checkpoint_path: str) -> MolecularDenoisingModel:
    """Load model from checkpoint"""
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # Recreate model 
    model_params = checkpoint['model_params']
    
    # Recreate the diffusion scheme from parameters    
    scheme_params = model_params['scheme_params']
    sigma_schedule = exponential_noise_schedule(
        clip_max=scheme_params['sigma_max'],
        base=np.e**0.5,
        start=0.0,
        end=5.0
    )
    
    scheme = MolecularDiffusion.create_variance_exploding(
        sigma=sigma_schedule,
        coord_norm=scheme_params['coord_norm'],
        feature_norm=scheme_params['feature_norm'],
        feature_bias=scheme_params['feature_bias']
    )
    
    # Create noise sampling and weighting    
    noise_sampling = log_uniform_sampling(
        scheme=scheme,
        clip_min=scheme_params['sigma_min'],
        uniform_grid=False
    )
    
    noise_weighting = edm_weighting(data_std=1.0)
    
    model = MolecularDenoisingModel(
        atom_nf=model_params['atom_nf'],
        residue_nf=model_params['residue_nf'],
        n_dims=model_params['n_dims'],
        joint_nf=model_params['joint_nf'],
        hidden_nf=model_params['hidden_nf'],
        n_layers=model_params['n_layers'],
        edge_embedding_dim=model_params['edge_embedding_dim'],
        update_pocket_coords=model_params['update_pocket_coords'],
        scheme=scheme,
        noise_sampling=noise_sampling,
        noise_weighting=noise_weighting
    )
    
    model.initialize()
    model.denoiser.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Model loaded successfully!")
    return model


def load_pocket_data_from_dataset(dataset_path: str, num_samples: int, model: MolecularDenoisingModel):
    """
    Load real pocket data from dataset to use as conditioning
    Returns pocket_data dict and corresponding ligand_sizes
    """
    dataset = torch.load(dataset_path)
    
    # Sample random complexes from dataset
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    
    all_pocket_coords = []
    all_pocket_features = []
    all_pocket_masks = []
    ligand_sizes = []
    
    for i, idx in enumerate(indices):
        data_point = dataset[idx]
        
        # Get pocket information
        pocket_coords = data_point.prot_coords.cuda()
        pocket_features = data_point.prot_features.cuda()
        
        # Create batch mask for this pocket (all atoms belong to batch index i)
        pocket_mask = torch.full((len(pocket_coords),), i, dtype=torch.long, device='cuda')
        
        all_pocket_coords.append(pocket_coords)
        all_pocket_features.append(pocket_features)
        all_pocket_masks.append(pocket_mask)
        
        # Get ligand size from the original complex (for generating similar-sized ligands)
        ligand_sizes.append(len(data_point.lig_coords))
    
    # Concatenate into batch format
    pocket_coords = torch.cat(all_pocket_coords, dim=0)
    pocket_features = torch.cat(all_pocket_features, dim=0)
    pocket_mask = torch.cat(all_pocket_masks, dim=0)
    
    # Convert pocket features to embeddings using the model's encoder
    with torch.no_grad():
        pocket_embeddings = model.denoiser.egnn_dynamics.residue_encoder(pocket_features)
        pocket_embeddings = F.normalize(pocket_embeddings, dim=-1)
    
    pocket_data = {
        'coords': pocket_coords,
        'embeddings': pocket_embeddings,
        'pocket_mask': pocket_mask
    }
    
    return pocket_data, ligand_sizes

def compare_distributions(dist1, dist2):
    """
    Compare two probability distributions using KL and JS divergence.
    Args:
        dist1: First distribution array
        dist2: Second distribution array
    Returns:
        dict: Dictionary containing KL and JS divergence metrics
    """
    dist1 = dist1 / np.sum(dist1)
    dist2 = dist2 / np.sum(dist2)
    
    # Compute KL divergence
    epsilon = 1e-10
    kl_div = np.sum(dist1 * np.log((dist1 + epsilon) / (dist2 + epsilon)))
    
    # Compute Jensen-Shannon divergence
    m = 0.5 * (dist1 + dist2)
    js_div = 0.5 * np.sum(dist1 * np.log((dist1 + epsilon) / (m + epsilon))) + \
             0.5 * np.sum(dist2 * np.log((dist2 + epsilon) / (m + epsilon)))
    
    return float(kl_div), float(js_div)


def sample_lig_pocket_sizes(N: int, lig_lower_bound: int = 10, lig_upper_bound: int = 50):
    """
    Sample N ligand sizes and corresponding pocket sizes from the saved distributions.
    Returns: tuple: (ligand_sizes, pocket_sizes)
    """
    ligand_sizes = []
    pocket_sizes = []
    
    # Convert to numpy array and normalize the slice we want (10-49)
    ligand_size_dist = np.array(ligand_size_distribution[lig_lower_bound:lig_upper_bound])
    ligand_size_dist = ligand_size_dist / ligand_size_dist.sum()

    for _ in range(N):
        ligand_size = np.random.choice(np.arange(10, 50), p=ligand_size_dist)
        ligand_sizes.append(ligand_size)
        pocket_dist = np.array(ligand_to_pocket_size_mapping[ligand_size])
        if pocket_dist.sum() > 0:
            pocket_dist = pocket_dist / pocket_dist.sum()
            pocket_size = np.random.choice(np.arange(len(pocket_dist)), p=pocket_dist)
        else:
            pocket_size = np.random.randint(20, 50)
        pocket_sizes.append(pocket_size)
    
    return ligand_sizes, pocket_sizes


def save_samples_to_graphs(samples: list[Data], num_samples: int, save_path: str):
    '''Save samples individually as graphs for visualization'''

    for i in range(num_samples):
        graph = Data(
            ligand_coords=samples['ligand_coords'][samples['ligand_mask']==i].cpu(),
            ligand_features=samples['ligand_features'][samples['ligand_mask']==i].cpu(),
            pocket_coords=samples['pocket_coords'][samples['pocket_mask']==i].cpu(),
            pocket_features=samples['pocket_features'][samples['pocket_mask']==i].cpu()
        )
        torch.save(graph, save_path.replace(".pt", f"_graph_{i}.pt"))


def sample_molecules(model: MolecularDenoisingModel,
                    dataset_path: str, 
                    num_steps: int = 50, 
                    schedule_type: str = "exponential", 
                    num_samples: int = 3):
    """Generate new molecules using the trained model with real pocket conditioning"""
    
    print(f"\n{'='*60}")
    print("SAMPLING NEW MOLECULES (CONDITIONAL)")
    print(f"{'='*60}")

    # Load real pockets from dataset
    pocket_data, ligand_sizes = load_pocket_data_from_dataset(
        dataset_path=dataset_path,
        num_samples=num_samples,
        model=model
    )
    
    print(f"Ligand sizes: {ligand_sizes}")
    print(f"Pocket sizes: {[len(pocket_data['pocket_mask'][pocket_data['pocket_mask']==i]) for i in range(num_samples)]}")
    
    sampler = create_molecular_sampler_from_model(
        model=model,
        ligand_sizes=ligand_sizes,
        pocket_data=pocket_data,
        num_steps=num_steps,
        schedule_type=schedule_type,
        return_full_paths=False
    )
    
    print("Sampling molecules conditioned on real binding pockets...")
    print(f"Using {num_steps} sampling steps...")
    
    # Generate samples
    samples = sampler.generate()
    
    print(f"\nGenerated molecules:")
    print(f"  Total ligand atoms: {samples['ligand_coords'].shape[0]}")
    print(f"  Total pocket residues: {samples['pocket_coords'].shape[0]}")
    print(f"  Ligand coordinate range: [{samples['ligand_coords'].min():.2f}, {samples['ligand_coords'].max():.2f}]")
    print(f"  Pocket coordinate range: [{samples['pocket_coords'].min():.2f}, {samples['pocket_coords'].max():.2f}]")
    
    # Check that categorical features are properly discretized 
    lig_feature_sums = samples['ligand_features'].sum(dim=1)
    pocket_feature_sums = samples['pocket_features'].sum(dim=1)
    
    print(f"  Ligand features properly one-hot: {torch.allclose(lig_feature_sums, torch.ones_like(lig_feature_sums))}")
    print(f"  Pocket features properly one-hot: {torch.allclose(pocket_feature_sums, torch.ones_like(pocket_feature_sums))}")
    
    # Show some statistics
    ligand_types = torch.argmax(samples['ligand_features'], dim=1)
    pocket_types = torch.argmax(samples['pocket_features'], dim=1)
    
    print(f"  Ligand atom type distribution: {torch.bincount(ligand_types)}")
    print(f"  Pocket residue type distribution: {torch.bincount(pocket_types)}")
    
    print(f"\n✅ Sampling completed successfully!")
    
    return samples


def evaluate_atom_aa_distributions(samples: list[Data]):
    """Evaluate the quality of sampled binding pockets
         - Jenson-Shannon divergence of atom distribution from training distribution
         - Jenson-Shannon divergence of amino acid distribution from training distribution
    """

    # Divergence of atom distribution from training distribution
    ligand_features = samples['ligand_features'].cpu()
    atom_distribution = ligand_features.sum(dim=0)
    training_atom_distribution = np.array([atom_freq_dist[atom] for atom in atom_decoder])
    atoms_dist_kl_divergence, atoms_dist_js_divergence = compare_distributions(atom_distribution.numpy(), training_atom_distribution)

    # Divergence of amino acid distribution from training distribution
    pocket_features = samples['pocket_features'].cpu()
    aa_distribution = pocket_features.sum(dim=0)
    training_aa_distribution = np.array([aa_freq_dist[aa] for aa in aa_decoder3])
    aa_dist_kl_divergence, aa_dist_js_divergence = compare_distributions(aa_distribution.numpy(), training_aa_distribution)

    return {
        'atoms_dist_kl_divergence': atoms_dist_kl_divergence, 
        'atoms_dist_js_divergence': atoms_dist_js_divergence, 
        'aa_dist_kl_divergence': aa_dist_kl_divergence,
        'aa_dist_js_divergence': aa_dist_js_divergence
        }


def evaluate_ring_size_distributions(ring_size_counts: Counter, ring_size_dist: dict):
    
    # Convert ring_size_counts Counter to array format matching ring_size_dist
    max_ring_size_from_dist = max(int(key) for key in ring_size_dist.keys())
    max_ring_size_from_counts = max(ring_size_counts.keys()) if ring_size_counts else 0
    max_ring_size = max(max_ring_size_from_dist, max_ring_size_from_counts)
    ring_size_array = np.zeros(max_ring_size + 1)
    
    for ring_size, count in ring_size_counts.items():
        ring_size_array[ring_size] = count
    
    # Convert ring_size_dist to array format
    ring_size_dist_array = np.zeros(max_ring_size + 1)
    for ring_size_str, count in ring_size_dist.items():
        ring_size = int(ring_size_str)
        if ring_size <= max_ring_size:
            ring_size_dist_array[ring_size] = count

    ring_size_dist_kl_divergence, ring_size_dist_js_divergence = compare_distributions(ring_size_array, ring_size_dist_array)

    return {
        'ring_size_dist_kl_divergence': ring_size_dist_kl_divergence,
        'ring_size_dist_js_divergence': ring_size_dist_js_divergence
    }


def uff_relax(mol, max_iter=200):
    """
    Uses RDKit's universal force field (UFF) implementation to optimize a
    molecule.
    """
    more_iterations_required = UFFOptimizeMolecule(mol, maxIters=max_iter)
    if more_iterations_required:
        warnings.warn(f'Maximum number of FF iterations reached. '
                      f'Returning molecule after {max_iter} relaxation steps.')
    return more_iterations_required


def process_molecule(rdmol, add_hydrogens=False, sanitize=True, relax_iter=0,
                     largest_frag=True):
    """
    Apply filters to an RDKit molecule. Makes a copy first.
    Args:
        rdmol: rdkit molecule
        add_hydrogens
        sanitize
        relax_iter: maximum number of UFF optimization iterations
        largest_frag: filter out the largest fragment in a set of disjoint
            molecules
    Returns:
        RDKit molecule or None if it does not pass the filters
    """

    # Create a copy
    mol = Chem.Mol(rdmol)

    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            warnings.warn('Sanitization failed. Returning None.')
            return None

    if add_hydrogens:
        mol = Chem.AddHs(mol, addCoords=(len(mol.GetConformers()) > 0))

    if largest_frag:
        mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        if sanitize:
            # sanitize the updated molecule
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                return None

    if relax_iter > 0:
        if not UFFHasAllMoleculeParams(mol):
            warnings.warn('UFF parameters not available for all atoms. '
                          'Returning None.')
            return None

        try:
            uff_relax(mol, relax_iter)
            if sanitize:
                # sanitize the updated molecule
                Chem.SanitizeMol(mol)
        except (RuntimeError, ValueError) as e:
            return None

    return mol



def make_mol_openbabel(positions, atom_types, atom_decoder):
    """
    Build an RDKit molecule using openbabel for creating bonds
    Args:
        positions: N x 3
        atom_types: N
        atom_decoder: maps indices to atom types
    Returns:
        rdkit molecule
    """
    atom_types = [atom_decoder[x] for x in atom_types]

    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name

        # Write xyz file
        utils.write_xyz_file(positions, atom_types, tmp_file)

        # Convert to sdf file with openbabel
        # openbabel will add bonds
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")
        ob_mol = openbabel.OBMol()
        obConversion.ReadFile(ob_mol, tmp_file)

        obConversion.WriteFile(ob_mol, tmp_file)

        # Read sdf file with RDKit
        tmp_mol = Chem.SDMolSupplier(tmp_file, sanitize=False)[0]

    # Build new molecule. This is a workaround to remove radicals.
    mol = Chem.RWMol()
    for atom in tmp_mol.GetAtoms():
        mol.AddAtom(Chem.Atom(atom.GetSymbol()))
    mol.AddConformer(tmp_mol.GetConformer(0))

    for bond in tmp_mol.GetBonds():
        mol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                    bond.GetBondType())

    return mol



def build_molecule(positions, atom_types, add_coords=False,
                   use_openbabel=True):
    """
    Build RDKit molecule
    Args:
        positions: N x 3
        atom_types: N
        dataset_info: dict
        add_coords: Add conformer to mol (always added if use_openbabel=True)
        use_openbabel: use OpenBabel to create bonds
    Returns:
        RDKit molecule
    """
    if use_openbabel:
        mol = make_mol_openbabel(positions, atom_types, atom_decoder)
    else:
        mol = make_mol_edm(positions, atom_types, dataset_info, add_coords)

    return mol



def build_mol_objects(samples, sanitize=False, relax_iter=0, largest_frag=False):
    molecules = []
    x = samples['ligand_coords'].detach().cpu()
    atom_type = samples['ligand_features'].argmax(1).detach().cpu()
    lig_mask = samples['ligand_mask'].cpu()
    # x = xh_lig[:, :self.x_dims].detach().cpu()
    # atom_type = xh_lig[:, self.x_dims:].argmax(1).detach().cpu()
    #lig_mask = lig_mask.cpu()

    molecules = []
    for mol_pc in zip(utils.batch_to_list(x, lig_mask),
                        utils.batch_to_list(atom_type, lig_mask)):

        mol = build_molecule(*mol_pc, add_coords=True)
        mol = process_molecule(mol,
                                add_hydrogens=False,
                                sanitize=sanitize,
                                relax_iter=relax_iter,
                                largest_frag=largest_frag)
        if mol is not None:
            molecules.append(mol)

    return molecules



def evaluate_mols(mols):
    """
    Takes a list of RDKit Mol objects.
    Returns a dictionary with summary statistics about molecule integrity,
    including ring size distributions.
    """
    summary = {
        "num_molecules": len(mols),
        "valid_molecules": 0,
        "invalid_molecules": 0,
        # "invalid_molecules_indices": [],
        "num_multifragment": 0,
        # "multifragment_indices": [],
        "num_disconnected_atoms": 0,
        # "molecules_with_disconnected_atoms": [],
        "total_valence_issues": 0,
        "molecules_with_valence_issues": [],
        "max_num_fragments": 0,
        "sum_num_fragments": 0,
        "sum_num_valence_issues": 0,
        "num_molecules_with_rings": 0,
        "sum_num_rings": 0,
        "max_num_rings": 0,
        "ring_size_counts": Counter(),
    }

    pt = Chem.GetPeriodicTable()

    for idx, mol in enumerate(mols):
        # 1. Validity:Sanitization
        sanitized = True
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            sanitized = False
        if sanitized:
            summary["valid_molecules"] += 1
        else:
            summary["invalid_molecules"] += 1
            # summary["invalid_molecules_indices"].append(idx)

        # 2. Fragments
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        num_frags = len(frags)
        summary["sum_num_fragments"] += num_frags
        if num_frags > 1:
            summary["num_multifragment"] += 1
            # summary["multifragment_indices"].append(idx)
        summary["max_num_fragments"] = max(summary["max_num_fragments"], num_frags)

        # 3. Disconnected atoms
        has_disconnected = any(len(atom.GetNeighbors()) == 0 for atom in mol.GetAtoms())
        if has_disconnected:
            summary["num_disconnected_atoms"] += 1
            # summary["molecules_with_disconnected_atoms"].append(idx)

        # 4. Valence issues
        mol_valence_issues = 0
        for atom in mol.GetAtoms():
            try:
                valence = atom.GetTotalValence()
                default_valence = pt.GetDefaultValence(atom.GetSymbol())
                if valence > default_valence + 1:
                    mol_valence_issues += 1
            except:
                mol_valence_issues += 1
        summary["total_valence_issues"] += mol_valence_issues
        summary["sum_num_valence_issues"] += mol_valence_issues
        if mol_valence_issues > 0:
            summary["molecules_with_valence_issues"].append(idx)

        # 5. Rings and ring sizes
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        num_rings = len(atom_rings)
        summary["sum_num_rings"] += num_rings
        if num_rings > 0:
            summary["num_molecules_with_rings"] += 1
        summary["max_num_rings"] = max(summary["max_num_rings"], num_rings)

        # Count ring sizes
        for ring in atom_rings:
            ring_size = len(ring)
            summary["ring_size_counts"][ring_size] += 1

    # 6. Means
    n = len(mols)
    summary["mean_num_fragments"] = summary["sum_num_fragments"] / n if n else 0
    summary["mean_num_valence_issues"] = summary["sum_num_valence_issues"] / n if n else 0
    summary["mean_num_rings"] = summary["sum_num_rings"] / n if n else 0

    # 7. Ring size distributions
    ring_size_dist_metrics = evaluate_ring_size_distributions(summary["ring_size_counts"], ring_size_dist)
    summary.update(ring_size_dist_metrics)

    # Convert ring_size_counts to regular dict for JSON compatibility
    summary["ring_size_counts"] = dict(summary["ring_size_counts"])

    return summary


def test_sample_molecules():
    """Test the sample_molecules function with real data"""
    
    print("Testing sample_molecules function...")
    
    # Create a simple test model
    model = MolecularDenoisingModel(
        atom_nf=10,
        residue_nf=21,
        joint_nf=16,
        hidden_nf=64,
        n_layers=3
    )
    model.initialize()
    
    # Test the function
    dataset_path = 'cleansplit_ood_train_combined.pt'
    
    try:
        samples = sample_molecules(
            model=model,
            dataset_path=dataset_path,
            num_steps=10,
            num_samples=2
        )
        
        print("\n✅ sample_molecules() executed successfully!")
        print(f"  Returned keys: {samples.keys()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    import torch.nn.functional as F
    test_sample_molecules()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", required=True, type=str)
#     parser.add_argument("--num_steps", type=int, default=25)
#     parser.add_argument("--n_samples", type=int, default=10)
#     args = parser.parse_args()
    
#     model = load_checkpoint(args.model_path)
#     samples = sample_molecules(model, num_steps=args.num_steps, num_samples=args.n_samples)

#     molecules = build_mol_objects(samples)

#     for key, value in evaluate_mols(molecules).items():
#         print(f'{key}: {value}')

#     utils.write_sdf_file(sdf_path='samples/0730_110059_bayesian_trial_1_epoch_599.sdf', molecules=molecules)
    
#     save_samples_to_graphs(samples, args.n_samples, args.model_path)


# # Save the samples to a json file, making tensors serializable
# samples_dict = {}
# for key, value in samples.items():
#     if isinstance(value, torch.Tensor):
#         samples_dict[key] = value.cpu().detach().numpy().tolist()
#     else:
#         samples_dict[key] = value
# with open('samples.json', 'w') as f:
#     json.dump(samples_dict, f, indent=4)

# for i in range(args.n_samples):
#     ligand_coords=samples['ligand_coords'][samples['ligand_mask']==i].cpu(),
#     ligand_features=samples['ligand_features'][samples['ligand_mask']==i].cpu(),
#     pocket_coords=samples['pocket_coords'][samples['pocket_mask']==i].cpu(),
#     pocket_features=samples['pocket_features'][samples['pocket_mask']==i].cpu()

#     print(f'Ligand coords: {ligand_coords.shape}')
#     print(f'Ligand features: {ligand_features.shape}')
#     print()
#     print(ligand_coords)
#     print(ligand_features)

# def analyze_sample(self, molecules, atom_types, aa_types, receptors=None):
#     # Distribution of node types
#     kl_div_atom = self.ligand_type_distribution.kl_divergence(atom_types) \
#         if self.ligand_type_distribution is not None else -1
#     kl_div_aa = self.pocket_type_distribution.kl_divergence(aa_types) \
#         if self.pocket_type_distribution is not None else -1

#     # Convert into rdmols
#     rdmols = [build_molecule(*graph, self.dataset_info) for graph in molecules]

#     # Other basic metrics
#     (validity, connectivity, uniqueness, novelty), (_, connected_mols) = \
#         self.ligand_metrics.evaluate_rdmols(rdmols)

#     qed, sa, logp, lipinski, diversity = \
#         self.molecule_properties.evaluate_mean(connected_mols)

#     out = {
#         'kl_div_atom_types': kl_div_atom,
#         'kl_div_residue_types': kl_div_aa,
#         'Validity': validity,
#         'Connectivity': connectivity,
#         'Uniqueness': uniqueness,
#         'Novelty': novelty,
#         'QED': qed,
#         'SA': sa,
#         'LogP': logp,
#         'Lipinski': lipinski,
#         'Diversity': diversity
#     }

#     # Simple docking score
#     if receptors is not None:
#         # out['smina_score'] = np.mean(smina_score(rdmols, receptors))
#         out['smina_score'] = np.mean(smina_score(connected_mols, receptors))

#     return out
