import torch
import numpy as np
from torch_geometric.data import Data
from molecular_diffusion import MolecularDenoisingModel
import argparse
from molecular_samplers import create_molecular_sampler_from_model
from molecular_diffusion import (
    log_uniform_sampling,
    edm_weighting,
    MolecularDiffusion,
    exponential_noise_schedule
)


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
            num_steps: int = 50, 
            schedule_type: str = "exponential", 
            num_samples: int = 3
            ):

    """Generate new molecules using the trained model"""
    
    print(f"\n{'='*60}")
    print("SAMPLING NEW MOLECULES")
    print(f"{'='*60}")

    # Create sampler
    ligand_sizes, pocket_sizes = sample_lig_pocket_sizes(num_samples)
    print(f"Ligand sizes: {ligand_sizes}")
    print(f"Pocket sizes: {pocket_sizes}")
    
    sampler = create_molecular_sampler_from_model(
        model=model,
        ligand_sizes=ligand_sizes,
        pocket_sizes=pocket_sizes,
        num_steps=num_steps,
        schedule_type=schedule_type,
        return_full_paths=False
    )
    
    print("Sampling unconditional molecules...")
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


def evaluate_samples(samples: list[Data]):
    """Evaluate the quality of sampled binding pockets
         - Jenson-Shannon divergence of atom distribution from training distribution
         - Jenson-Shannon divergence of amino acid distribution from training distribution
    """

    # Divergence of atom distribution from training distribution
    ligand_features = samples['ligand_features'].cpu()
    atom_distribution = ligand_features.sum(dim=0)
    training_atom_distribution = np.array([atom_freq_dist[atom] for atom in atom_decoder])
    _, atoms_dist_js_divergence = compare_distributions(atom_distribution.numpy(), training_atom_distribution)

    # Divergence of amino acid distribution from training distribution
    pocket_features = samples['pocket_features'].cpu()
    aa_distribution = pocket_features.sum(dim=0)
    training_aa_distribution = np.array([aa_freq_dist[aa] for aa in aa_decoder3])
    _, aa_dist_js_divergence = compare_distributions(aa_distribution.numpy(), training_aa_distribution)

    return {
        'atoms_dist_js_divergence': atoms_dist_js_divergence, 
        'aa_dist_js_divergence': aa_dist_js_divergence
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--n_samples", type=int, default=3)
    args = parser.parse_args()
    
    model = load_checkpoint(args.model_path)

    samples = sample_molecules(model, args.n_samples)

    save_samples_to_graphs(samples, args.n_samples, args.model_path)