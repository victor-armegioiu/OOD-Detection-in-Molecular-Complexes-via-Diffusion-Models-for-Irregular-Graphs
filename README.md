# Molecular Diffusion OOD Detection

This repository contains the implementation for the paper **"Out-of-Distribution Detection in Molecular Complexes via Diffusion Models for Irregular Graphs"**. We introduce a novel framework that combines denoising diffusion probabilistic models (DDPMs) with probability flow ODE trajectory analysis to detect out-of-distribution molecular complexes. Our approach addresses the complexity bias inherent in pure likelihood-based OOD detection by augmenting log-likelihood scores with 18 geometric trajectory features extracted from the diffusion process.

The framework operates on 3D protein-ligand interaction graphs, which combine discrete chemical features (atom/residue types) with continuous 3D coordinates. We unify these attributes through a continuous spherical embedding space, enabling end-to-end training via a single SE(3)-equivariant graph neural network.

**Paper**: [https://arxiv.org/abs/2512.18454](https://arxiv.org/abs/2512.18454)<br> 
**📊 Data and pre-trained models**: [https://doi.org/10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)<br>
**📊 Source Data**: [https://www.pdbbind-plus.org.cn/](https://www.pdbbind-plus.org.cn/)

## Repository Structure

### Diffusion Model Components

#### [`egnn_dynamics.py`](egnn_dynamics.py)
Implements the SE(3)-equivariant graph neural network (EGNN) architecture that serves as the denoising network. Key features:
- **EGNNDynamics**: Core denoising network that predicts clean coordinates and chemical type probabilities
- **SE(3) equivariance**: Maintains rotational and translational invariance
- **Unified coordinate-feature processing**: Handles both continuous 3D coordinates and discrete chemical embeddings

#### [`molecular_diffusion.py`](molecular_diffusion.py)
Contains the complete molecular diffusion framework combining continuous coordinates with discrete features:
- **MolecularDiffusion**: Main diffusion scheme with variance exploding formulation
- **Noise schedules**: Exponential noise scheduling optimized for molecular data
- **Posterior mean interpolation**: Maps discrete chemical features to continuous embeddings via learnable prototypes
- **Loss computation**: Implements coordinate MSE loss and categorical cross-entropy loss with proper weighting

#### [`molecular_samplers.py`](molecular_samplers.py)
Implements SDE-based sampling algorithms for generating molecular structures:
- **Probability flow ODE integration**: Deterministic sampling via neural ODEs
- **Euler-Maruyama stepping**: Stochastic differential equation solvers
- **Molecular state handling**: Batch processing of ligand-pocket complexes with proper masking
- **Center-of-mass constraints**: Maintains molecular geometry during sampling

### Training

#### [`main.py`](main.py)
Complete training pipeline with hyperparameter optimization support:
- **Training loop**: Implements the full molecular diffusion training procedure
- **Multiple optimization modes**: Single runs, grid search, random search, and Bayesian optimization
- **Evaluating samples at regular intervals**: Molecular validity, fragment and ring counts, ring sizes etc.
- **Checkpointing**: Model saving/loading with training state restoration
- **Wandb integration**: Experiment tracking and visualization

### OOD Detection & Likelihood Computation

#### [`enhanced_likelihood.py`](enhanced_likelihood.py)
Core OOD detection implementation with trajectory analysis:
- **MolecularLikelihoodEvaluator**: Computes exact log-likelihoods via probability flow ODEs
- **TrajectoryStatistics**: Collects 18 geometric features during forward integration:
  - Vector field magnitudes and distributions
  - Trajectory curvature and smoothness measures  
  - Path efficiency and tortuosity
  - Lipschitz stability estimates
  - Flow energy and coupling correlations
- **Hutchinson trace estimation**: Efficient divergence computation for likelihood evaluation
- **Gradient checkpointing**: Memory-efficient integration for large molecular complexes

### Dataset Preparation

#### [`protlig_encoder.py`](protlig_encoder.py)
Converts raw protein-ligand structures into graph representations:
- **Multi-format support**: Processes PDB, SDF, and MOL2 files
- **Graph construction**: Creates PyTorch Geometric Data objects with node features, edges, and 3D positions


#### [`Dataset.py`](Dataset.py)
PyTorch Dataset class for molecular graph data:
- **PDBbind_Dataset**: Loads and processes graphs, combining them into PyTorch datasets
- **Data splitting**: Supports train/test splits via JSON configuration files

#### [`data_split/combined_train_test_split.json`](data_split/combined_train_test_split.json)
Contains the carefully constructed train-test split with CASF2016 and seven OOD datasets with isolated protein families:
- **Family-based exclusion**: OOD test sets formed by excluding entire protein families
- **Seven OOD test sets**: `1nvq_ood_test`, `1sqa_ood_test`, `2p15_ood_test`, etc.
- **Training data**: `train` In-distribution complexes for diffusion model training

#### [`data_split/combined_train_val_split.json`](data_split/combined_train_val_split.json)
Further splits the `train` dataset from [`data_split/combined_train_test_split.json`](data_split/combined_train_test_split.json) into a final training and validation dataset with a 90/10 ratio.

## Reproduction Instructions

### 1. Data Preparation

**Option A: Download pre-processed Datasets**

Download the pre-processed datasets from Zenodo and place them in your working directory:
```bash
# Training and validation sets
wget https://zenodo.org/record/XXXXXXX/files/cleansplit_ood_train_combined.pt
wget https://zenodo.org/record/XXXXXXX/files/cleansplit_ood_val_combined.pt

# OOD test sets  
wget https://zenodo.org/record/XXXXXXX/files/cleansplit_1nvq_test.pt
wget https://zenodo.org/record/XXXXXXX/files/cleansplit_1sqa_test.pt
# ... additional test sets
```

**Option B: Process Raw Data from Scratch**

If you have access to the raw PDBbind database:

1. **Convert structures to graphs:**
```bash
python protlig_encoder.py \
    --protein_source /path/to/dir/with/PDB/files \
    --ligand_source /path/to/dir/with/SDF/MOL2/files \
    --output_dir processed_graphs \
    --granularity residue-level-fully-connected
```

2. **Create datasets with train-test splits:**
```bash
# Training set
python Dataset.py \
    --data_dir processed_graphs \
    --save_path dataset_train.pt \
    --data_split data_split/combined_train_val_split.json \
    --dataset train

# Validation set  
python Dataset.py \
    --data_dir processed_graphs \
    --save_path dataset_val.pt \
    --data_split data_split/combined_train_val_split.json \
    --dataset validation

# OOD test sets (example for 1nvq dataset)
python Dataset.py \
    --data_dir processed_graphs \
    --save_path dataset_1nvq_test.pt \
    --data_split data_split/combined_train_test_split.json \
    --dataset 1nvq_ood_test
```

### 2. Training the Diffusion Model

**Basic training:**
```bash
python main.py \
    --mode single \
    --train_dataset cleansplit_ood_train_combined.pt \
    --eval_dataset cleansplit_ood_val_combined.pt \
    --batch_size 16 \
    --learning_rate 0.0001

# To resume training from a saved checkpoint
    --resume_checkpoint_path training_runs/.../checkpoint_epoch_990.pt
```

**Advanced training with hyperparameter optimization:**
```bash
# Bayesian optimization
python main.py \
    --mode bayesian \
    --num_trials 50 \
    --train_dataset cleansplit_ood_train_combined.pt \
    --eval_dataset cleansplit_ood_val_combined.pt \
```

**Key hyperparameters:**
- `--n_layers`: Number of EGNN layers (default: 6)
- `--joint_nf`: Joint embedding dimension (default: 256) 
- `--hidden_nf`: Hidden layer dimension (default: 256)
- `--edge_embedding_dim`: Size of the edge embedding (default: 64)
- `--num_sampling_steps`: Diffusion sampling steps (default: 400)
- `--learning_rate`: (default: 0.0001)
- `--batch_size`: Size of graph batches (default: 16)
- `--early_stopping_patience`: Number of epochs without improvement before stopping training(default:100)

Training checkpoints are saved in `training_runs/` with automatic experiment naming.

### 3. Computing Likelihoods and Trajectory Features

Once you have a trained model, compute log-likelihoods and trajectory statistics for any dataset:

**Basic usage:**
```bash
python enhanced_likelihood.py \
    --dataset_path cleansplit_1nvq_test.pt \
    --checkpoint_path training_runs/your_run/checkpoint_epoch_999.pt \
    --results_folder likelihood_results \
    --num_steps 5 \
    --num_hutchinson_samples 20
```

**Parameters:**
- `--num_steps`: Number of integration steps for probability flow ODE (higher = more accurate)
- `--num_hutchinson_samples`: Monte Carlo samples for trace estimation (higher = less variance)

**Output:**
The script generates a JSON file containing:
- `log_likelihood`: Exact log-likelihood for each molecular complex
- `trajectory_features`: 18-dimensional feature vectors including:
  - Vector field statistics (mean, std, max magnitudes)
  - Path efficiency and tortuosity measures
  - Curvature and smoothness indicators
  - Flow energy and Lipschitz stability estimates

### Example OOD Detection Workflow

1. **Train model on in-distribution data:**
```bash
python main.py --mode single --train_dataset cleansplit_ood_train_combined.pt --eval_dataset cleansplit_ood_val_combined.pt --num_epochs 1500
```

2. **Compute likelihoods for test sets:**
```bash
# In-distribution validation set
python enhanced_likelihood.py --dataset_path cleansplit_ood_val_combined.pt --checkpoint_path training_runs/latest/checkpoint.pt --results_folder results

# OOD test sets
for dataset in cleansplit_1nvq_test.pt cleansplit_1sqa_test.pt cleansplit_2p15_test.pt; do
    python enhanced_likelihood.py --dataset_path $dataset --checkpoint_path training_runs/latest/checkpoint.pt --results_folder results
done
```

3. **Analyze results:**
The output JSON files contain likelihood scores and trajectory features that can be used to train binary OOD classifiers or compute ROC curves for evaluation.

## Requirements

- Python 3.8+
- PyTorch 1.12+ with CUDA support
- PyTorch Geometric
- Additional dependencies: numpy, scipy, matplotlib, wandb, optuna

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{graber2026outofdistributiondetectionmolecularcomplexes,
      title={Out-of-Distribution Detection in Molecular Complexes via Diffusion Models for Irregular Graphs}, 
      author={David Graber and Victor Armegioiu and Rebecca Buller and Siddhartha Mishra},
      year={2026},
      eprint={2512.18454},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.18454}, 
}
```
