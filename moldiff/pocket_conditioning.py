import torch
import dataclasses
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import wandb
from typing import List, Dict
from moldiff.Dataset import PDBbind_Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from datetime import datetime

# Import our molecular modules
from moldiff.molecular_diffusion import (
    MolecularDenoisingModel, 
    exponential_noise_schedule,
    log_uniform_sampling,
    edm_weighting,
    MolecularDiffusion, 
    remove_mean_batch
)
from moldiff.metrics import (
    load_checkpoint,
    sample_molecules, 
    evaluate_atom_aa_distributions
)
from moldiff.egnn_dynamics import EGNNDynamics
import torch.nn.functional as F


# dataclass(frozen=True, kw_only=True)
class ConditionalMolecularDenoisingModel(MolecularDenoisingModel): 
    
    def __post_init__(self):
        """Initializes the noiser and denoiser components"""
        super().__post_init__()

        # TODO ask victor whether it even makes sense to freeze them (I would have to take from pretrained), 
        # or if I can still learn them even though I don't freeze at all
        # # freeze pocket encoder to lookup embeddings if using pocket conditioning
        # for param in self.denoiser.residue_encoder.parameters():
        #         param.requires_grad = False
    def los

