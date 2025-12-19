import os
import numpy as np
import torch
import json
import glob
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union
from torch_geometric.data import Dataset, Data


class PDBbind_Dataset(Dataset):
    """A Dataset class for protein-ligand interaction graphs.

    This class represents a Dataset for protein-ligand interaction graphs, loading and processing
    graph data from a specified folder. It supports data splitting, inference mode, and various
    ablation study options.

    Attributes:
        data_dir (str): Directory containing the graph files
        data_split (Optional[str]): Path to the JSON file containing dataset splits
        dataset (Optional[str]): Which subset to load ('train' or 'test')
        input_data (Dict[int, Data]): Dictionary mapping indices to processed graph data

    Features:
        - Loads all graphs from a specified folder
        - Supports data splitting through an external JSON dictionary
        - Processes graphs with position and feature information
        - Command-line interface for dataset construction
    """


    def __init__(
        self,
        root: str,
        data_split: Optional[str] = None,
        dataset: Optional[str] = None,
        include_sidechains: Optional[bool] = False
    ) -> None:

        """
        Initialize the PDBbind Dataset.
        Args:
            root: Path to the folder containing the graphs
            data_split: Filepath to dictionary (JSON file) containing the data split
            dataset: Which subset to load ('train' or 'test') as defined in data_split
        """
        self.include_sidechains = include_sidechains
        self.logger = logging.getLogger(__name__)
        self.data_dir = root
        self.data_split = data_split
        self.dataset = dataset
        
        super().__init__(root)

        self._indices = None
        self.transform = None

        # Initialize the dataset
        self.filepaths = self._get_file_paths()
        self.logger.info(f"Number of graphs loaded: {len(self.filepaths)}")
        
        # Process all graphs
        self.input_data = self._process_graphs()

        



    def _get_file_paths(self) -> List[str]:

        """
        Get paths of all graph files to be included in the dataset.
        Returns: List of file paths to process
        """

        if self.data_split:
            if not os.path.exists(self.data_split):
                raise FileNotFoundError(f"Data split file not found: {self.data_split}")
            
            with open(self.data_split, 'r', encoding='utf-8') as json_file:
                split_dict = json.load(json_file)

            if self.dataset not in split_dict:
                raise ValueError(f"Dataset '{self.dataset}' not found in split dictionary")

            included_complexes = split_dict[self.dataset]
            dataset_filepaths = []
            for id in included_complexes:
                search_pattern = os.path.join(self.data_dir, f"{id.upper()}*.pt") #TEMPORARILY ADDED .UPPER() DUE TO MISATO UPPER CASE KEYS
                matching_files = glob.glob(search_pattern)
                dataset_filepaths.extend(matching_files)
            return dataset_filepaths
        
        return [str(f) for f in Path(self.data_dir).glob("*.pt")]



    def _process_graphs(self) -> Dict[int, Data]:

        """
        Process all graphs in the dataset.
        Returns: Dictionary mapping indices to processed graph data
        """

        processed_data = {}
        lig_nf = 10
        prot_nf = 21

        if not self.include_sidechains:

            for idx, file in enumerate(self.filepaths):
                self.logger.info(f"Processing graph {file.split('/')[-1]}: {idx} of {len(self.filepaths)}")

                graph = torch.load(file, weights_only=False)
                complex_id = os.path.basename(file).replace('.pt', '')

                # Combine position and feature information
                lig_coords = graph.pos[graph.lig_mask]
                prot_coords = graph.pos[graph.prot_mask]
                lig_features = graph.x[graph.lig_mask][:, :lig_nf]
                prot_features = graph.x[graph.prot_mask][:, -prot_nf:]

                processed_data[idx] = Data(
                                            lig_coords=lig_coords.float(),
                                            lig_features=lig_features.float(),
                                            prot_coords=prot_coords.float(),
                                            prot_features=prot_features.float(),
                                            id=complex_id
                                            )
        else:
            for idx, file in enumerate(self.filepaths):
                self.logger.info(f"Processing graph {file.split('/')[-1]}: {idx} of {len(self.filepaths)}")

                graph = torch.load(file, weights_only=False)
                complex_id = os.path.basename(file).replace('.pt', '')

                # Combine position and feature information
                lig_coords = graph.pos[graph.lig_mask]
                prot_coords = graph.pos[graph.prot_mask & graph.ca_mask]
                lig_features = graph.x[graph.lig_mask][:, :lig_nf]
                prot_features = graph.x[graph.prot_mask & graph.ca_mask][:, 2*lig_nf:]
                prot_sidechain_coords = graph.pos[graph.prot_mask]
                prot_sidechain_features = graph.x[graph.prot_mask][:, lig_nf:2*lig_nf] # here we only want atom features

                processed_data[idx] = Data(
                                            lig_coords=lig_coords.float(),
                                            lig_features=lig_features.float(),
                                            prot_coords=prot_coords.float(),
                                            prot_features=prot_features.float(),
                                            prot_sidechain_coords=prot_sidechain_coords.float(), 
                                            prot_sidechain_features=prot_sidechain_features.float(),
                                            id=complex_id
                                            )
            

        return processed_data


    def __len__(self) -> int:
        return len(self.input_data)

    def len(self) -> int:
        return len(self.input_data)

    def get(self, idx: int) -> Data:
        return self.input_data[idx]

    @classmethod
    def create_sliced_dataset(cls, input_data: Dict[int, Data]) -> 'PDBbind_Dataset':
        """Create a sliced dataset from input_data dictionary."""
        # Create new instance without calling __init__
        instance = cls.__new__(cls)
        
        # Initialize PyTorch Geometric attributes
        instance._indices = None
        instance.transform = None
        instance.input_data = input_data
        
        # Set minimal required attributes
        instance.data_dir = ""
        instance.data_split = None
        instance.dataset = None
        instance.logger = logging.getLogger(__name__)
        
        return instance



def main() -> None:
    """Command-line interface for dataset construction."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Construct a dataset from a directory containing graph Data() objects.")
    parser.add_argument("--data_dir", required=True,
        help="The path to the folder containing all Data() objects (graphs) of the dataset.")    
    parser.add_argument('--save_path', required=True, type=str,
        help='Path to save the dataset ending with .pt')
    parser.add_argument("--data_split", default=None,
        help="Filepath to dictionary (json file) containing the data split for the graphs in the folder")
    parser.add_argument("--dataset", default=None,
        help="If a split dict is given, which subset to load ['train', 'test']")
    parser.add_argument("--sidechains", type=lambda x: True if x=="True" else False)

    args = parser.parse_args()

    if not args.save_path.endswith('.pt'):
        logger.error("Save path must end with .pt")
        return

    try:
        dataset = PDBbind_Dataset(
            args.data_dir,
            data_split=args.data_split,
            dataset=args.dataset,
            include_sidechains=args.sidechains
        )
        
        torch.save(dataset, args.save_path)
        logger.info(f"Dataset successfully saved to {args.save_path}")
        
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        raise


if __name__ == "__main__":
    main() 