"""
Distribution Comparison Tool

This script loads .json metrics files from a directory and compares the distributions of a given metric between the files
visually and statistically. It provides:
- Histogram plots of all distributions
- Statistical metrics (mean, std, skewness, kurtosis)
- Pairwise distribution differences (KL divergence, Wasserstein distance)
- Summary statistics in the plot
- Optional if error dictionary is provided, it will plot the metric vs the error values
- Optional preprocessing: dataset merging, clipping, outlier normalization, and range normalization
- Optional export: save processed data as CSV file with value-label pairs
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.metrics import mutual_info_score
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare distributions from .json metrics files')

    # Required arguments
    parser.add_argument('directory', type=str, help='Directory containing .json files')
    parser.add_argument('--metric', type=str, required=True, help='Metric to compare')

    # Export options
    parser.add_argument('--save_plot', '-s', 
                        help='Path to save the comparison plot')
    parser.add_argument('--export', 
                        help='Path to export processed data as CSV file')
    parser.add_argument('--figsize', nargs=2, type=int, default=[20, 16], 
                        help='Figure size as [width, height]')

    # Preprocessing options
    parser.add_argument('--clip_percentiles', nargs=2, type=float, 
                        help='Clip data to specified percentiles (e.g., 1 99 for 1st to 99th percentile)')
    parser.add_argument('--cut_outliers', action='store_true',
                        help='Cut outliers instead of clipping them')
    parser.add_argument('--remove_outliers', action='store_true', help='Remove outliers using IQR method')
    parser.add_argument('--robust_scaling', action='store_true', help='Normalize outliers using robust scaling (brings outliers closer to main distribution)')
    parser.add_argument('--normalize_range', nargs=2, type=float,
                        help='Normalize entire distributions to specified range (e.g., 0 1 for [0,1] range)')
    parser.add_argument('--merge_patterns', nargs='+', type=str,
                        help='Patterns to merge datasets (e.g., train validation will merge datasets containing these words)')
    parser.add_argument('--error_dict', type=str, default=None,
                        help='Path to the error distribution file mapping IDs to error values')
    parser.add_argument('--plot_heatmaps', action='store_true',
                        help='Create individual heatmap plots for each distribution')

    
    return parser.parse_args()


class DistributionComparator:
    """A class to compare multiple distributions from .json metrics files."""
    
    def __init__(self, 
                directory_path: str, 
                metric: str,
                clip_percentiles: Optional[Tuple[float, float]] = None, 
                remove_outliers: bool = False,
                cut_outliers: bool = False,
                robust_scaling: bool = False,
                normalize_range: Optional[Tuple[float, float]] = None,
                merge_patterns: Optional[List[str]] = None):
        """
        Initialize the comparator with a directory path.
        
        Args:
            directory_path: Path to directory containing .json files
            clip_percentiles: Tuple of (lower_percentile, upper_percentile) for clipping, e.g., (1, 99)
            cut_outliers: Whether to cut outliers instead of clipping them
            robust_scaling: Whether to normalize outliers using robust scaling
            normalize_range: Tuple of (min_val, max_val) to normalize distributions to, e.g., (0, 1)
            merge_patterns: List of patterns to merge datasets (e.g., ['train', 'validation'] will merge datasets containing these words)
        """

        self.directory_path = Path(directory_path)
        self.metric = metric
        self.clip_percentiles = clip_percentiles
        self.cut_outliers = cut_outliers
        self.remove_outliers = remove_outliers
        self.robust_scaling = robust_scaling
        self.normalize_range = normalize_range
        self.merge_patterns = merge_patterns or []
        self.distributions = {}
        self.distributions_ids = {}  # Store distribution IDs aligned with distributions
        self.error_values = {}
        self.original_distributions = {}  # Store original data for reference
        self.metrics = {}
        self.pairwise_metrics = {}

        print(f"\nComparing distributions in {self.directory_path}")
        print(f"  Metric: {self.metric}")
        print(f"  Clip percentiles: {self.clip_percentiles}")
        print(f"  Cut outliers: {self.cut_outliers}")
        print(f"  Robust scaling: {self.robust_scaling}")
        print(f"  Normalize range: {self.normalize_range}")
        print(f"  Merge patterns: {self.merge_patterns}")

        
    def preprocess_distribution(self, data: np.ndarray, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess a single distribution with clipping and/or normalization.
        Args:
            data: Input distribution data
            ids: Input distribution IDs (must be same length as data)
        Returns:
            Tuple of (processed_data, processed_ids)
        """
        if len(data) != len(ids):
            raise ValueError(f"Data and IDs must have the same length: {len(data)} vs {len(ids)}")
            
        processed_data = data.copy()
        processed_ids = ids.copy()
        

        # Step 1: Removing the outliers with IQR method
        if self.remove_outliers:
            median = np.median(processed_data)    
            q75, q25 = np.percentile(processed_data, [75, 25])
            iqr = q75 - q25
    
            if iqr > 0:  # Avoid division by zero
                # Define outlier bounds (1.5 * IQR rule)
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                print(f"  Outlier bounds: [{lower_bound:.2e}, {upper_bound:.2e}]")

                if self.cut_outliers:
                    # CUTTING THE OUTLIERS - need to remove corresponding IDs too
                    valid_mask = (processed_data < upper_bound) & (processed_data > lower_bound)
                    processed_data = processed_data[valid_mask]
                    processed_ids = processed_ids[valid_mask]
                    print(f"  Cut off outliers outside the range [{lower_bound:.2e}, {upper_bound:.2e}]")
                else:
                    # CLIPPING THE OUTLIERS
                    processed_data = np.clip(processed_data, lower_bound, upper_bound)
                    print(f"  Clipped to [{lower_bound}, {upper_bound}]")
            else:
                print(f"  No outliers removed")


        # Step 2: Apply robust scaling
        if self.robust_scaling:
            # Use median and IQR for robust scaling
            median = np.median(processed_data)
            q75, q25 = np.percentile(processed_data, [75, 25])
            iqr = q75 - q25
            
            if iqr > 0:  # Avoid division by zero
                # Define bounds
                lower_bound = q25 - 0.5 * iqr
                upper_bound = q75 + 0.5 * iqr
                
                mask = (processed_data < lower_bound) | (processed_data > upper_bound)
                indices = np.where(mask)[0]
                
                if len(indices) > 0:
                    # Normalize outliers by bringing them closer to the main distribution
                    # Use a sigmoid-like transformation to smoothly bring outliers in
                    data_to_scale = processed_data[mask]
                    
                    # Calculate how far outliers are from bounds
                    distances = np.where(data_to_scale < lower_bound, 
                                       lower_bound - data_to_scale,
                                       data_to_scale - upper_bound)
                    
                    # Apply a smooth transformation to bring outliers closer
                    # Use a factor that reduces the distance by 99%
                    reduction_factor = 0.1
                    new_distances = distances * reduction_factor
                    
                    # Apply the transformation
                    processed_data[mask] = np.where(
                        data_to_scale < lower_bound,
                        lower_bound - new_distances,
                        upper_bound + new_distances
                    )
                    print(f"  Normalized {len(indices)} outliers using robust scaling")


        # Step 3: Normalize entire distribution to specified range (after clipping)
        if self.normalize_range is not None:
            target_min, target_max = self.normalize_range
            data_min = np.min(processed_data)
            data_max = np.max(processed_data)
            
            if data_max > data_min:  # Avoid division by zero
                # Min-max normalization to target range
                processed_data = (processed_data - data_min) / (data_max - data_min) * (target_max - target_min) + target_min
                print(f"  Normalized entire distribution to range [{target_min}, {target_max}]")
            else:
                print(f"  Warning: Cannot normalize - all values are identical ({data_min})")
        
        return processed_data, processed_ids
    

    def _merge_datasets_by_patterns(self, loaded_data: Dict[str, np.ndarray], loaded_ids: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Merge datasets based on specified patterns.
        
        Args:
            loaded_data: Dictionary of loaded dataset data
            loaded_ids: Dictionary of loaded dataset IDs
            
        Returns:
            Tuple of (merged_data, merged_ids) dictionaries
        """
        if not self.merge_patterns:
            return loaded_data, loaded_ids
        
        merged_data = {}
        merged_ids = {}
        datasets_to_merge = []
        datasets_to_keep = []
        
        # Find all datasets that match any of the patterns
        for dataset_name, data in loaded_data.items():
            matches_pattern = any(pattern.lower() in dataset_name.lower() for pattern in self.merge_patterns)
            if matches_pattern:
                datasets_to_merge.append((dataset_name, data, loaded_ids[dataset_name]))
            else:
                merged_data[dataset_name] = data
                merged_ids[dataset_name] = loaded_ids[dataset_name]
        
        # Merge all matching datasets into one
        if len(datasets_to_merge) > 1:
            merged_name = self.merge_patterns[0]
            merged_values = []
            merged_id_values = []
            original_names = []
            
            for dataset_name, data, ids in datasets_to_merge:
                merged_values.append(data)
                merged_id_values.append(ids)
                original_names.append(dataset_name)
            
            # Concatenate all data and IDs
            merged_data[merged_name] = np.concatenate(merged_values)
            merged_ids[merged_name] = np.concatenate(merged_id_values)
            print(f"  Merged datasets {original_names} into '{merged_name}' ({len(merged_data[merged_name])} total values)")
        else:
            return loaded_data, loaded_ids
        
        return merged_data, merged_ids


    def export_processed_data(self, export_path: str) -> None:
        """
        Export processed values and their dataset membership to a CSV file.
        
        Args:
            export_path: Path to save the exported data
        """
        if not self.distributions:
            raise ValueError("No distributions loaded. Call load_distributions() first.")
        
        # Prepare data for export
        export_data = []
        
        for dataset_name, data in self.distributions.items():
            # Add each value with its dataset label
            for value in data:
                export_data.append([value, dataset_name])
        
        # Convert to DataFrame for easy export
        df = pd.DataFrame(export_data, columns=['value', 'label'])
        
        # Export to CSV file
        df.to_csv(export_path, index=False)
        print(f"Exported {len(export_data)} values to: {export_path}")
        
        # Print summary of exported data
        print(f"\nExport summary:")
        for dataset_name, data in self.distributions.items():
            print(f"  {dataset_name}: {len(data)} values")


    def load_distributions(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Load all .json files from the directory and apply preprocessing.
        Returns: Tuple of (distributions, distribution_ids) dictionaries
        """
        if not self.directory_path.exists():
            raise FileNotFoundError(f"Directory {self.directory_path} does not exist")
            
        json_files = list(self.directory_path.glob("*metrics.json"))
        if not json_files:
            raise FileNotFoundError(f"No .json files found in {self.directory_path}")
            
        print(f"Found {len(json_files)} .json files:")
        
        loaded_distributions = {}
        loaded_distributions_ids = {}        
        
        # Iterate over all json files (distributions) and extract
        # 1. the arrays of metric values (np.array)
        # 2. the arrays of ids in the same order as the metric values (np.array)

        for json_file in json_files:
            print(f"Loading {json_file.name}...")
            try:   
                with open(json_file, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load {json_file.name}: {e}")
                continue
            
            distribution_values = []
            distribution_ids = []
            for id, cmplx in data.items():
                if self.metric in cmplx and np.isfinite(cmplx[self.metric]):
                    distribution_values.append(cmplx[self.metric])
                    distribution_ids.append(id)
            
            if not distribution_values or len(distribution_values) == 0:
                print(f"  Skipping {json_file.name} - no valid data found for metric '{self.metric}'")
                continue
            print(f"  Found {len(distribution_values)} valid data for metric '{self.metric}'")
            
            # Store data for preprocessing, modify filenames to simplify retrieval keys
            key = json_file.stem.replace('_metrics', '').replace('dataset_', '').replace('pdbbind_', '')

            loaded_distributions[key] = np.array(distribution_values)
            loaded_distributions_ids[key] = np.array(distribution_ids)

        if not loaded_distributions:
            raise ValueError("No valid distributions could be loaded")
            
        # Apply merging patterns and sort for consistent ordering
        self.distributions, self.distributions_ids = self._merge_datasets_by_patterns(loaded_distributions, loaded_distributions_ids)
        self.distributions = dict(sorted(self.distributions.items()))
        self.distributions_ids = dict(sorted(self.distributions_ids.items()))
        
        # Store original distributions for reference
        for name, data in self.distributions.items():
            self.original_distributions[name] = data.copy()
            
        # Apply preprocessing to all distributions
        for name, data in self.distributions.items():
            print(f"Processing {name}...")
            processed_data, processed_ids = self.preprocess_distribution(data, self.distributions_ids[name])
            self.distributions[name] = processed_data
            self.distributions_ids[name] = processed_ids
            print(f"  ✓ {name}: {len(processed_data)} values, "
                  f"range [{processed_data.min():.3f}, {processed_data.max():.3f}]")
        
        return self.distributions, self.distributions_ids
    

    def load_error_distribution(self, error_dict: str) -> None:
        """
        Load the error distribution from a file.
        """
        with open(error_dict, 'r') as f:
            self.error_values = json.load(f)
        
        print(f"Loaded {len(self.error_values)} error values")


    def _extract_tensor_from_dict(self, data_dict: dict) -> np.ndarray:
        """Extract tensor data from a dictionary."""
        tensors = []
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                tensors.append(value.flatten().numpy())
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                if isinstance(value[0], torch.Tensor):
                    tensors.extend([v.flatten().numpy() for v in value])
        
        if tensors:
            return np.concatenate(tensors)
        else:
            raise ValueError("No tensor data found in dictionary")
    
    def compute_individual_metrics(self) -> Dict[str, Dict]:
        """
        Compute statistical metrics for each distribution.
        
        Returns:
            Dictionary of metrics for each distribution
        """
        for name, data in self.distributions.items():
            self.metrics[name] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'median': np.median(data),
                'std': np.std(data),
                'q25': np.percentile(data, 25),
                'q75': np.percentile(data, 75),
                'min': np.min(data),
                'max': np.max(data),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data),
                'count': len(data)
            }
        return self.metrics
    
    def compute_pairwise_metrics(self) -> Dict[str, Dict]:
        """
        Compute pairwise differences between distributions.
        
        Returns:
            Dictionary of pairwise metrics
        """
        names = list(self.distributions.keys())
        
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names[i+1:], i+1):
                data1 = self.distributions[name1]
                data2 = self.distributions[name2]
                
                pair_key = f"{name1}_vs_{name2}"
                
                # Wasserstein distance (Earth Mover's Distance)
                wasserstein_dist = wasserstein_distance(data1, data2)
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_pvalue = stats.ks_2samp(data1, data2)
                
                # Jensen-Shannon divergence (approximation using histogram)
                hist1, bins = np.histogram(data1, bins=50, density=True)
                hist2, _ = np.histogram(data2, bins=bins, density=True)
                
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                hist1 = hist1 + epsilon
                hist2 = hist2 + epsilon
                
                # Normalize
                hist1 = hist1 / np.sum(hist1)
                hist2 = hist2 / np.sum(hist2)
                
                # Jensen-Shannon divergence
                m = 0.5 * (hist1 + hist2)
                js_div = 0.5 * (np.sum(hist1 * np.log(hist1 / m)) + np.sum(hist2 * np.log(hist2 / m)))
                
                # Mean absolute difference
                mean_diff = np.abs(np.mean(data1) - np.mean(data2))
                
                self.pairwise_metrics[pair_key] = {
                    'wasserstein_distance': wasserstein_dist,
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    'jensen_shannon_divergence': js_div,
                    'mean_difference': mean_diff
                }
        
        return self.pairwise_metrics


    def create_comparison_plot(self, save_path: Optional[str] = None, 
                             figsize: Tuple[int, int] = (16, 12)) -> None:
        """
        Create a comprehensive comparison plot.
        
        Args:
            save_path: Path to save the plot (optional)
            figsize: Figure size as (width, height)
        """
        if not self.distributions:
            raise ValueError("No distributions loaded. Call load_distributions() first.")
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Main histogram plot
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot histograms
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.distributions)))
        for (name, data), color in zip(self.distributions.items(), colors):
            ax1.hist(data, bins=50, alpha=0.6, label=name, color=color, density=True)
        
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Distribution Comparison - {self.metric}', fontsize=16, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Individual metrics table
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.axis('tight')
        ax2.axis('off')
        
        # Create metrics table
        metrics_data = []
        for name, metrics in self.metrics.items():
            metrics_data.append([
                name,
                f"{metrics['median']:.2e}",
                f"{metrics['std']:.2e}",
                f"{metrics['q25']:.2e}",
                f"{metrics['q75']:.2e}",
                f"{metrics['count']}"
            ])
        
        table = ax2.table(cellText=metrics_data,
                         colLabels=['Distribution', 'Median', 'Std', '25th %ile', '75th %ile', 'Count'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(metrics_data) + 1):
            for j in range(6):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#E8F5E8' if i % 2 == 0 else 'white')
        
        ax2.set_title(f'Individual Distribution Metrics - {self.metric}', fontsize=14, fontweight='bold', pad=20)
        
        # Pairwise metrics table
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('tight')
        ax3.axis('off')
        
        if self.pairwise_metrics:
            pairwise_data = []
            for pair_name, metrics in self.pairwise_metrics.items():
                pairwise_data.append([
                    pair_name.replace('_vs_', ' vs '),
                    f"{metrics['wasserstein_distance']:.2e}",
                    f"{metrics['jensen_shannon_divergence']:.3f}",
                    f"{metrics['ks_pvalue']:.3e}"
                ])
            
            table2 = ax3.table(cellText=pairwise_data,
                              colLabels=['Pair', 'Wasserstein', 'JS Divergence', 'KS p-value'],
                              cellLoc='center',
                              loc='center',
                              bbox=[0, 0, 1, 1])
            table2.auto_set_font_size(False)
            table2.set_fontsize(8)
            table2.scale(1, 2)
            
            # Style the table
            for i in range(len(pairwise_data) + 1):
                for j in range(4):
                    cell = table2[(i, j)]
                    if i == 0:  # Header row
                        cell.set_facecolor('#2196F3')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#E3F2FD' if i % 2 == 0 else 'white')
            
            ax3.set_title(f'Pairwise Distribution Differences - {self.metric}', fontsize=14, fontweight='bold', pad=20)
        
        # Box plot
        ax4 = fig.add_subplot(gs[2, :])
        data_for_box = [data for _, data in self.distributions.items()]
        labels = [name for name, _ in self.distributions.items()]
        
        bp = ax4.boxplot(data_for_box, labels=labels, patch_artist=True, showfliers=False)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # ax4.set_ylabel('Value')
        ax4.set_ylabel('Value (log scale)')
        ax4.set_yscale('symlog')
        ax4.set_title(f'Distribution Summary (Box Plot) - {self.metric}', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if they're long
        if max(len(label) for label in labels) > 10:
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # Add overall title
        title_parts = [f'Distribution Comparison Analysis - {self.metric}\nDirectory: {self.directory_path}']
        
        # Add preprocessing information to title
        preprocessing_info = []
        if self.merge_patterns:
            preprocessing_info.append(f"Merged: {', '.join(self.merge_patterns)}")
        if self.clip_percentiles is not None:
            preprocessing_info.append(f"Clipped: {self.clip_percentiles[0]}-{self.clip_percentiles[1]}th percentile")
        if self.normalize_range is not None:
            preprocessing_info.append(f"Range: [{self.normalize_range[0]}, {self.normalize_range[1]}]")

        if preprocessing_info:
            title_parts.append(f"Preprocessing: {', '.join(preprocessing_info)}")
        
        fig.suptitle('\n'.join(title_parts), fontsize=18, fontweight='bold', y=0.98)
        
        # Save plot if path provided
        if save_path:
            if self.normalize_range: save_path = save_path.replace('.png', '_norm.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {save_path}")
        
        plt.tight_layout()
        plt.show()


    def plot_metric_vs_error_scatter(self, save_path: Optional[str] = None, 
                                     figsize: Tuple[int, int] = (16, 12)) -> None:
        """
        Create a scatter plot of the metric vs. error showing distributional positioning.
        Uses the same outlier filtering as the contour version to focus on the main body.
        Each distribution uses a distinct marker and color with transparency for overlap visibility.
        """
        if not self.error_values:
            raise ValueError("No error distribution loaded. Call load_error_distribution() first.")
        
        if not self.distributions or not self.distributions_ids:
            raise ValueError("No distributions loaded. Call load_distributions() first.")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Colors and markers for different distributions
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.distributions)))
        marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '8', '<', '>', 'H']
        
        def remove_outliers_2d(x: np.ndarray, y: np.ndarray, factor: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
            """Remove outliers using IQR method for 2D data."""
            q1_x, q3_x = np.percentile(x, [25, 75])
            q1_y, q3_y = np.percentile(y, [25, 75])
            iqr_x = q3_x - q1_x
            iqr_y = q3_y - q1_y
            lower_x = q1_x - factor * iqr_x
            upper_x = q3_x + factor * iqr_x
            lower_y = q1_y - factor * iqr_y
            upper_y = q3_y + factor * iqr_y
            mask = ((x >= lower_x) & (x <= upper_x) & (y >= lower_y) & (y <= upper_y))
            return x[mask], y[mask]
        
        for (idx, ((name, metric_data), color)) in enumerate(zip(self.distributions.items(), colors)):
            # Corresponding IDs and error values
            ids = self.distributions_ids[name]
            error_data = []
            metric_data_filtered = []
            
            for i, id_val in enumerate(ids):
                if id_val in self.error_values:
                    # Mirror the contour version's absolute-error choice
                    error_data.append(np.abs(self.error_values[id_val]))
                    metric_data_filtered.append(metric_data[i])
            
            if not error_data:
                print(f"Warning: No error values found for distribution {name}")
                continue
            
            error_data = np.array(error_data)
            metric_data_filtered = np.array(metric_data_filtered)
            
            # Focus on the main body by removing outliers in both dimensions
            metric_clean, error_clean = remove_outliers_2d(metric_data_filtered, error_data)
            
            if len(metric_clean) == 0:
                print(f"Warning: No points remain for {name} after outlier removal")
                continue
            
            marker = marker_cycle[idx % len(marker_cycle)]
            ax.scatter(
                metric_clean,
                error_clean,
                s=36,
                c=[color],
                marker=marker,
                alpha=0.6,
                linewidths=0,
                edgecolors='none',
                label=name,
            )
        
        # Labels and aesthetics
        ax.set_xlabel(f'{self.metric}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error Value', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Distribution of {self.metric} vs Error Values (Scatter)\n'
            f'(Main body shown via IQR filtering)',
            fontsize=14,
            fontweight='bold',
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Distribution')
        ax.grid(True, alpha=0.3)
        
        # Overall title and preprocessing info
        title_parts = [
            f'Metric vs Error Scatter Analysis - {self.metric}\nDirectory: {self.directory_path}'
        ]
        preprocessing_info = []
        if self.merge_patterns:
            preprocessing_info.append(f"Merged: {', '.join(self.merge_patterns)}")
        if self.clip_percentiles is not None:
            preprocessing_info.append(f"Clipped: {self.clip_percentiles[0]}-{self.clip_percentiles[1]}th percentile")
        if self.normalize_range is not None:
            preprocessing_info.append(f"Range: [{self.normalize_range[0]}, {self.normalize_range[1]}]")
        if preprocessing_info:
            title_parts.append(f"Preprocessing: {', '.join(preprocessing_info)}")
        fig.suptitle('\n'.join(title_parts), fontsize=16, fontweight='bold', y=0.98)
        
        # Save plot if path provided
        if save_path:
            if self.normalize_range:
                save_path = save_path.replace('.png', '_norm.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Metric vs Error scatter plot saved to: {save_path}")
        
        plt.tight_layout()
        plt.show()

    def plot_metric_vs_error(self, save_path: Optional[str] = None, 
                                    figsize: Tuple[int, int] = (16, 12)) -> None:
        """
        Create a density plot of the metric vs. error showing distributional positioning.
        """
        if not self.error_values:
            raise ValueError("No error distribution loaded. Call load_error_distribution() first.")
        
        if not self.distributions or not self.distributions_ids:
            raise ValueError("No distributions loaded. Call load_distributions() first.")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Colors for different distributions
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.distributions)))
        
        for (name, metric_data), color in zip(self.distributions.items(), colors):
            # Get corresponding IDs and error values
            ids = self.distributions_ids[name]
            
            # Extract error values for this distribution
            error_data = []
            metric_data_filtered = []
            
            for i, id_val in enumerate(ids):
                if id_val in self.error_values:
                    error_data.append(np.abs(self.error_values[id_val]))
                    metric_data_filtered.append(metric_data[i])
            
            if not error_data:
                print(f"Warning: No error values found for distribution {name}")
                continue
                
            error_data = np.array(error_data)
            metric_data_filtered = np.array(metric_data_filtered)
            
            # Remove outliers using IQR method for both dimensions
            def remove_outliers_2d(x, y, factor=1.5):
                """Remove outliers using IQR method for 2D data"""
                # Calculate IQR for x and y
                q1_x, q3_x = np.percentile(x, [25, 75])
                q1_y, q3_y = np.percentile(y, [25, 75])
                iqr_x = q3_x - q1_x
                iqr_y = q3_y - q1_y
                
                # Define bounds
                lower_x = q1_x - factor * iqr_x
                upper_x = q3_x + factor * iqr_x
                lower_y = q1_y - factor * iqr_y
                upper_y = q3_y + factor * iqr_y
                
                # Filter data
                mask = ((x >= lower_x) & (x <= upper_x) & 
                       (y >= lower_y) & (y <= upper_y))
                
                return x[mask], y[mask]
            
            # Remove outliers to focus on main distribution
            metric_clean, error_clean = remove_outliers_2d(metric_data_filtered, error_data)
            
            if len(metric_clean) < 10:  # Need minimum points for contour
                print(f"Warning: Too few points for {name} after outlier removal ({len(metric_clean)} points)")
                continue
            
            # Create 2D histogram for density estimation
            try:
                # Use kernel density estimation for smooth contours
                from scipy.stats import gaussian_kde
                
                # Create grid for contour plot
                x_min, x_max = metric_clean.min(), metric_clean.max()
                y_min, y_max = error_clean.min(), error_clean.max()
                
                # Add some padding
                x_range = x_max - x_min
                y_range = y_max - y_min
                x_min -= 0.1 * x_range
                x_max += 0.1 * x_range
                y_min -= 0.1 * y_range
                y_max += 0.1 * y_range
                
                # Create grid
                x_grid = np.linspace(x_min, x_max, 100)
                y_grid = np.linspace(y_min, y_max, 100)
                X, Y = np.meshgrid(x_grid, y_grid)
                positions = np.vstack([X.ravel(), Y.ravel()])
                
                # Fit KDE
                values = np.vstack([metric_clean, error_clean])
                kde = gaussian_kde(values)
                Z = np.reshape(kde(positions).T, X.shape)
                
                # Plot contours (only lines, no filled background)
                contour_levels = np.linspace(Z.min(), Z.max(), 8)
                contour = ax.contour(X, Y, Z, levels=contour_levels, colors=[color], alpha=0.8, linewidths=2)
                
                # Add label for this distribution
                ax.plot([], [], color=color, linewidth=3, label=name)
                
            except Exception as e:
                print(f"Warning: Could not create contour for {name}: {e}")
                # Fallback to scatter plot if contour fails
                ax.scatter(metric_clean, error_clean, color=color, alpha=0.6, s=20, label=name)
        
        # Customize the plot
        ax.set_xlabel(f'{self.metric}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Distribution of {self.metric} vs Error Values\n(Contour plots showing density)', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add overall title
        title_parts = [f'Metric vs Error Distribution Analysis - {self.metric}\nDirectory: {self.directory_path}']
        
        # Add preprocessing information to title
        preprocessing_info = []
        if self.merge_patterns:
            preprocessing_info.append(f"Merged: {', '.join(self.merge_patterns)}")
        if self.clip_percentiles is not None:
            preprocessing_info.append(f"Clipped: {self.clip_percentiles[0]}-{self.clip_percentiles[1]}th percentile")
        if self.normalize_range is not None:
            preprocessing_info.append(f"Range: [{self.normalize_range[0]}, {self.normalize_range[1]}]")

        if preprocessing_info:
            title_parts.append(f"Preprocessing: {', '.join(preprocessing_info)}")
        
        fig.suptitle('\n'.join(title_parts), fontsize=16, fontweight='bold', y=0.98)
        
        # Save plot if path provided
        if save_path:
            if self.normalize_range: 
                save_path = save_path.replace('.png', '_norm.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Metric vs Error plot saved to: {save_path}")
        
        plt.tight_layout()
        plt.show()


    def plot_heatmaps(self, save_path: Optional[str] = None, 
                     figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Create individual heatmap plots for each distribution showing metric vs error density.
        Uses the same outlier filtering strategy to focus on the main body of distributions.
        Each distribution gets its own subplot with a 2D histogram heatmap.
        """
        if not self.error_values:
            raise ValueError("No error distribution loaded. Call load_error_distribution() first.")
        
        if not self.distributions or not self.distributions_ids:
            raise ValueError("No distributions loaded. Call load_distributions() first.")
        
        # Calculate grid layout for subplots
        n_distributions = len(self.distributions)
        n_cols = min(3, n_distributions)  # Max 3 columns
        n_rows = (n_distributions + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows))
        if n_distributions == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        def remove_outliers_2d(x: np.ndarray, y: np.ndarray, factor: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
            # """Remove outliers using IQR method for 2D data."""
            # q1_x, q3_x = np.percentile(x, [25, 75])
            # q1_y, q3_y = np.percentile(y, [25, 75])
            # iqr_x = q3_x - q1_x
            # iqr_y = q3_y - q1_y
            # lower_x = q1_x - factor * iqr_x
            # upper_x = q3_x + factor * iqr_x
            # lower_y = q1_y - factor * iqr_y
            # upper_y = q3_y + factor * iqr_y
            # mask = ((x >= lower_x) & (x <= upper_x) & (y >= lower_y) & (y <= upper_y))
            # return x[mask], y[mask]
            return x, y
        
        for idx, (name, metric_data) in enumerate(self.distributions.items()):
            ax = axes[idx]
            
            print(f"\nCreating heatmap for {name}")

            # Get corresponding IDs and error values
            ids = self.distributions_ids[name]
            for id_val in ids[:10]:
                if id_val in self.error_values:
                    print(f"    {id_val}: {self.error_values[id_val]}")
            
            error_data = []
            metric_data_filtered = []
            for i, id_val in enumerate(ids):
                if id_val in self.error_values:
                    # Use absolute error values like in other functions
                    error_data.append(np.abs(self.error_values[id_val]))
                    metric_data_filtered.append(metric_data[i])


            
            if not error_data:
                ax.text(0.5, 0.5, f'No error values\nfound for {name}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{name} - No Data', fontweight='bold')
                ax.set_xlim(0, 1)  # Fixed x-axis limits
                ax.set_ylim(0, 6)  # Fixed y-axis limits
                continue

            # Calculate the RMSD of the error data
            rmsd = np.sqrt(np.mean(np.square(error_data)))
            print(f"  RMSD: {rmsd}")

            error_data = np.array(error_data)
            metric_data_filtered = np.array(metric_data_filtered)
            
            # Focus on the main body by removing outliers
            metric_clean, error_clean = remove_outliers_2d(metric_data_filtered, error_data)
            
            if len(metric_clean) < 5:
                ax.text(0.5, 0.5, f'Insufficient data\nfor {name}\n({len(metric_clean)} points)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'{name} - Insufficient Data', fontweight='bold')
                ax.set_xlim(0, 1)  # Fixed x-axis limits
                ax.set_ylim(0, 6)  # Fixed y-axis limits
                continue
            
            # Create 2D histogram for heatmap
            try:
                # Define fixed bins for all heatmaps to ensure comparability
                n_bins = 30
                x_bins = np.linspace(0, 1, n_bins + 1)  # Fixed x-range: 0 to 1
                y_bins = np.linspace(0, 6, n_bins + 1)  # Fixed y-range: 0 to 6
                
                # Create 2D histogram with fixed bins
                hist, x_edges, y_edges = np.histogram2d(metric_clean, error_clean, 
                                                      bins=[x_bins, y_bins], density=True)
                
                # Create heatmap with fixed extent
                im = ax.imshow(hist.T, origin='lower', aspect='auto', 
                              extent=[0, 1, 0, 6],  # Fixed extent for all heatmaps
                              cmap='viridis', interpolation='bilinear')
                
                # Add colorbar for this subplot
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Density', fontsize=10)
                
                # Customize subplot
                ax.set_xlabel(f'{self.metric}', fontsize=10, fontweight='bold')
                ax.set_ylabel('Error Value', fontsize=10, fontweight='bold')
                ax.set_title(f'{name}\n({len(metric_clean)} points)', fontsize=12, fontweight='bold')
                ax.set_xlim(0, 1)  # Fixed x-axis limits
                ax.set_ylim(0, 6)  # Fixed y-axis limits
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error creating\nheatmap for {name}:\n{str(e)[:50]}...', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=9)
                ax.set_title(f'{name} - Error', fontweight='bold')
                ax.set_xlim(0, 1)  # Fixed x-axis limits
                ax.set_ylim(0, 6)  # Fixed y-axis limits
        
        # Hide unused subplots
        for idx in range(n_distributions, len(axes)):
            axes[idx].set_visible(False)
        
        # Overall title
        title_parts = [f'Individual Distribution Heatmaps - {self.metric}\nDirectory: {self.directory_path}']
        
        # Add preprocessing information to title
        preprocessing_info = []
        if self.merge_patterns:
            preprocessing_info.append(f"Merged: {', '.join(self.merge_patterns)}")
        if self.clip_percentiles is not None:
            preprocessing_info.append(f"Clipped: {self.clip_percentiles[0]}-{self.clip_percentiles[1]}th percentile")
        if self.normalize_range is not None:
            preprocessing_info.append(f"Range: [{self.normalize_range[0]}, {self.normalize_range[1]}]")
        if preprocessing_info:
            title_parts.append(f"Preprocessing: {', '.join(preprocessing_info)}")
        
        fig.suptitle('\n'.join(title_parts), fontsize=16, fontweight='bold', y=0.95)
        
        # Save plot if path provided
        if save_path:
            if self.normalize_range:
                save_path = save_path.replace('.png', '_norm.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Heatmap plots saved to: {save_path}")
        
        plt.tight_layout()
        plt.show()


    def print_summary(self) -> None:
        """Print a summary of the analysis."""
        print("\n" + "="*80)
        print("DISTRIBUTION COMPARISON SUMMARY")
        print("="*80)
        
        print(f"\nDirectory: {self.directory_path}")
        print(f"Number of distributions: {len(self.distributions)}")
        
        # Print preprocessing information
        print("\nPreprocessing Applied:")
        print("-" * 30)
        if self.merge_patterns:
            print(f"✓ Dataset merging: Merged datasets containing patterns {self.merge_patterns}")
        else:
            print("✗ No dataset merging applied")
            
        if self.clip_percentiles is not None:
            print(f"✓ Clipping: {self.clip_percentiles[0]}th to {self.clip_percentiles[1]}th percentiles")
        else:
            print("✗ No clipping applied")

        if self.normalize_range is not None:
            print(f"✓ Range normalization: Normalized to [{self.normalize_range[0]}, {self.normalize_range[1]}]")
        else:
            print("✗ No range normalization applied")
        
        print("\nIndividual Distribution Statistics:")
        print("-" * 50)
        for name, metrics in sorted(self.metrics.items()):
            print(f"\n{name}:")
            print(f"  Count: {metrics['count']:,}")
            print(f"  Median: {metrics['median']:.2e}")
            print(f"  Std: {metrics['std']:.2e}")
            print(f"  25th percentile: {metrics['q25']:.2e}")
            print(f"  75th percentile: {metrics['q75']:.2e}")
            print(f"  Range: [{metrics['min']:.2e}, {metrics['max']:.2e}]")
            print(f"  Skewness: {metrics['skewness']:.4f}")
            print(f"  Kurtosis: {metrics['kurtosis']:.4f}")
            
            # Show original vs processed range if preprocessing was applied
            if name in self.original_distributions:
                orig_data = self.original_distributions[name]
                print(f"  Original range: [{orig_data.min():.4f}, {orig_data.max():.4f}]")
        
        if self.pairwise_metrics:
            print("\nPairwise Distribution Differences:")
            print("-" * 50)
            for pair_name, metrics in self.pairwise_metrics.items():
                print(f"\n{pair_name}:")
                print(f"  Wasserstein Distance: {metrics['wasserstein_distance']:.2e}")
                print(f"  Jensen-Shannon Divergence: {metrics['jensen_shannon_divergence']:.4f}")
                print(f"  KS Statistic: {metrics['ks_statistic']:.4f}")
                print(f"  KS p-value: {metrics['ks_pvalue']:.2e}")
                print(f"  Mean Difference: {metrics['mean_difference']:.4f}")
        
        print("\n" + "="*80)


def main():
    """Main function to run the distribution comparison."""

    args = parse_arguments()

    try:
        # Parse clip percentiles if provided
        clip_percentiles = None
        if args.clip_percentiles:
            if len(args.clip_percentiles) != 2:
                raise ValueError("clip_percentiles must specify exactly 2 values (lower, upper)")
            lower, upper = args.clip_percentiles
            if not (0 <= lower < upper <= 100):
                raise ValueError("clip_percentiles must be between 0 and 100, with lower < upper")
            clip_percentiles = (lower, upper)

        # Parse normalization range if provided
        normalize_range = None
        if args.normalize_range:
            if len(args.normalize_range) != 2:
                raise ValueError("normalize_range must specify exactly 2 values (min, max)")
            min_val, max_val = args.normalize_range
            if min_val >= max_val:
                raise ValueError("normalize_range min value must be less than max value")
            normalize_range = (min_val, max_val)

        # Initialize comparator with preprocessing options
        comparator = DistributionComparator(
            args.directory, 
            metric=args.metric,
            clip_percentiles=clip_percentiles,
            remove_outliers=args.remove_outliers,
            cut_outliers=args.cut_outliers,
            robust_scaling=args.robust_scaling,
            normalize_range=normalize_range,
            merge_patterns=args.merge_patterns
        )
        
        # Load distributions and preprocess them
        print("Loading distributions...")
        distributions, distribution_ids = comparator.load_distributions()
        print(f"Loaded {len(distributions)} distributions")
        for key, value in distributions.items():
            print(f"  {key}: {value.shape}")
        print(f"Loaded {len(distribution_ids)} distribution IDs")
        for key, value in distribution_ids.items():
            print(f"  {key}: {value.shape}")
        
        # Compute metrics
        print("\nComputing individual metrics...")
        comparator.compute_individual_metrics()
        
        print("Computing pairwise metrics...")
        comparator.compute_pairwise_metrics()
        
        # Export data if requested
        if args.export:
            print("\nExporting processed data...")
            comparator.export_processed_data(args.export)
        
        # Create plot
        if args.save_plot:
            print("Creating comparison plot...")
            comparator.create_comparison_plot(
                save_path=os.path.join(args.directory, 'distcomp_' + args.save_plot),
                figsize=tuple(args.figsize)
            )
            if args.error_dict:
                print("Loading error distribution...")
                comparator.load_error_distribution(args.error_dict)
                comparator.plot_metric_vs_error_scatter(
                    save_path=os.path.join(args.directory, 'error_vs_' + args.save_plot),
                )
                
                # Create heatmaps if requested
                if args.plot_heatmaps:
                    print("Creating individual heatmap plots...")
                    comparator.plot_heatmaps(
                        save_path=os.path.join(args.directory, 'heatmaps_' + args.save_plot),
                        figsize=(12, 8)
                )
        
        # Print summary
        comparator.print_summary()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 