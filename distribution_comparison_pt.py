"""
Distribution Comparison Tool

This script loads .pt tensor files from a directory and compares their distributions
visually and statistically. It provides:
- Histogram plots of all distributions
- Statistical metrics (mean, std, skewness, kurtosis)
- Pairwise distribution differences (KL divergence, Wasserstein distance)
- Summary statistics in the plot
- Optional preprocessing: dataset merging, clipping, outlier normalization, and range normalization
- Optional export: save processed data as CSV file with value-label pairs
"""

import os
import sys
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

class DistributionComparator:
    """A class to compare multiple distributions from .pt tensor files."""
    
    def __init__(self, 
                directory_path: str, 
                clip_percentiles: Optional[Tuple[float, float]] = None, 
                cut_outliers: bool = False,
                normalize_outliers: bool = False, 
                merge_patterns: Optional[List[str]] = None):
        """
        Initialize the comparator with a directory path.
        
        Args:
            directory_path: Path to directory containing .pt files
            clip_percentiles: Tuple of (lower_percentile, upper_percentile) for clipping, e.g., (1, 99)
            cut_outliers: Whether to cut outliers instead of clipping them
            normalize_outliers: Whether to normalize outliers using robust scaling
            merge_patterns: List of patterns to merge datasets (e.g., ['train', 'validation'] will merge datasets containing these words)
        """
        self.directory_path = Path(directory_path)
        self.clip_percentiles = clip_percentiles
        self.cut_outliers = cut_outliers
        self.normalize_outliers = normalize_outliers
        self.merge_patterns = merge_patterns or []
        self.distributions = {}
        self.original_distributions = {}  # Store original data for reference
        self.metrics = {}
        self.pairwise_metrics = {}

        print(f"Comparing distributions in {self.directory_path}")
        print(f"  Clip percentiles: {self.clip_percentiles}")
        print(f"  Cut outliers: {self.cut_outliers}")
        print(f"  Normalize outliers: {self.normalize_outliers}")
        print(f"  Merge patterns: {self.merge_patterns}")

        
    def preprocess_distribution(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess a single distribution with clipping and/or normalization.
        
        Args:
            data: Input distribution data
            
        Returns:
            Preprocessed distribution data
        """
        processed_data = data.copy()
        
        # Step 1: Normalize outliers using robust scaling
        if self.normalize_outliers:
            # Use median and IQR for robust scaling
            median = np.median(processed_data)
            q75, q25 = np.percentile(processed_data, [75, 25])
            iqr = q75 - q25
            
            if iqr > 0:  # Avoid division by zero
                # Define outlier bounds (1.5 * IQR rule)
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                
                # Find outliers
                outlier_mask = (processed_data < lower_bound) | (processed_data > upper_bound)
                outlier_indices = np.where(outlier_mask)[0]
                
                if len(outlier_indices) > 0:
                    # Normalize outliers by bringing them closer to the main distribution
                    # Use a sigmoid-like transformation to smoothly bring outliers in
                    outlier_data = processed_data[outlier_mask]
                    
                    # Calculate how far outliers are from bounds
                    distances = np.where(outlier_data < lower_bound, 
                                       lower_bound - outlier_data,
                                       outlier_data - upper_bound)
                    
                    # Apply a smooth transformation to bring outliers closer
                    # Use a factor that reduces the distance by 99%
                    reduction_factor = 0.1
                    new_distances = distances * reduction_factor
                    
                    # Apply the transformation
                    processed_data[outlier_mask] = np.where(
                        outlier_data < lower_bound,
                        lower_bound - new_distances,
                        upper_bound + new_distances
                    )
                    
                    print(f"  Normalized {len(outlier_indices)} outliers using robust scaling")
        
        # Step 2: Clipping at percentiles (before range normalization)
        if self.clip_percentiles is not None:
            lower_percentile, upper_percentile = self.clip_percentiles
            lower_bound = np.percentile(processed_data, lower_percentile)
            upper_bound = np.percentile(processed_data, upper_percentile)

            if not self.cut_outliers:
                # CLIPPING THE OUTLIERS
                processed_data = np.clip(processed_data, lower_bound, upper_bound)
                print(f"  Clipped to [{lower_percentile}th, {upper_percentile}th] percentiles: "
                      f"[{lower_bound:.2e}, {upper_bound:.2e}]")
        
            if self.cut_outliers:
                # CUTTING THE OUTLIERS
                processed_data = processed_data[(processed_data > lower_bound) & (processed_data < upper_bound)]
                print(f"  Cut off outliers at [{lower_percentile}th, {upper_percentile}th] percentiles: "
                    f"[{lower_bound:.2e}, {upper_bound:.2e}]")
        
        return processed_data
    

    def _merge_datasets_by_patterns(self, loaded_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Merge datasets based on specified patterns.
        
        Args:
            loaded_data: Dictionary of loaded dataset data
            
        Returns:
            Dictionary with merged datasets
        """
        if not self.merge_patterns:
            return loaded_data
        
        merged_data = {}
        datasets_to_merge = []
        datasets_to_keep = []
        
        # Find all datasets that match any of the patterns
        for dataset_name, data in loaded_data.items():
            matches_pattern = any(pattern.lower() in dataset_name.lower() for pattern in self.merge_patterns)
            if matches_pattern:
                datasets_to_merge.append((dataset_name, data))
            else:
                merged_data[dataset_name] = data
        
        # Merge all matching datasets into one
        if len(datasets_to_merge) > 1:
            merged_name = self.merge_patterns[0]
            merged_values = []
            original_names = []
            
            for dataset_name, data in datasets_to_merge:
                merged_values.append(data)
                original_names.append(dataset_name)
            
            # Concatenate all data
            merged_data[merged_name] = np.concatenate(merged_values)
            print(f"  Merged datasets {original_names} into '{merged_name}' ({len(merged_data[merged_name])} total values)")
        else:
            return loaded_data
        
        return merged_data
    
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
    
    def load_distributions(self) -> Dict[str, torch.Tensor]:
        """
        Load all .pt files from the directory and apply preprocessing.
        
        Returns:
            Dictionary mapping filename to tensor data
        """
        if not self.directory_path.exists():
            raise FileNotFoundError(f"Directory {self.directory_path} does not exist")
            
        pt_files = list(self.directory_path.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {self.directory_path}")
            
        print(f"Found {len(pt_files)} .pt files:")
        
        loaded_data = {}
        for pt_file in pt_files:
            try:
                # Load tensor data
                data = torch.load(pt_file, map_location='cpu')
                data = torch.tensor(data)
                
                # Handle different tensor formats
                if isinstance(data, torch.Tensor):
                    # If it's a tensor, flatten it
                    flat_data = data.flatten().numpy()
                elif isinstance(data, dict):
                    # If it's a dict, try to find tensor values
                    flat_data = self._extract_tensor_from_dict(data)
                elif isinstance(data, list):
                    # If it's a list, concatenate all tensors
                    flat_data = np.concatenate([t.flatten().numpy() if isinstance(t, torch.Tensor) else t for t in data])
                else:
                    print(f"Warning: Skipping {pt_file.name} - unsupported data type: {type(data)}")
                    continue
                
                # Remove any NaN or infinite values
                flat_data = flat_data[np.isfinite(flat_data)]
                
                if len(flat_data) == 0:
                    print(f"Warning: Skipping {pt_file.name} - no valid data")
                    continue
                
                # Store original data
                loaded_data[pt_file.stem] = flat_data.copy()
                
                print(f"  Loading {pt_file.name}...")
                
            except Exception as e:
                print(f"Warning: Could not load {pt_file.name}: {e}")
                
        if not loaded_data:
            raise ValueError("No valid distributions could be loaded")
            
        # Apply merging patterns
        self.distributions = self._merge_datasets_by_patterns(loaded_data)
        
        # Store original distributions for reference
        for name, data in self.distributions.items():
            self.original_distributions[name] = data.copy()
            
        # Apply preprocessing to all distributions
        for name, data in self.distributions.items():
            print(f"Processing {name}...")
            processed_data = self.preprocess_distribution(data)
            self.distributions[name] = processed_data
            print(f"  ✓ {name}: {len(processed_data)} values, "
                  f"range [{processed_data.min():.3f}, {processed_data.max():.3f}]")
        
        return self.distributions
    
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
        ax1.set_title('Distribution Comparison', fontsize=16, fontweight='bold')
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
                name.replace('dataset_', ''),
                f"{metrics['mean']:.2e}",
                f"{metrics['std']:.2e}",
                f"{metrics['skewness']:.3f}",
                f"{metrics['kurtosis']:.3f}",
                f"{metrics['count']}"
            ])
        
        table = ax2.table(cellText=metrics_data,
                         colLabels=['Distribution', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Count'],
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
        
        ax2.set_title('Individual Distribution Metrics', fontsize=14, fontweight='bold', pad=20)
        
        # Pairwise metrics table
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('tight')
        ax3.axis('off')
        
        if self.pairwise_metrics:
            pairwise_data = []
            for pair_name, metrics in self.pairwise_metrics.items():
                pairwise_data.append([
                    pair_name.replace('_vs_', ' vs ').replace('dataset_', ''),
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
            
            ax3.set_title('Pairwise Distribution Differences', fontsize=14, fontweight='bold', pad=20)
        
        # Box plot
        ax4 = fig.add_subplot(gs[2, :])
        data_for_box = [data for data in self.distributions.values()]
        labels = list(self.distributions.keys())
        
        bp = ax4.boxplot(data_for_box, labels=labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_ylabel('Value')
        ax4.set_title('Distribution Summary (Box Plot)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if they're long
        if max(len(label) for label in labels) > 10:
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # Add overall title
        title_parts = [f'Distribution Comparison Analysis\nDirectory: {self.directory_path}']
        
        # Add preprocessing information to title
        preprocessing_info = []
        if self.merge_patterns:
            preprocessing_info.append(f"Merged: {', '.join(self.merge_patterns)}")
        if self.normalize_outliers:
            preprocessing_info.append("Outliers normalized")
        if self.clip_percentiles is not None:
            preprocessing_info.append(f"Clipped: {self.clip_percentiles[0]}-{self.clip_percentiles[1]}th percentile")
        if preprocessing_info:
            title_parts.append(f"Preprocessing: {', '.join(preprocessing_info)}")
        
        fig.suptitle('\n'.join(title_parts), fontsize=18, fontweight='bold', y=0.98)
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {save_path}")
        
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
            
        if self.normalize_outliers:
            print("✓ Outlier normalization: Applied robust scaling")
        else:
            print("✗ No outlier normalization applied")
            
        if self.clip_percentiles is not None:
            print(f"✓ Clipping: {self.clip_percentiles[0]}th to {self.clip_percentiles[1]}th percentiles")
        else:
            print("✗ No clipping applied")
            
        
        print("\nIndividual Distribution Statistics:")
        print("-" * 50)
        for name, metrics in self.metrics.items():
            print(f"\n{name}:")
            print(f"  Count: {metrics['count']:,}")
            print(f"  Mean: {metrics['mean']:.2e}")
            print(f"  Std: {metrics['std']:.2e}")
            print(f"  Median: {metrics['median']:.4f}")
            print(f"  Range: [{metrics['min']:.4f}, {metrics['max']:.4f}]")
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
    parser = argparse.ArgumentParser(description='Compare distributions from .pt tensor files')
    parser.add_argument('directory', help='Directory containing .pt files')
    parser.add_argument('--save_plot', '-s', help='Path to save the comparison plot')
    parser.add_argument('--figsize', nargs=2, type=int, default=[20, 16], 
                       help='Figure size as width height (default: 16 12)')
    parser.add_argument('--clip_percentiles', nargs=2, type=float, 
                       help='Clip data to specified percentiles (e.g., 1 99 for 1st to 99th percentile)')
    parser.add_argument('--cut_outliers', action='store_true',
                       help='Cut outliers instead of clipping them')
    parser.add_argument('--normalize_outliers', action='store_true',
                       help='Normalize outliers using robust scaling (brings outliers closer to main distribution)')
    parser.add_argument('--merge_patterns', nargs='+', type=str,
                       help='Patterns to merge datasets (e.g., train validation will merge datasets containing these words)')
    parser.add_argument('--export', help='Path to export processed data as CSV file')
    
    args = parser.parse_args()
    
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
        
        # Initialize comparator with preprocessing options
        comparator = DistributionComparator(
            args.directory, 
            clip_percentiles=clip_percentiles,
            cut_outliers=args.cut_outliers,
            normalize_outliers=args.normalize_outliers,
            merge_patterns=args.merge_patterns
        )
        
        # Load distributions
        print("Loading distributions...")
        comparator.load_distributions()
        
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
                save_path=args.save_plot,
                figsize=tuple(args.figsize)
            )
        else:
            print("Creating comparison plot...")
            comparator.create_comparison_plot(
                save_path=f"{args.directory}/comparison.png",
                figsize=tuple(args.figsize)
            )
        
        # Print summary
        comparator.print_summary()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 