"""
Distribution Comparison Tool

Loads .json metrics files from a directory and compares distributions of a given metric.
Provides statistical analysis, visualizations, and hierarchical exponential curve fitting.

Features:
- Histogram and box plot comparisons
- Statistical metrics and pairwise distribution differences
- Heatmap visualization for metric vs error distributions  
- Hierarchical exponential scatterplot fitting (5-curve system)
- Preprocessing: outlier removal, robust scaling, normalization, asinh scaling
- CSV export for processed data and metrics
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy import stats
from scipy.stats import wasserstein_distance
from scipy.optimize import curve_fit
from sklearn.metrics import mutual_info_score
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# MATPLOTLIB PLOTTING SETTINGS
# -----------------------------------------------------------------------------------------
# Fix Arial font issue by using a fallback font family
plt.rcParams.update({
    "font.family": "sans serif",
    "svg.fonttype": "none",          # keep text as text in SVG
    'font.size': 12,                 # Set the base font size 
    'axes.labelsize': 12,            # X and Y axis labels
    'xtick.labelsize': 10,           # X-axis tick labels
    'ytick.labelsize': 10,           # Y-axis tick labels
    'legend.fontsize': 10,           # Legend font size
    'axes.titlesize': 13,            # Plot title size
    'axes.linewidth': 0.5            # Thinner axis lines
})

# For figure width 1/2 A4 page --> figsize = (6, x), then reduce figure width by half in AI
# For figure width 1/3 A4 page --> figsize = (4, x), then reduce figure width by half in AI
# For figure width 1/4 A4 page --> figsize = (3, x), then reduce figure width by half in AI
# plt.style.use('seaborn-v0_8')

# Define custom box properties
boxprops = dict(linewidth=0.5, facecolor='None', color='black')
medianprops = dict(linewidth=0.5, color='black')
flierprops = dict(marker='o', markersize=3, linestyle='none', markeredgewidth=0.5, markeredgecolor='black')
whiskerprops = dict(linewidth=0.5)
capprops = dict(linewidth=0.5)

# Create CUSTOM colormap for heatmaps
colors = ["#ffffff", "#ff0000"]
custom_cmap = LinearSegmentedColormap.from_list("white_to_red", colors)
# -----------------------------------------------------------------------------------------


def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare distributions from .json metrics files')

    # Required arguments
    parser.add_argument('directory', type=str, help='Directory containing .json files')
    parser.add_argument('--metric', type=str, required=True, help='Metric to compare')

    # Export options
    parser.add_argument('--export', help='Path to export processed data as CSV file')

    # Preprocessing options
    parser.add_argument('--remove_outliers', action='store_true', help='Remove outliers using IQR method')
    parser.add_argument('--robust_scaling', action='store_true', help='Normalize outliers using robust scaling (brings outliers closer to main distribution)')
    parser.add_argument('--normalize', action='store_true', help='Normalize entire distributions to [-1,0] range')
    parser.add_argument('--asinh_scaling', action='store_true', help='Apply global asinh scaling to all distributions (preserves sign)')
    
    parser.add_argument('--show_outliers', action='store_true', dest='show_outliers', 
                        help='Show outliers (fliers) in the boxplot. By default outliers are hidden')
    parser.add_argument('--merge_patterns', nargs='+', type=str,
                        help='Patterns to merge datasets (e.g., train validation will merge datasets containing these words)')
    parser.add_argument('--error_dict', type=str, default=None,
                        help='Path to the error distribution file mapping IDs to error values')
    parser.add_argument('--plot_heatmaps', action='store_true',
                        help='Create individual heatmap plots for each distribution')
    parser.add_argument('--plot_exponential_scatterplot', nargs='+', type=str, metavar='DIST_NAME',
                        help='Create scatterplot with exponential curve fitting for specified distributions (provide 2 or more distribution names)')
    return parser.parse_args()


class DistributionComparator:
    """Compare multiple distributions from .json metrics files with statistical analysis and visualization."""
    
    def __init__(self, 
                directory_path: str, 
                metric: str,
                remove_outliers: bool = False,
                robust_scaling: bool = False,
                normalize: bool = False,
                merge_patterns: Optional[List[str]] = None,
                asinh_scaling: bool = False):
        """
        Initialize distribution comparator.
        
        Args:
            directory_path: Path to directory with .json files
            metric: Metric name to compare
            remove_outliers: Remove outliers using IQR method
            robust_scaling: Apply robust scaling normalization
            normalize: Normalize to [-1,0] range  
            merge_patterns: Patterns for merging datasets
            asinh_scaling: Apply asinh transformation
        """

        self.directory_path = Path(directory_path)
        self.metric = metric
        self.remove_outliers = remove_outliers
        self.robust_scaling = robust_scaling
        self.normalize = normalize
        self.merge_patterns = merge_patterns or []
        self.distributions = {}
        self.distributions_ids = {}  # Store distribution IDs aligned with distributions
        self.error_values = {}
        self.original_distributions = {}  # Store original data for reference
        self.metrics = {}
        self.pairwise_metrics = {}
        self.asinh_scaling = asinh_scaling

        self.global_min = None
        self.global_max = None
        self.global_median = None
        self.global_q1 = None
        self.global_q3 = None

        self.global_error_min = None
        self.global_error_max = None

        print(f"\nComparing distributions in {self.directory_path}")
        print(f"  Metric: {self.metric}")
        print(f"  Remove outliers: {self.remove_outliers}")
        print(f"  Robust scaling: {self.robust_scaling}")
        print(f"  Normalize: {self.normalize}")
        print(f"  Merge patterns: {self.merge_patterns}")
        print(f"  Asinh scaling: {self.asinh_scaling}")

    
    def _load_distributions(self) -> None:
        """
        Load all .json files from the directory and apply preprocessing.
        Applies individual outlier removal, then global preprocessing (asinh scaling, robust scaling, normalization).
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
        self.distributions, self.distributions_ids = self._merge_distributions_by_patterns(loaded_distributions, loaded_distributions_ids)
        self.distributions = dict(sorted(self.distributions.items()))
        self.distributions_ids = dict(sorted(self.distributions_ids.items()))
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(self.distributions)))
            
        # Apply preprocessing to distributions individually
        for name, data in self.distributions.items():
            self.original_distributions[name] = data.copy()
            print(f"Processing {name}...")
            processed_data, processed_ids = self._preprocess_distribution(data, self.distributions_ids[name])
            self.distributions[name] = processed_data
            self.distributions_ids[name] = processed_ids
            print(f"  ✓ {name}: {len(processed_data)} values, "
                  f"range [{processed_data.min():.3f}, {processed_data.max():.3f}]")

        print(f"\nLoaded {len(self.distributions)} distributions")
        for key, value in self.distributions.items():
            print(f"  {key}: {value.shape}")

        # Compute global metrics
        self._compute_global_metrics()
            
        # Apply global preprocessing
        self._apply_global_preprocessing()


    def _preprocess_distribution(self, data: np.ndarray, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess a single distribution with outlier removal only.
        Robust scaling, range normalization, and asinh scaling are applied globally after all distributions are loaded.
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

                # CUTTING THE OUTLIERS - need to remove corresponding IDs too
                valid_mask = (processed_data < upper_bound) & (processed_data > lower_bound)
                processed_data = processed_data[valid_mask]
                processed_ids = processed_ids[valid_mask]
                print(f"  Cut off outliers outside the range [{lower_bound:.2e}, {upper_bound:.2e}]")

            else:
                print(f"  No outliers removed")
        return processed_data, processed_ids


    def _compute_global_metrics(self) -> None:
        """
        Compute global statistics (min, max, median, Q1, Q3) across all distributions.
        Updates self.global_min, self.global_max, self.global_median, self.global_q1, self.global_q3.
        """
        if not self.distributions:
            return
            
        # Concatenate all distribution data
        all_data = np.concatenate([data for data in self.distributions.values()])
        
        self.global_min = np.min(all_data)
        self.global_max = np.max(all_data)
        self.global_median = np.median(all_data)
        self.global_q1 = np.percentile(all_data, 25)
        self.global_q3 = np.percentile(all_data, 75)
        self.global_90 = np.percentile(all_data, 90)
        self.global_10 = np.percentile(all_data, 10)
        
        print(f"\nGlobal statistics:")
        print(f"  Min: {self.global_min:.2e}")
        print(f"  Max: {self.global_max:.2e}")
        print(f"  Median: {self.global_median:.2e}")
        print(f"  Q1: {self.global_q1:.2e}")
        print(f"  Q3: {self.global_q3:.2e}")


    def _apply_global_preprocessing(self) -> None:
        """Apply global preprocessing: asinh scaling, robust scaling, range normalization."""
        if not self.distributions:
            return
            
        # Apply global asinh scaling if requested
        if self.asinh_scaling:
            print(f"\nApplying global asinh scaling...")
            # Use positive scale based on absolute global median to preserve sign
            c = abs(self.global_median) if self.global_median is not None else 1.0
            if c == 0:
                c = 1.0
            for name, data in self.distributions.items():
                original_min, original_max = np.min(data), np.max(data)
                transformed = np.arcsinh(data / c)
                self.distributions[name] = transformed
                new_min, new_max = np.min(transformed), np.max(transformed)
                print(f"    {name}: asinh scaled [{original_min:.2e}, {original_max:.2e}] -> [{new_min:.2e}, {new_max:.2e}] (c={c:.2e})")
                
            print(f"  Recomputing global metrics after asinh scaling...")
            self._compute_global_metrics()
        

        # Apply global robust scaling if requested
        if self.robust_scaling:
            print(f"\nApplying global robust scaling...")
            # Use global median and IQR for robust scaling
            global_iqr = self.global_q3 - self.global_q1
            
            if global_iqr > 0:
                lower_bound = self.global_q1 - 1.0 * global_iqr
                upper_bound = self.global_q3 + 1.0 * global_iqr
                print(f"    Global bounds: [{lower_bound:.2e}, {upper_bound:.2e}]")

                for name, data in self.distributions.items():
                    original_min, original_max = np.min(data), np.max(data)
                    mask = (data < lower_bound) | (data > upper_bound)
                    indices = np.where(mask)[0]
                    
                    if len(indices) > 0:
                        data_to_scale = data[mask]
                        distances = np.where(data_to_scale < lower_bound, 
                                           lower_bound - data_to_scale,
                                           data_to_scale - upper_bound)
                        
                        # Apply a smooth transformation to bring outliers closer
                        reduction_factor = 0.1
                        new_distances = distances * reduction_factor
                        
                        scaled_data = np.where(
                            data_to_scale < lower_bound,
                            lower_bound - new_distances,
                            upper_bound + new_distances
                        )
                        self.distributions[name][mask] = scaled_data
                        new_min, new_max = np.min(self.distributions[name]), np.max(self.distributions[name])
                        print(f"    {name}: Normalized {len(indices)} outliers, range [{original_min:.2e}, {original_max:.2e}] -> [{new_min:.2e}, {new_max:.2e}]")
                    
                    else:
                        print(f"    {name}: No outliers found")
            else:
                print(f"    Warning: Global IQR is zero, skipping robust scaling")
            
            print(f"  Recomputing global metrics after robust scaling...")
            self._compute_global_metrics()
        

        # Apply global range normalization if requested
        if self.normalize:
            print(f"  Applying global range normalization...")
            target_min = -1 
            target_max = 0
            
            if self.global_max > self.global_min:  # Avoid division by zero
                # Min-max normalization to target range using global min/max
                for name, data in self.distributions.items():
                    original_min, original_max = np.min(data), np.max(data)
                    self.distributions[name] = (data - self.global_min) / (self.global_max - self.global_min) * (target_max - target_min) + target_min
                    new_min, new_max = np.min(self.distributions[name]), np.max(self.distributions[name])
                    print(f"    {name}: [{original_min:.2e}, {original_max:.2e}] -> [{new_min:.2e}, {new_max:.2e}]")
                print(f"    Normalized all distributions to range [{target_min}, {target_max}] using global min/max")
                
                print(f"  Recomputing global metrics after range normalization...")
                self._compute_global_metrics()
            else:
                print(f"    Warning: Cannot normalize - all values are identical ({self.global_min})")
        


    def _merge_distributions_by_patterns(self, loaded_data: Dict[str, np.ndarray], loaded_ids: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Merge distributions based on specified patterns.
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


    def _export_processed_data(self, export_path: str) -> None:
        """
        Export processed values and their dataset membership to a CSV file.
        """
        if not self.distributions:
            raise ValueError("No distributions loaded. Call _load_distributions() first.")
        
        export_data = []
        for dataset_name, data in self.distributions.items():
            for value in data:
                export_data.append([value, dataset_name])
        
        df = pd.DataFrame(export_data, columns=['value', 'label'])
        df.to_csv(export_path, index=False)
        print(f"Exported {len(export_data)} values to: {export_path}")
        
        print(f"\nExport summary:")
        for dataset_name, data in self.distributions.items():
            print(f"  {dataset_name}: {len(data)} values")


    def _compute_individual_metrics(self) -> Dict[str, Dict]:
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
    

    def _compute_pairwise_metrics(self) -> Dict[str, Dict]:
        """Compute pairwise distribution metrics (Wasserstein, KS-test, Jensen-Shannon)."""
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


    def _export_summary_csv(self) -> None:
        """
        Export the individual metrics and pairwise metrics tables to CSV files.
        """
        if not self.metrics:
            raise ValueError("No individual metrics computed. Call _compute_individual_metrics() first.")

        # Individual metrics
        indiv_rows = []
        for name, m in self.metrics.items():
            indiv_rows.append({
                'distribution': name,
                'median': m['median'],
                'std': m['std'],
                'q25': m['q25'],
                'q75': m['q75'],
                'min': m['min'],
                'max': m['max'],
                'skewness': m['skewness'],
                'kurtosis': m['kurtosis'],
                'count': m['count'],
            })
        df_indiv = pd.DataFrame(indiv_rows)
        indiv_path = os.path.join(self.directory_path, f"distcomp_{self.metric}_individual_metrics.csv")
        df_indiv.to_csv(indiv_path, index=False)
        print(f"Individual metrics CSV saved to: {indiv_path}")

        # Pairwise metrics (optional)
        if self.pairwise_metrics:
            pair_rows = []
            for pair_name, pm in self.pairwise_metrics.items():
                pair_rows.append({
                    'pair': pair_name.replace('_vs_', ' vs '),
                    'wasserstein_distance': pm['wasserstein_distance'],
                    'jensen_shannon_divergence': pm['jensen_shannon_divergence'],
                    'ks_statistic': pm['ks_statistic'],
                    'ks_pvalue': pm['ks_pvalue'],
                    'mean_difference': pm['mean_difference'],
                })
            df_pair = pd.DataFrame(pair_rows)
            pair_path = os.path.join(self.directory_path, f"distcomp_{self.metric}_pairwise_metrics.csv")
            df_pair.to_csv(pair_path, index=False)
            print(f"Pairwise metrics CSV saved to: {pair_path}")


    def _export_train_reference_summary(self) -> None:
        """
        Create concise outputs focusing on how far each distribution is from the training distribution.
        Also prints a brief sorted (by Wasserstein) summary to stdout.
        """
        # Identify reference distribution containing 'train'
        train_name = next((k for k in self.distributions.keys() if 'train' in k.lower()), None)
        if train_name is None:
            print("No training distribution found (name containing 'train'). Skipping train-reference summary.")
            return

        if not self.pairwise_metrics:
            print("No pairwise metrics computed. Skipping train-reference summary.")
            return

        base_path = os.path.join(self.directory_path, 'distcomp_trainref')

        rows = []
        # Helper to fetch metrics regardless of pair order
        def get_pair_metrics(a: str, b: str) -> Optional[Dict[str, float]]:
            key1 = f"{a}_vs_{b}"
            key2 = f"{b}_vs_{a}"
            if key1 in self.pairwise_metrics:
                return self.pairwise_metrics[key1]
            if key2 in self.pairwise_metrics:
                return self.pairwise_metrics[key2]
            return None

        for other_name in self.distributions.keys():
            if other_name == train_name:
                continue
            pm = get_pair_metrics(train_name, other_name)
            if pm is None:
                continue
            rows.append({
                'run_id': str(self.directory_path),
                'metric': self.metric,
                'train_name': train_name,
                'other_name': other_name,
                'wasserstein_distance': pm['wasserstein_distance'],
                'jensen_shannon_divergence': pm['jensen_shannon_divergence'],
                'ks_statistic': pm['ks_statistic'],
                'ks_pvalue': pm['ks_pvalue'],
                'mean_difference': pm['mean_difference'],
                'remove_outliers': self.remove_outliers,
                'robust_scaling': self.robust_scaling,
                'normalize': self.normalize,
                'asinh_scaling': self.asinh_scaling,
            })

        if not rows:
            print("No train-vs-other rows to export.")
            return

        # Long format
        df_long = pd.DataFrame(rows)
        long_path = f"{base_path}_long.csv"
        df_long.to_csv(long_path, index=False)
        print(f"Train-reference (long) CSV saved to: {long_path}")

        # Wide format: single row per run/metric with columns per other_name and metric
        # Build dictionary for wide row
        wide_row = {
            'run_id': str(self.directory_path),
            'metric': self.metric,
            'train_name': train_name,
            'remove_outliers': self.remove_outliers,
            'robust_scaling': self.robust_scaling,
            'normalize': self.normalize,
            'asinh_scaling': self.asinh_scaling,
        }
        def sanitize(col: str) -> str:
            return ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in col)
        for r in rows:
            on = sanitize(r['other_name'])
            wide_row[f'wd__{on}'] = r['wasserstein_distance']
            wide_row[f'js__{on}'] = r['jensen_shannon_divergence']
            wide_row[f'ks__{on}'] = r['ks_statistic']
            wide_row[f'md__{on}'] = r['mean_difference']
        df_wide = pd.DataFrame([wide_row])
        wide_path = f"{base_path}_wide.csv"
        df_wide.to_csv(wide_path, index=False)
        print(f"Train-reference (wide) CSV saved to: {wide_path}")

        # Concise stdout: sort by Wasserstein
        rows_sorted = sorted(rows, key=lambda x: x['wasserstein_distance'])
        print("\nTrain vs others (sorted by Wasserstein):")
        for r in rows_sorted:
            print(
                f"  {train_name} vs {r['other_name']}: "
                f"WD={r['wasserstein_distance']:.4g}, "
                f"JS={r['jensen_shannon_divergence']:.4g}, "
                f"KS={r['ks_statistic']:.4g}, "
                f"MD={r['mean_difference']:.4g}"
            )


    def _load_error_distribution(self, error_dict: str) -> None:
        """Load error distribution from JSON file and compute global error statistics."""
        with open(error_dict, 'r') as f:
            self.error_values = json.load(f)

        self.global_error_min = np.min(np.abs(list(self.error_values.values())))
        self.global_error_max = np.max(np.abs(list(self.error_values.values())))
        print(f"Global error range: [{self.global_error_min:.2e}, {self.global_error_max:.2e}]")
        print(f"Loaded {len(self.error_values)} error values")

    def _get_error_data_for_distribution(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Load error values specifically for the given distribution based on IDs.
        '''
        if not self.error_values:
            raise ValueError("No error distribution loaded. Call _load_error_distribution() first.")
        
        ids = self.distributions_ids[name]
        error_data = []
        error_mask = np.zeros(len(ids), dtype=bool)
        for i, id_val in enumerate(ids):
            if id_val in self.error_values:
                error_data.append(np.abs(self.error_values[id_val])) # Absolute error values
                error_mask[i] = True
        return error_data, error_mask    


    def create_comparison_plots(self, 
                               figsize: Tuple[int, int] = (12, 8),
                               x_range = [None, None],
                               show_outliers: bool = False) -> None:
        """
        Create separate histogram, step plot, and individual distribution plots.
        Saves multiple files:
        - Combined histogram of all distributions
        - Combined step plot of all distributions  
        - Individual histogram for each distribution
        """

        if not self.distributions:
            raise ValueError("No distributions loaded. Call _load_distributions() first.")

        if x_range[0] is None:
            x_range[0] = self.global_min
        if x_range[1] is None:
            x_range[1] = self.global_max
        
        # Apply matplotlib settings before plotting to ensure consistency
        plt.rcParams.update({
            "font.family": "sans serif",
            "svg.fonttype": "none",
            'font.size': 12,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.titlesize': 13,
            'axes.linewidth': 0.5
        })

        save_path_base = os.path.join(self.directory_path, f'distcomp_{self.metric}')
        

        # ========== 1. COMBINED HISTOGRAM PLOT ==========
        fig_hist, ax_hist = plt.subplots(figsize=(figsize[0], figsize[1]*1.5))
        
        # Define consistent bin edges 
        bin_edges = np.linspace(x_range[0], x_range[1], 51)  # 50 bins from x_range[0] to x_range[1]
        
        for (name, data), color in zip(self.distributions.items(), self.colors):
            ax_hist.hist(data, bins=bin_edges, alpha=0.5, label=name, color=color, density=True)
        
        ax_hist.set_ylabel('Density')
        ax_hist.set_title(f'Combined Histogram - {self.metric}')
        ax_hist.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_hist.grid(True, alpha=0.3)
        # ax_hist.set_ylim(0, 7.5)
        ax_hist.set_xlim(x_range[0], x_range[1])
        
        # Save combined histogram
        hist_path = f"{save_path_base}_combined_histogram.svg"
        plt.savefig(hist_path, dpi=300, bbox_inches='tight', facecolor='white', format='svg')
        print(f"Saved combined histogram to: {hist_path}")
        plt.close(fig_hist)

        # ========== 2. INDIVIDUAL HISTOGRAMS FOR EACH DISTRIBUTION ==========
        # Define consistent bin edges (same as combined plot)
        bin_edges = np.linspace(x_range[0], x_range[1], 51)  # 50 bins from x_range[0] to x_range[1]
        
        for i, (name, data) in enumerate(self.distributions.items()):
            fig_indiv, ax_indiv = plt.subplots(figsize=figsize)
            
            color = self.colors[i]
            ax_indiv.hist(data, bins=bin_edges, alpha=0.6, color=color, density=True, linewidth=0.5)
            
            # Add vertical line at median
            median_val = np.median(data)
            ax_indiv.axvline(median_val, color='black', linestyle='--', linewidth=2, alpha=0.8)
            
            ax_indiv.set_ylabel('Density')
            # ax_indiv.set_title(f'{name} - {self.metric}')
            ax_indiv.grid(True, alpha=0.3)
            # ax_indiv.set_ylim(0, 6.5)
            ax_indiv.set_xlim(x_range[0], x_range[1])
            
            # Add some statistics as text
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            stats_text = f'Dataset: {name}\nMean: {mean_val:.3f}\nMedian: {median_val:.3f}\nN: {len(data):,}'
            ax_indiv.text(0.015, 0.95, stats_text, transform=ax_indiv.transAxes, fontsize=10,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=1.0))
            
            # Save individual histogram
            safe_name = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in name)
            indiv_path = f"{save_path_base}_histogram_{safe_name}.svg"
            plt.savefig(indiv_path, dpi=300, bbox_inches='tight', facecolor='white', format='svg')
            print(f"Saved individual histogram to: {indiv_path}")
            plt.close(fig_indiv)


        # ========== 3. Boxplot ==========
        # Create the plot
        fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]*1.3))

        # hide_box = ["cleansplit_3dd0_ood_test_similarity", "cleansplit_train_similarity"]
        hide_box = ["cleansplit_train_similarity"]

        data_for_box = []
        color_for_box = []
        labels = []
        for (name, data), color in zip(self.distributions.items(), self.colors):
            if name not in hide_box:
                data_for_box.append(data)
                color_for_box.append(color)
                labels.append(name)
        
        bp = ax.boxplot(data_for_box, 
                        labels=labels,
                        patch_artist=True, 
                        showfliers=show_outliers,
                        boxprops=boxprops,
                        medianprops=medianprops,
                        whiskerprops=whiskerprops,
                        flierprops=flierprops,
                        capprops=capprops
                        )
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], color_for_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel(self.metric.replace("_", " ").capitalize())
        ax.grid(True, alpha=0.3)
        
        # Move x-axis labels to the top
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        
        # Rotate x-axis labels if they're long
        if max(len(label) for label in labels) > 10:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='left')
        
        # Save the boxplot
        svg_path = f"{save_path_base}_boxplot" + ".svg"
        plt.savefig(svg_path, dpi=300, bbox_inches='tight', facecolor='white', format='svg')
        # png_path = f"{save_path_base}_boxplot" + ".png"
        # plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)


        # Export metrics tables
        self._export_summary_csv()
        
        print(f"\n✅ Created {len(self.distributions) + 2} separate plots:")
        print(f"   - 1 combined histogram")
        print(f"   - 1 combined step plot") 
        print(f"   - {len(self.distributions)} individual histograms")
        print(f"All plots saved with consistent width: {figsize[0]} units")
        

    def plot_heatmaps(self, 
                      save_path: str, 
                      figsize: Tuple[int, int] = (5, 5),
                      n_bins: int = 10,
                      error_range = [None, None],
                      metric_range = [None, None]
                      ) -> None:
        """
        Create individual heatmap plots for metric vs error density with quadrant analysis.
        Generates both density heatmaps and percentage-based quadrant plots.
        
        Args:
            save_path: Base path prefix for output files
            figsize: Figure size per subplot (default: (5, 5))
            n_bins: Number of histogram bins (default: 10)
            error_range: Y-axis range for error values
            metric_range: X-axis range for metric values
        """

        plt.rcParams.update({
        "font.family": "sans serif",
        "svg.fonttype": "none",          # keep text as text in SVG
        'font.size': 12,                 # Set the base font size 
        'axes.labelsize': 12,            # X and Y axis labels
        'xtick.labelsize': 10,           # X-axis tick labels
        'ytick.labelsize': 10,           # Y-axis tick labels
        'legend.fontsize': 10,           # Legend font size
        'axes.titlesize': 13,            # Plot title size
        'axes.linewidth': 0.5            # Thinner axis lines
        })  

        if not self.error_values:
            raise ValueError("No error distribution loaded. Call _load_error_distribution() first.")
        
        if not self.distributions or not self.distributions_ids:
            raise ValueError("No distributions loaded. Call _load_distributions() first.")

        if metric_range[0] is None:
            metric_range[0] = self.global_min
        if metric_range[1] is None:
            metric_range[1] = self.global_max
        print(f"Metric range for heatmaps: [{metric_range[0]:.2e}, {metric_range[1]:.2e}]")
        
        if error_range[0] is None:
            error_range[0] = min(0, self.global_error_min)
        if error_range[1] is None:
            error_range[1] = self.global_error_max
        print(f"Error range for heatmaps: [{error_range[0]:.2e}, {error_range[1]:.2e}]")


        # Create one figure per distribution
        for idx, (name, metric_data) in enumerate(self.distributions.items()):
            
            print(f"\nCreating heatmap for {name}")  
            fig, ax = plt.subplots(figsize=figsize)      
            
            # Get error data for this distribution
            error_data, error_mask = self._get_error_data_for_distribution(name)
            metric_data = np.array(metric_data)[error_mask] # Only include points with error values

            print(f"  N = {len(metric_data)} points with log-likelihoods and error values")
            print(f"  Range of metric data: [{np.min(metric_data):.2e}, {np.max(metric_data):.2e}]")
            print(f"  Range of error data: [{np.min(error_data):.2e}, {np.max(error_data):.2e}]")
            print(f"  RMSD: {np.sqrt(np.mean(np.square(error_data))):.2e}")
            
            
            # Create 2D histogram for heatmap
            try:
                # Define fixed bins for all heatmaps to ensure comparability
                x_bins = np.linspace(metric_range[0], metric_range[1], n_bins + 1)
                y_bins = np.linspace(error_range[0], error_range[1], n_bins + 1)
                
                # Create 2D histogram with fixed bins
                hist, x_edges, y_edges = np.histogram2d(metric_data, error_data, 
                                                    bins=[x_bins, y_bins], density=True)                

                if np.max(hist) > 0:
                    hist = hist / np.max(hist)

                im = ax.imshow(
                    hist.T,
                    origin='lower',
                    aspect='auto',
                    extent=[metric_range[0], metric_range[1], error_range[0], error_range[1]],
                    cmap=custom_cmap,
                    interpolation='none',
                    norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0),
                )
                ax.set_facecolor('white')
                
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('black')
                    spine.set_linewidth(1.0)
                ax.grid(False)
                
                # # Add colorbar for this subplot
                # cbar = plt.colorbar(im, ax=ax, shrink=0.82)
                # cbar.set_label('Density (0–1)')
                # cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
                
                ax.set_xlabel(f'{self.metric}', fontweight='bold')
                ax.set_ylabel('Error Value', fontweight='bold')
                # ax.set_title(f'{name}\n({len(metric_data)} points)', fontweight='bold')
                ax.set_xlim(metric_range[0], metric_range[1])  # Fixed x-axis limits
                ax.set_ylim(error_range[0], error_range[1])  # Fixed y-axis limits
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('black')
                    spine.set_linewidth(1.0)
                ax.grid(False)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error creating\nheatmap for {name}:\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
                ax.set_title(f'{name} - Error', fontweight='bold')
                ax.set_xlim(metric_range[0], metric_range[1])  # Fixed x-axis limits
                ax.set_ylim(error_range[0], error_range[1])  # Fixed y-axis limits
            
            safe_name = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in name)
            svg_path = f"{save_path}_{safe_name}.svg"
            plt.savefig(svg_path, dpi=300, bbox_inches='tight', facecolor='white', format='svg')
            # png_path = f"{save_path}_{safe_name}.png"
            # plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)


    def plot_exponential_scatterplot(self, 
                                      save_path: str, 
                                      distribution_names: List[str],
                                      max_points_for_fitting: int = 500,
                                      figsize: Tuple[int, int] = (8, 6),
                                      error_range = [None, None],
                                      metric_range = [None, None]
                                      ) -> None:
        """
        Create scatterplot with hierarchical exponential curve fitting (5-curve system).
        
        The scatterplot and curve fitting are based on the specified distributions,
        but error range analysis is performed on ALL loaded distributions for comprehensive statistics.
        
        Args:
            save_path: Path to save plot
            distribution_names: List of distribution names (or patterns) to use for plotting and curve fitting
            figsize: Figure size (default: (8, 6))
            error_range: Y-axis range for error values
            metric_range: X-axis range for metric values
        """

        # Apply matplotlib settings before plotting to ensure consistency
        plt.rcParams.update({
            "font.family": "sans serif",
            "svg.fonttype": "none",
            'font.size': 12,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.titlesize': 13,
            'axes.linewidth': 0.5
        })

        if not self.error_values:
            raise ValueError("No error distribution loaded. Call _load_error_distribution() first.")
        
        if not self.distributions or not self.distributions_ids:
            raise ValueError("No distributions loaded. Call _load_distributions() first.")

        if len(distribution_names) < 2:
            raise ValueError("At least 2 distributions must be specified.")


        # Validate distribution names and match patterns
        distribution_names_validated = distribution_names.copy()
        
        for i, name in enumerate(distribution_names):
            if name not in self.distributions:
                for dist_name in self.distributions.keys():
                    if name in dist_name:
                        distribution_names_validated[i] = dist_name
                        break
        distribution_names = [dist_name for dist_name in distribution_names_validated if dist_name in self.distributions]

        if metric_range[0] is None:
            metric_range[0] = self.global_min
        if metric_range[1] is None:
            metric_range[1] = self.global_max
        print(f"Metric range for scatterplot: [{metric_range[0]:.2e}, {metric_range[1]:.2e}]")
        
        if error_range[0] is None:
            error_range[0] = min(0, self.global_error_min)
        if error_range[1] is None:
            error_range[1] = self.global_error_max
        print(f"Error range for scatterplot: [{error_range[0]:.2e}, {error_range[1]:.2e}]")

        def exponential_func(x, a, b, c):
            return a * np.exp(b * x) + c

        def fit_exponential_curve(x: np.ndarray, 
                                  y: np.ndarray,
                                  x_range: Tuple[float, float]):
            """ Forces exponential fitting with |b| threshold constraints.
            Returns fitted curve data, equation string, and parameters."""
            try:
                # Define exponential function
                def exponential_func(x, a, b, c):
                    return a * np.exp(b * x) + c
                
                # Calculate data characteristics
                x_span = x_range[1] - x_range[0]
                y_span = np.max(y) - np.min(y)
                y_mean = np.mean(y)
                
                # Define minimum threshold for |b| to ensure meaningful exponential behavior
                min_b_threshold = 0.01 / x_span  # At least 1% change over x-span
                
                # Initial guesses with weak to strong exponential character
                initial_guesses = [
                    ([y_span, -5.0/x_span, np.min(y)], "Strong decay"),
                    ([y_span, 5.0/x_span, np.min(y)], "Strong growth"),
                    ([y_span, -8.0/x_span, np.min(y)], "Very strong decay"),
                    ([y_span, 8.0/x_span, np.min(y)], "Very strong growth"),
                    ([y_mean, -3.0/x_span, y_mean/2], "Moderate decay (centered)"),
                    ([y_mean, 3.0/x_span, y_mean/2], "Moderate growth (centered)"),
                    ([2*y_span, -0.5/x_span, np.min(y)], "Wide amplitude decay"),
                    ([2*y_span, 0.5/x_span, np.min(y)], "Wide amplitude growth"),
                ]
                
                best_exp_params = None
                best_exp_score = float('inf')
                best_guess_name = None
                
                print(f"        Exponential fit with |b| >= {min_b_threshold:.3e}...")
                
                # Try exponential fitting with different initial guesses and bounds
                for i, (p0, guess_name) in enumerate(initial_guesses):
                    try:
                        # Set parameter bounds to enforce meaningful exponential behavior
                        # bounds: ([a_min, b_min, c_min], [a_max, b_max, c_max])
                        # Force |b| to be above threshold by setting bounds
                        lower_bounds = [-np.inf, -np.inf, -np.inf]
                        upper_bounds = [np.inf, np.inf, np.inf]
                        
                        if p0[1] > 0:  # Growing exponential
                            lower_bounds[1] = min_b_threshold
                        else:  # Decaying exponential
                            upper_bounds[1] = -min_b_threshold
                        
                        popt_exp, _ = curve_fit(
                            exponential_func, x, y, 
                            p0=p0, 
                            bounds=(lower_bounds, upper_bounds),
                            maxfev=10000
                        )
                        y_pred_exp = exponential_func(x, *popt_exp)
                        mse_exp = np.mean((y - y_pred_exp)**2)
                        
                        # Check if parameters are reasonable
                        a, b, c = popt_exp
                        if not (np.isnan(a) or np.isnan(b) or np.isnan(c)) and abs(b) >= min_b_threshold:
                            print(f"        Guess {i+1} ({guess_name}): MSE={mse_exp:.3e}, b={b:.3e} ✓")
                            if mse_exp < best_exp_score:
                                best_exp_score = mse_exp
                                best_exp_params = popt_exp
                                best_guess_name = guess_name
                        else:
                            print(f"        Guess {i+1} ({guess_name}): Invalid parameters (b={b:.3e}, threshold={min_b_threshold:.3e})")
                    except Exception as e:
                        print(f"        Guess {i+1} ({guess_name}): Failed ({str(e)[:30]}...)")
                        continue
                
                # If bounded fitting failed, try unbounded fitting as a fallback
                if best_exp_params is None:
                    print(f"        Bounded fitting failed, trying unbounded...")

                    for i, (p0, guess_name) in enumerate(initial_guesses):
                        try:
                            popt_exp, _ = curve_fit(exponential_func, x, y, p0=p0, maxfev=10000)
                            y_pred_exp = exponential_func(x, *popt_exp)
                            mse_exp = np.mean((y - y_pred_exp)**2)
                            
                            a, b, c = popt_exp
                            if not (np.isnan(a) or np.isnan(b) or np.isnan(c)):
                                print(f"        Unbounded Guess {i+1} ({guess_name}): MSE={mse_exp:.3e}, b={b:.3e} ✓")
                                if mse_exp < best_exp_score:
                                    best_exp_score = mse_exp
                                    best_exp_params = popt_exp
                                    best_guess_name = guess_name
                        except:
                            continue
                
                if best_exp_params is not None:
                    a, b, c = best_exp_params
                    x_fit = np.linspace(x_range[0], x_range[1], 100)
                    y_fit = exponential_func(x_fit, a, b, c)
                    equation = f'y = {a:.2e} * exp({b:.2e} * x) + {c:.2e}'
                    
                    print(f"      ✓ EXPONENTIAL FIT: Best guess: {best_guess_name}")
                    print(f"        Final equation: {equation}")
                    print(f"        Exponential MSE: {best_exp_score:.3e}, |b|={abs(b):.3e} (>= {min_b_threshold:.3e})")
                    
                    return x_fit, y_fit, equation, (a, b, c)
                else:
                    print(f"      ✗ EXPONENTIAL FIT FAILED: Could not find meaningful exponential with |b| >= {min_b_threshold:.3e}")
                    raise Exception(f"No valid exponential fit with |b| >= {min_b_threshold:.3e}")
                
            except Exception as e:
                print(f"    Warning: Could not fit exponential curve: {e}")
                return np.array([]), np.array([]), "Fit failed", (None, None, None)

        def hierarchical_exponential_fit(x_data: np.ndarray, 
                                        y_data: np.ndarray, 
                                        x_range: Tuple[float, float]):
            """ 
            Creates 5-level hierarchical exponential fitting system:
            main → above/below → above-above/below-below curves.
            """
            fits = {}
            
            # 1. Main fit on all data
            print(f"  FITTING MAIN CURVE on {len(x_data)} points...")
            x_fit_main, y_fit_main, equation_main, params_main = fit_exponential_curve(x_data, y_data, x_range)
            
            if len(x_fit_main) > 0:
                fits['main'] = {
                    'x_fit': x_fit_main, 
                    'y_fit': y_fit_main, 
                    'equation': equation_main, 
                    'params': params_main,
                    'style': {'color': 'red', 'linewidth': 2, 'linestyle': '--', 'alpha': 0.9}
                }
                
                # Only continue with hierarchical fitting if we have exponential parameters
                if params_main[0] is not None:
                    print(f"     → Main fit successful, proceeding with hierarchical fitting...")
                    # Evaluate main fit at data points
                    a_main, b_main, c_main = params_main
                    y_main_at_data = exponential_func(x_data, a_main, b_main, c_main)
                    
                    # 2. Above-fit (data above main fit)
                    above_mask = y_data > y_main_at_data
                    print(f"  FITTING ABOVE-CURVE on {np.sum(above_mask)} points...")
                    if np.sum(above_mask) >= 3:
                        x_above = x_data[above_mask]
                        y_above = y_data[above_mask]
                        x_fit_above, y_fit_above, equation_above, params_above = fit_exponential_curve(x_above, y_above, x_range)
                        
                        if len(x_fit_above) > 0:
                            fits['above'] = {
                                'x_fit': x_fit_above, 
                                'y_fit': y_fit_above, 
                                'equation': equation_above, 
                                'params': params_above,
                                'style': {'color': 'black', 'linewidth': 1.5, 'linestyle': ':', 'alpha': 0.8}
                            }
                            
                            # 4. Above-above-fit (data above the above-fit)
                            if params_above[0] is not None:  # Only if we have exponential params
                                print(f"     → Above-fit successful, proceeding with above-above fitting...")
                                a_above, b_above, c_above = params_above
                                y_above_at_data = exponential_func(x_above, a_above, b_above, c_above)
                                above_above_mask = y_above > y_above_at_data
                                
                                print(f"  FITTING ABOVE-ABOVE-CURVE on {np.sum(above_above_mask)} points...")
                                if np.sum(above_above_mask) >= 3:
                                    x_above_above = x_above[above_above_mask]
                                    y_above_above = y_above[above_above_mask]
                                    x_fit_above_above, y_fit_above_above, equation_above_above, params_above_above = fit_exponential_curve(x_above_above, y_above_above, x_range)
                                    
                                    if len(x_fit_above_above) > 0:
                                        fits['above_above'] = {
                                            'x_fit': x_fit_above_above, 
                                            'y_fit': y_fit_above_above, 
                                            'equation': equation_above_above, 
                                            'params': params_above_above,
                                            'style': {'color': 'black', 'linewidth': 2, 'linestyle': '-', 'alpha': 0.9}
                                        }
                                else:
                                    print(f"     → Skipping above-above fit (insufficient points: {np.sum(above_above_mask)})")
                            else:
                                print(f"     → Skipping above-above fit (above-fit was linear)")
                        else:
                            print(f"     → Above-fit failed")
                    else:
                        print(f"     → Skipping above-fit (insufficient points: {np.sum(above_mask)})")
                    
                    # 3. Below-fit (data below main fit)
                    below_mask = y_data <= y_main_at_data
                    print(f"  FITTING BELOW-CURVE on {np.sum(below_mask)} points...")
                    if np.sum(below_mask) >= 3:
                        x_below = x_data[below_mask]
                        y_below = y_data[below_mask]
                        x_fit_below, y_fit_below, equation_below, params_below = fit_exponential_curve(x_below, y_below, x_range)
                        
                        if len(x_fit_below) > 0:
                            fits['below'] = {
                                'x_fit': x_fit_below, 
                                'y_fit': y_fit_below, 
                                'equation': equation_below, 
                                'params': params_below,
                                'style': {'color': 'black', 'linewidth': 1.5, 'linestyle': ':', 'alpha': 0.8}
                            }
                            
                            # 5. Below-below-fit (data below the below-fit)
                            if params_below[0] is not None:  # Only if we have exponential params
                                print(f"     → Below-fit successful, proceeding with below-below fitting...")
                                a_below, b_below, c_below = params_below
                                y_below_at_data = exponential_func(x_below, a_below, b_below, c_below)
                                below_below_mask = y_below < y_below_at_data
                                
                                print(f"  FITTING BELOW-BELOW-CURVE on {np.sum(below_below_mask)} points...")
                                if np.sum(below_below_mask) >= 3:
                                    x_below_below = x_below[below_below_mask]
                                    y_below_below = y_below[below_below_mask]
                                    x_fit_below_below, y_fit_below_below, equation_below_below, params_below_below = fit_exponential_curve(x_below_below, y_below_below, x_range)
                                    
                                    if len(x_fit_below_below) > 0:
                                        fits['below_below'] = {
                                            'x_fit': x_fit_below_below, 
                                            'y_fit': y_fit_below_below, 
                                            'equation': equation_below_below, 
                                            'params': params_below_below,
                                            'style': {'color': 'black', 'linewidth': 2, 'linestyle': '-', 'alpha': 0.9}
                                        }
                                else:
                                    print(f"     → Skipping below-below fit (insufficient points: {np.sum(below_below_mask)})")
                            else:
                                print(f"     → Skipping below-below fit (below-fit was linear)")
                        else:
                            print(f"     → Below-fit failed")
                    else:
                        print(f"     → Skipping below-fit (insufficient points: {np.sum(below_mask)})")
                else:
                    print(f"     → Main fit was linear, skipping hierarchical fitting")
            else:
                print(f"     → Main fit failed completely")
            return fits


        print(f"\nCombining distributions: {distribution_names}")
        print(f"Data characteristics:")
        print(f"  X-range (metric): [{metric_range[0]:.3e}, {metric_range[1]:.3e}] (span: {metric_range[1] - metric_range[0]:.3e})")
        print(f"  Y-range (error): [{error_range[0]:.3e}, {error_range[1]:.3e}] (span: {error_range[1] - error_range[0]:.3e})")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        max_points_for_plotting = 100
        distribution_data = {}
        metric_data_subsampled_all = []
        error_data_subsampled_all = []

        for i, (name, data) in enumerate(self.distributions.items()):
            if name not in distribution_names:
                continue
            error_data, error_mask = self._get_error_data_for_distribution(name)
            metric_data_filtered = np.array(data)[error_mask]
            
            if len(error_data) == 0:
                print(f"Warning: No error values found for distribution {name}")
                continue

            print(f"  {name}: {len(metric_data_filtered)} points")
            print(f"    Metric range: [{np.min(metric_data_filtered):.2e}, {np.max(metric_data_filtered):.2e}]")
            print(f"    Error range: [{np.min(error_data):.2e}, {np.max(error_data):.2e}]")
            print(f"    RMSD: {np.sqrt(np.mean(np.square(error_data))):.2e}")

            # Subsample to specific number od data points for plotting
            indeces = np.arange(len(metric_data_filtered))
            if len(indeces) > max_points_for_plotting:
                np.random.seed(42)
                indeces = np.random.choice(indeces, size=max_points_for_plotting, replace=False)
            metric_data_subsampled = metric_data_filtered[indeces]
            error_data_subsampled = np.array(error_data)[indeces]

            distribution_data[name] = {
                'metric_data': metric_data_subsampled,
                'error_data': error_data_subsampled,
                'color': self.colors[i % len(self.colors)]}       


            # Subsample to a specific number of data points for fitting
            indeces = np.arange(len(metric_data_filtered))
            if len(indeces) > max_points_for_fitting:
                np.random.seed(42)
                indeces = np.random.choice(indeces, size=max_points_for_fitting, replace=False)
            metric_data_subsampled = metric_data_filtered[indeces]
            error_data_subsampled = np.array(error_data)[indeces]

            metric_data_subsampled_all.extend(metric_data_subsampled)
            error_data_subsampled_all.extend(error_data_subsampled) 

        metric_data_subsampled_all = np.array(metric_data_subsampled_all)
        error_data_subsampled_all = np.array(error_data_subsampled_all)

        
        # Create scatterplot
        hide_markers = ["cleansplit_ood_train_combined", 
                        "cleansplit_3dd0_test"]
                        # ,"cleansplit_2vw5_test"]
        try:
            for name, data in distribution_data.items():
                if name not in hide_markers:
                    ax.scatter(
                        data['metric_data'],
                        data['error_data'],
                        alpha=0.6,
                        s=50,
                        color=data['color'],
                        edgecolors='none',
                        label=f'{name} ({len(data["metric_data"])} pts)'
                    )
                
            # Fit exponential curves using hierarchical approach
            if len(metric_data_subsampled_all) >= 3:
                print(f"  STARTING HIERARCHICAL EXPONENTIAL CURVE FITTING...")
                fits = hierarchical_exponential_fit(metric_data_subsampled_all, error_data_subsampled_all, metric_range)
                
                # Plot all fitted curves and provide summary
                fit_labels = {
                    'main': 'Main exponential fit',
                    'above': 'Above-fit exponential', 
                    'below': 'Below-fit exponential',
                    'above_above': 'Above-above-fit exponential',
                    'below_below': 'Below-below-fit exponential'
                }
                
                print(f"\n  FITTING SUMMARY:")
                curves_fitted = 0
                for fit_name in ['main', 'above', 'below', 'above_above', 'below_below']:
                    if fit_name in fits:
                        fit_data = fits[fit_name]
                        ax.plot(
                            fit_data['x_fit'], 
                            fit_data['y_fit'], 
                            # label=fit_labels[fit_name],
                            **fit_data['style']
                        )
                        curves_fitted += 1
                        # Determine fit type
                        fit_type = "Exponential" if fit_data['params'][0] is not None else "Linear"
                        print(f"    ✓ {fit_labels[fit_name]}: {fit_type}")
                        print(f"      → {fit_data['equation']}")
                    else:
                        print(f"    ✗ {fit_labels[fit_name]}: Not fitted (insufficient data or failed)")
                
                print(f"\n  TOTAL CURVES FITTED: {curves_fitted}/5")
                

                # Print statistics about data separation
                if 'main' in fits and fits['main']['params'][0] is not None:
                    a_main, b_main, c_main = fits['main']['params']
                    y_main_at_data = exponential_func(metric_data_subsampled_all, a_main, b_main, c_main)
                    above_mask = error_data_subsampled_all > y_main_at_data
                    below_mask = error_data_subsampled_all <= y_main_at_data
                    print(f"\n  DATA SEPARATION STATISTICS:")
                    print(f"    Main curve separates: {np.sum(above_mask)} points above, {np.sum(below_mask)} points below")
                    
                    if 'above' in fits and np.sum(above_mask) > 0 and fits['above']['params'][0] is not None:
                        x_above_data = metric_data_subsampled_all[above_mask]
                        y_above_data = error_data_subsampled_all[above_mask]
                        a_above, b_above, c_above = fits['above']['params']
                        y_above_at_data = exponential_func(x_above_data, a_above, b_above, c_above)
                        above_above_mask = y_above_data > y_above_at_data
                        print(f"    Above curve separates: {np.sum(above_above_mask)} points above, {np.sum(~above_above_mask)} points below")
                    
                    if 'below' in fits and np.sum(below_mask) > 0 and fits['below']['params'][0] is not None:
                        x_below_data = metric_data_subsampled_all[below_mask]
                        y_below_data = error_data_subsampled_all[below_mask]
                        a_below, b_below, c_below = fits['below']['params']
                        y_below_at_data = exponential_func(x_below_data, a_below, b_below, c_below)
                        below_below_mask = y_below_data < y_below_at_data
                        print(f"    Below curve separates: {np.sum(below_below_mask)} points below, {np.sum(~below_below_mask)} points above")
                
                # Analyze error ranges for ALL distributions
                self._analyze_exponential_error_ranges(fits, exponential_func)
                
            else:
                print(f"  INSUFFICIENT DATA: Only {len(metric_data_subsampled_all)} points available (minimum 3 required)")
            
            # Add legend for data points and fitted curves (positioned outside plot area)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Ensure background is white
            ax.set_facecolor('white')
            
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.0)
            ax.grid(False)
            
            # Customize subplot
            ax.set_xlabel(f'{self.metric}', fontweight='bold')
            ax.set_ylabel('Error Value', fontweight='bold')
            
            # Create title with distribution names
            # dist_names_str = ' + '.join(distribution_names)
            # ax.set_title(f'{dist_names_str}\n({len(metric_data_subsampled_all)} total points)')
            ax.set_xlim(metric_range[0], metric_range[1])  # Fixed x-axis limits
            ax.set_ylim(error_range[0], error_range[1])  # Fixed y-axis limits
            
            # Re-affirm frame styling after setting limits
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.0)
            ax.grid(False)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating\nscatterplot:\n{str(e)[:50]}...', 
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title('Error', fontweight='bold')
            ax.set_xlim(metric_range[0], metric_range[1])  # Fixed x-axis limits
            ax.set_ylim(error_range[0], error_range[1])  # Fixed y-axis limits
        
        # Save the plot
        svg_path = save_path + '.svg'
        plt.savefig(svg_path, dpi=300, bbox_inches='tight', facecolor='white', format='svg')
        # png_path = save_path + '.png'
        # plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved combined scatterplot to: {save_path}")


    def _analyze_exponential_error_ranges(self, fits: dict, exponential_func) -> None:
        """
        Analyze error ranges for ALL loaded distributions using fitted exponential curves.
        This provides comprehensive statistics across the entire dataset, not just the distributions
        used for curve fitting.
        
        Args:
            fits: Dictionary of fitted curves from hierarchical_exponential_fit
            exponential_func: Exponential function for curve evaluation
        """
        if not self.error_values:
            print("    No error values loaded - skipping error range analysis")
            return
        
        # Collect ALL data from ALL loaded distributions (not just selected ones)
        all_metric_data_full = []
        all_error_data_full = []
        distribution_counts = {}
        
        print(f"\n  ERROR RANGE PROBABILITY ANALYSIS (ALL DISTRIBUTIONS):")
        print(f"    {'='*70}")
        
        # Gather data from all distributions
        for name in self.distributions.keys():
            if 'train' not in name: # Exclude training distribution
                try:
                    error_data, error_mask = self._get_error_data_for_distribution(name)
                    metric_data_filtered = np.array(self.distributions[name])[error_mask]
                    
                    if len(error_data) > 0:
                        all_metric_data_full.extend(metric_data_filtered)
                        all_error_data_full.extend(error_data)
                        distribution_counts[name] = len(error_data)
                        print(f"    {name}: {len(error_data):,} points")
                except Exception as e:
                    print(f"    ⚠️  Failed to load data for {name}: {str(e)[:50]}...")
                    continue
        
        all_metric_data_full = np.array(all_metric_data_full)
        all_error_data_full = np.array(all_error_data_full)
        total_points = len(all_metric_data_full)
        
        if total_points == 0:
            print("    No valid data found across all distributions")
            return
        
        print(f"    TOTAL DATA ACROSS ALL DISTRIBUTIONS: {total_points:,} points")
        print(f"    {'-'*70}")
        
        # Evaluate all fitted curves at all data points for comprehensive analysis
        curve_evaluations = {}
        for curve_name in ['above_above', 'above', 'main', 'below', 'below_below']:
            if curve_name in fits and fits[curve_name]['params'][0] is not None:
                params = fits[curve_name]['params']
                curve_evaluations[curve_name] = exponential_func(all_metric_data_full, *params)
                print(f"    ✓ Evaluating {curve_name} curve on {total_points:,} points")
        
        if not curve_evaluations:
            print(f"    No valid fitted curves available for analysis")
            return
        
        # Define error ranges based on available curves
        error_ranges = []
        
        if 'above_above' in curve_evaluations and 'below_below' in curve_evaluations:
            # Range 1: Between above-above and below-below (widest range)
            above_above_vals = curve_evaluations['above_above']
            below_below_vals = curve_evaluations['below_below']
            between_extreme_mask = (all_error_data_full >= below_below_vals) & (all_error_data_full <= above_above_vals)
            between_extreme_count = np.sum(between_extreme_mask)
            between_extreme_prob = between_extreme_count / total_points * 100
            error_ranges.append({
                'name': 'Above-Above ↔ Below-Below',
                'description': 'Widest error envelope (most permissive)',
                'count': between_extreme_count,
                'probability': between_extreme_prob,
                'mask': between_extreme_mask
            })
        
        if 'above' in curve_evaluations and 'below' in curve_evaluations:
            # Range 2: Between above and below (medium range)
            above_vals = curve_evaluations['above']
            below_vals = curve_evaluations['below']
            between_mid_mask = (all_error_data_full >= below_vals) & (all_error_data_full <= above_vals)
            between_mid_count = np.sum(between_mid_mask)
            between_mid_prob = between_mid_count / total_points * 100
            error_ranges.append({
                'name': 'Above ↔ Below',
                'description': 'Medium error envelope (moderate)',
                'count': between_mid_count,
                'probability': between_mid_prob,
                'mask': between_mid_mask
            })
        
        # Additional specific ranges for more granular analysis
        if 'main' in curve_evaluations:
            main_vals = curve_evaluations['main']
            
            if 'above_above' in curve_evaluations:
                # Range 3: Between main and above-above
                main_to_above_above_mask = (all_error_data_full >= main_vals) & (all_error_data_full <= curve_evaluations['above_above'])
                main_to_above_above_count = np.sum(main_to_above_above_mask)
                main_to_above_above_prob = main_to_above_above_count / total_points * 100
                error_ranges.append({
                    'name': 'Main ↔ Above-Above',
                    'description': 'Upper error range (above average)',
                    'count': main_to_above_above_count,
                    'probability': main_to_above_above_prob,
                    'mask': main_to_above_above_mask
                })
            
            if 'below_below' in curve_evaluations:
                # Range 4: Between below-below and main
                below_below_to_main_mask = (all_error_data_full >= curve_evaluations['below_below']) & (all_error_data_full <= main_vals)
                below_below_to_main_count = np.sum(below_below_to_main_mask)
                below_below_to_main_prob = below_below_to_main_count / total_points * 100
                error_ranges.append({
                    'name': 'Below-Below ↔ Main',
                    'description': 'Lower error range (below average)',
                    'count': below_below_to_main_count,
                    'probability': below_below_to_main_prob,
                    'mask': below_below_to_main_mask
                })
        
        # Print comprehensive error range statistics
        if error_ranges:
            print(f"\n    ERROR ENVELOPE STATISTICS (ALL DISTRIBUTIONS):")
            print(f"    {'-'*70}")
            
            for i, range_info in enumerate(error_ranges, 1):
                print(f"\n    {i}. {range_info['name']} Envelope:")
                print(f"       {range_info['description']}")
                print(f"       Points within range: {range_info['count']:,} / {total_points:,}")
                print(f"       Probability: {range_info['probability']:.2f}%")
                
                # Calculate confidence level terminology
                if range_info['probability'] >= 95:
                    confidence_level = "Very High Confidence"
                elif range_info['probability'] >= 90:
                    confidence_level = "High Confidence"
                elif range_info['probability'] >= 80:
                    confidence_level = "Good Confidence"
                elif range_info['probability'] >= 70:
                    confidence_level = "Moderate Confidence"
                elif range_info['probability'] >= 50:
                    confidence_level = "Low Confidence"
                else:
                    confidence_level = "Very Low Confidence"
                
                print(f"       Confidence Level: {confidence_level}")
            
            # Summary table for quick reference
            print(f"\n    QUICK REFERENCE TABLE:")
            print(f"    {'-'*70}")
            print(f"    {'Error Range':<25} {'Points':<12} {'Probability':<12} {'Per Distrib.':<12}")
            print(f"    {'-'*70}")
            for range_info in error_ranges:
                short_name = range_info['name'].replace(' ↔ ', '-').replace('Above-Above', 'AA').replace('Below-Below', 'BB')
                avg_per_dist = range_info['count'] / len(distribution_counts) if len(distribution_counts) > 0 else 0
                print(f"    {short_name:<25} {range_info['count']:,<12} {range_info['probability']:.1f}%{'':<9} {avg_per_dist:.0f}")
            
            # Calculate outliers (points outside all envelopes)
            if 'above_above' in curve_evaluations and 'below_below' in curve_evaluations:
                outlier_mask = (all_error_data_full < curve_evaluations['below_below']) | (all_error_data_full > curve_evaluations['above_above'])
                outlier_count = np.sum(outlier_mask)
                outlier_prob = outlier_count / total_points * 100
                
                print(f"\n   OUTLIER ANALYSIS (ALL DISTRIBUTIONS):")
                print(f"    Points outside all envelopes: {outlier_count:,} ({outlier_prob:.2f}%)")
                
                if outlier_count > 0:
                    high_outliers = np.sum(all_error_data_full > curve_evaluations['above_above'])
                    low_outliers = np.sum(all_error_data_full < curve_evaluations['below_below'])
                    print(f"       High outliers (> Above-Above): {high_outliers:,} ({high_outliers/total_points*100:.2f}%)")
                    print(f"       Low outliers (< Below-Below): {low_outliers:,} ({low_outliers/total_points*100:.2f}%)")
            
            # Per-distribution breakdown (if multiple distributions)
            if len(distribution_counts) > 1:
                print(f"\n    PER-DISTRIBUTION BREAKDOWN:")
                print(f"    {'-'*70}")
                
                dist_idx = 0
                for dist_name, dist_count in distribution_counts.items():
                    # Extract data for this specific distribution
                    dist_start = dist_idx
                    dist_end = dist_idx + dist_count
                    
                    if 'above_above' in curve_evaluations and 'below_below' in curve_evaluations:
                        dist_extreme_mask = between_extreme_mask[dist_start:dist_end] if 'above_above' in curve_evaluations and 'below_below' in curve_evaluations else np.array([])
                        dist_extreme_count = np.sum(dist_extreme_mask) if len(dist_extreme_mask) > 0 else 0
                        dist_extreme_prob = (dist_extreme_count / dist_count * 100) if dist_count > 0 else 0
                        
                        print(f"       {dist_name}: {dist_extreme_count:,}/{dist_count:,} ({dist_extreme_prob:.1f}%) in widest envelope")
                    
                    dist_idx += dist_count
            
            print(f"    {'='*70}\n")
        else:
            print(f"    No error ranges could be calculated (insufficient curve fits)\n")


    def print_summary(self) -> None:
        """Print comprehensive analysis summary including statistics and pairwise comparisons."""
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

        if self.remove_outliers:
            print(f"✓ Outliers removed using IQR method")
        else:
            print("✗ No outliers removed")
            
        if self.robust_scaling:
            print(f"✓ Robust scaling: Applied global robust scaling to outliers")
        else:
            print("✗ No robust scaling applied")
            
        if self.asinh_scaling:
            print(f"✓ Asinh scaling: Applied global asinh transformation (preserves sign)")
        else:
            print("✗ No asinh scaling applied")
            
        if self.normalize:
            print(f"✓ Range normalization: Normalized to [-1,0] range")
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

    # Initialize comparator with preprocessing options
    comparator = DistributionComparator(
        args.directory, 
        metric=args.metric,
        remove_outliers=args.remove_outliers,
        robust_scaling=args.robust_scaling,
        normalize = args.normalize,
        merge_patterns=args.merge_patterns,
        asinh_scaling=args.asinh_scaling
    )
    
    # Load distributions and preprocess them
    print("Loading distributions...")
    comparator._load_distributions()
    
    # Compute metrics
    print("\nComputing individual metrics...")
    comparator._compute_individual_metrics()
    
    print("Computing pairwise metrics...")
    comparator._compute_pairwise_metrics()
    comparator._export_train_reference_summary()
    
    # Export data if requested
    if args.export:
        print("\nExporting processed data...")
        comparator._export_processed_data(args.export)
    
    # Create plots
    print("Creating comparison plot...")
    comparator.create_comparison_plots(
        figsize=(12, 2), 
        show_outliers=args.show_outliers)

    # Plots including error values
    if args.error_dict:

        error_type = os.path.basename(args.error_dict).split('.')[0]
        save_dir = os.path.join(args.directory, error_type)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print(f"Loading error distribution from {args.error_dict}...")
        comparator._load_error_distribution(args.error_dict)

        # Create heatmaps if requested
        if args.plot_heatmaps:
            print("\nCreating individual heatmap plots...")
            comparator.plot_heatmaps(
                save_path=os.path.join(save_dir, 'heatmaps_' + args.metric),
                figsize=(4, 3),
                n_bins=24,
                error_range=[None, 6]
                )
        
        # Create exponential scatterplot if requested
        if args.plot_exponential_scatterplot:
            print(f"\nCreating combined scatterplot with exponential curve fitting for: {args.plot_exponential_scatterplot}")
            try:
                comparator.plot_exponential_scatterplot(
                    save_path=os.path.join(save_dir, 'combined_exponential_scatterplot'),
                    distribution_names=args.plot_exponential_scatterplot,
                    figsize=(12, 8),
                    error_range=[None, 6]
                )
            except ValueError as e:
                print(f"Error creating exponential scatterplot: {e}")
                print(f"Available distributions: {list(comparator.distributions.keys())}")
    
    # Print summary
    comparator.print_summary()

if __name__ == "__main__":
    main() 