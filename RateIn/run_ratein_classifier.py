"""
Comprehensive Rate-In OOD Evaluation Across Multiple Datasets

Evaluates Rate-In on multiple datasets, computes per-example scores,
generates visualizations, and saves detailed results.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    roc_curve,
    confusion_matrix
)

from ratein import RateInOptimizer


@dataclass
class ExampleResult:
    """Results for a single example"""
    example_id: int
    ood_score: float
    prediction_mean: float
    prediction_variance: float
    prediction_std: float
    convergence_iters: float
    rate_variance: float
    rate_range: float
    optimized_rates: List[float]
    mc_predictions: Optional[List[float]] = None  # All 30 MC samples


@dataclass
class DatasetResults:
    """Aggregated results for one dataset"""
    dataset_name: str
    n_examples: int
    mean_ood_score: float
    std_ood_score: float
    median_ood_score: float
    min_ood_score: float
    max_ood_score: float
    mean_prediction_variance: float
    mean_convergence_iters: float
    mean_rate_variance: float
    examples: List[ExampleResult]


class MultiDatasetRateInEvaluator:
    """
    Evaluates Rate-In across multiple datasets and generates comprehensive reports.
    """
    
    def __init__(
        self,
        model: nn.Module,
        output_dir: str = "./ratein_results",
        target_info_loss: float = 0.10,
        n_mc_samples: int = 30,
        device: str = 'cuda',
        save_mc_predictions: bool = False
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.save_mc_predictions = save_mc_predictions
        
        # Initialize Rate-In optimizer
        self.ratein = RateInOptimizer(
            model=model,
            target_info_loss=target_info_loss,
            n_mc_samples=n_mc_samples
        )
        
        # Storage for all results
        self.dataset_results: Dict[str, DatasetResults] = {}
        
        print(f"Initialized MultiDatasetRateInEvaluator")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Device: {device}")
        print(f"  MC samples: {n_mc_samples}")
    
    def evaluate_single_batch(
        self, 
        batch, 
        example_id: int
    ) -> ExampleResult:
        """
        Evaluate Rate-In on a single batch (could contain multiple graphs).
        
        Returns results for the batch.
        """
        
        batch = batch.to(self.device)
        
        try:
            # Run Rate-In evaluation
            ood_score, stats = self.ratein.evaluate_sample(batch)
            
            # Get MC predictions if requested
            mc_preds = None
            if self.save_mc_predictions:
                self.ratein.set_dropout_rates(stats['optimized_rates'])
                self.ratein.enable_dropout()
                mc_preds = []
                with torch.no_grad():
                    for _ in range(self.ratein.n_mc_samples):
                        out = self.model(batch)
                        mc_preds.append(out.cpu().item())
            
            # Package result
            result = ExampleResult(
                example_id=example_id,
                ood_score=ood_score,
                prediction_mean=stats['prediction_mean'],
                prediction_variance=stats['prediction_variance'],
                prediction_std=np.sqrt(stats['prediction_variance']),
                convergence_iters=stats['convergence_iters'],
                rate_variance=stats['rate_variance'],
                rate_range=max(stats['optimized_rates']) - min(stats['optimized_rates']),
                optimized_rates=stats['optimized_rates'],
                mc_predictions=mc_preds
            )
            
            return result
            
        except Exception as e:
            print(f"Error evaluating example {example_id}: {e}")
            # Return dummy result
            return ExampleResult(
                example_id=example_id,
                ood_score=float('inf'),
                prediction_mean=0.0,
                prediction_variance=0.0,
                prediction_std=0.0,
                convergence_iters=0.0,
                rate_variance=0.0,
                rate_range=0.0,
                optimized_rates=[0.0] * len(self.ratein.dropout_modules)
            )
    
    def evaluate_dataset(
        self,
        dataset_name: str,
        dataloader,
        max_examples: Optional[int] = None
    ) -> DatasetResults:
        """
        Evaluate Rate-In on an entire dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., "HIV_val", "CA_test")
            dataloader: PyTorch DataLoader
            max_examples: If set, only evaluate this many examples
        
        Returns:
            DatasetResults with all per-example and aggregate statistics
        """
        
        print(f"\n{'='*70}")
        print(f"Evaluating Dataset: {dataset_name}")
        print(f"{'='*70}")
        
        self.model.eval()
        self.model.to(self.device)
        
        example_results = []
        example_id = 0
        
        for batch in tqdm(dataloader, desc=f"Processing {dataset_name}"):
            if max_examples and example_id >= max_examples:
                break
            
            result = self.evaluate_single_batch(batch, example_id)
            example_results.append(result)
            example_id += 1
        
        # Compute aggregate statistics
        valid_scores = [r.ood_score for r in example_results if not np.isinf(r.ood_score)]
        
        if len(valid_scores) == 0:
            print(f"WARNING: No valid scores for {dataset_name}")
            valid_scores = [0.0]
        
        dataset_result = DatasetResults(
            dataset_name=dataset_name,
            n_examples=len(example_results),
            mean_ood_score=np.mean(valid_scores),
            std_ood_score=np.std(valid_scores),
            median_ood_score=np.median(valid_scores),
            min_ood_score=np.min(valid_scores),
            max_ood_score=np.max(valid_scores),
            mean_prediction_variance=np.mean([r.prediction_variance for r in example_results]),
            mean_convergence_iters=np.mean([r.convergence_iters for r in example_results]),
            mean_rate_variance=np.mean([r.rate_variance for r in example_results]),
            examples=example_results
        )
        
        # Store results
        self.dataset_results[dataset_name] = dataset_result
        
        # Print summary
        print(f"\nDataset Summary:")
        print(f"  Examples: {dataset_result.n_examples}")
        print(f"  OOD Score: {dataset_result.mean_ood_score:.6f} ± {dataset_result.std_ood_score:.6f}")
        print(f"  Median: {dataset_result.median_ood_score:.6f}")
        print(f"  Range: [{dataset_result.min_ood_score:.6f}, {dataset_result.max_ood_score:.6f}]")
        
        return dataset_result
    
    def evaluate_all_datasets(
        self,
        datasets: Dict[str, torch.utils.data.DataLoader],
        max_examples_per_dataset: Optional[int] = None
    ):
        """
        Evaluate Rate-In on multiple datasets.
        
        Args:
            datasets: Dict mapping dataset_name -> DataLoader
            max_examples_per_dataset: Limit examples per dataset (for quick testing)
        """
        
        print("\n" + "="*70)
        print("MULTI-DATASET RATE-IN EVALUATION")
        print("="*70)
        print(f"Datasets to evaluate: {list(datasets.keys())}")
        print(f"Max examples per dataset: {max_examples_per_dataset or 'All'}")
        
        for dataset_name, dataloader in datasets.items():
            self.evaluate_dataset(
                dataset_name=dataset_name,
                dataloader=dataloader,
                max_examples=max_examples_per_dataset
            )
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
    
    # ========================================================================
    # Binary OOD Detection Metrics
    # ========================================================================
    
    def compute_binary_metrics(
        self,
        id_dataset: str,
        ood_dataset: str
    ) -> Dict:
        """
        Compute binary OOD detection metrics (ID vs one OOD dataset).
        
        Returns:
            Dict with AUROC, AUPR, FPR@95TPR, etc.
        """
        
        if id_dataset not in self.dataset_results:
            raise ValueError(f"Dataset {id_dataset} not evaluated yet")
        if ood_dataset not in self.dataset_results:
            raise ValueError(f"Dataset {ood_dataset} not evaluated yet")
        
        # Get scores
        id_scores = [r.ood_score for r in self.dataset_results[id_dataset].examples 
                     if not np.isinf(r.ood_score)]
        ood_scores = [r.ood_score for r in self.dataset_results[ood_dataset].examples
                      if not np.isinf(r.ood_score)]
        
        # Combine
        all_scores = np.array(id_scores + ood_scores)
        labels = np.array([0] * len(id_scores) + [1] * len(ood_scores))
        
        # Compute metrics
        auroc = roc_auc_score(labels, all_scores)
        aupr = average_precision_score(labels, all_scores)
        
        fpr, tpr, thresholds = roc_curve(labels, all_scores)
        fpr_at_95tpr = fpr[np.argmax(tpr >= 0.95)] if np.any(tpr >= 0.95) else 1.0
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            'id_dataset': id_dataset,
            'ood_dataset': ood_dataset,
            'auroc': auroc,
            'aupr': aupr,
            'fpr_at_95tpr': fpr_at_95tpr,
            'optimal_threshold': optimal_threshold,
            'optimal_tpr': tpr[optimal_idx],
            'optimal_fpr': fpr[optimal_idx],
            'n_id': len(id_scores),
            'n_ood': len(ood_scores)
        }
    
    def compute_all_binary_metrics(
        self,
        id_dataset: str
    ) -> Dict[str, Dict]:
        """
        Compute binary metrics for ID vs all OOD datasets.
        
        Returns:
            Dict mapping ood_dataset_name -> metrics
        """
        
        results = {}
        
        for dataset_name in self.dataset_results.keys():
            if dataset_name != id_dataset:
                metrics = self.compute_binary_metrics(id_dataset, dataset_name)
                results[dataset_name] = metrics
        
        return results
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    def plot_score_distributions(self, id_dataset: str, save: bool = True):
        """
        Plot OOD score distributions for all datasets.
        """
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Histogram
        ax = axes[0]
        for dataset_name, results in self.dataset_results.items():
            scores = [r.ood_score for r in results.examples if not np.isinf(r.ood_score)]
            label = f"{dataset_name} (n={len(scores)})"
            color = 'blue' if dataset_name == id_dataset else 'red'
            alpha = 0.7 if dataset_name == id_dataset else 0.5
            
            ax.hist(scores, bins=30, alpha=alpha, label=label, color=color)
        
        ax.set_xlabel('OOD Score', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Rate-In OOD Score Distributions', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Box plot
        ax = axes[1]
        data_to_plot = []
        labels = []
        colors = []
        
        for dataset_name, results in self.dataset_results.items():
            scores = [r.ood_score for r in results.examples if not np.isinf(r.ood_score)]
            data_to_plot.append(scores)
            labels.append(dataset_name)
            colors.append('lightblue' if dataset_name == id_dataset else 'lightcoral')
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('OOD Score', fontsize=12)
        ax.set_title('Rate-In OOD Score Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'score_distributions.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_roc_curves(self, id_dataset: str, save: bool = True):
        """
        Plot ROC curves for ID vs all OOD datasets.
        """
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for dataset_name in self.dataset_results.keys():
            if dataset_name == id_dataset:
                continue
            
            # Get scores
            id_scores = [r.ood_score for r in self.dataset_results[id_dataset].examples 
                         if not np.isinf(r.ood_score)]
            ood_scores = [r.ood_score for r in self.dataset_results[dataset_name].examples
                          if not np.isinf(r.ood_score)]
            
            all_scores = np.array(id_scores + ood_scores)
            labels = np.array([0] * len(id_scores) + [1] * len(ood_scores))
            
            # Compute ROC
            fpr, tpr, _ = roc_curve(labels, all_scores)
            auroc = roc_auc_score(labels, all_scores)
            
            ax.plot(fpr, tpr, linewidth=2, label=f'{dataset_name} (AUROC={auroc:.3f})')
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curves: {id_dataset} (ID) vs OOD Datasets', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'roc_curves.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_metric_comparison(self, id_dataset: str, save: bool = True):
        """
        Bar chart comparing AUROC/AUPR across OOD datasets.
        """
        
        binary_metrics = self.compute_all_binary_metrics(id_dataset)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        datasets = list(binary_metrics.keys())
        aurocs = [binary_metrics[d]['auroc'] for d in datasets]
        auprs = [binary_metrics[d]['aupr'] for d in datasets]
        
        # AUROC
        ax = axes[0]
        bars = ax.bar(range(len(datasets)), aurocs, color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title('OOD Detection: AUROC Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.axhline(y=0.5, color='r', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, val in zip(bars, aurocs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # AUPR
        ax = axes[1]
        bars = ax.bar(range(len(datasets)), auprs, color='coral', alpha=0.8)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.set_ylabel('AUPR', fontsize=12)
        ax.set_title('OOD Detection: AUPR Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, auprs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'metric_comparison.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_prediction_variance_vs_score(self, save: bool = True):
        """
        Scatter plot: Prediction variance vs OOD score for all datasets.
        """
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for dataset_name, results in self.dataset_results.items():
            variances = [r.prediction_variance for r in results.examples]
            scores = [r.ood_score for r in results.examples if not np.isinf(r.ood_score)]
            
            # Truncate if mismatch (shouldn't happen but safe)
            n = min(len(variances), len(scores))
            variances = variances[:n]
            scores = scores[:n]
            
            ax.scatter(variances, scores, alpha=0.6, s=30, label=dataset_name)
        
        ax.set_xlabel('Prediction Variance (MC Dropout)', fontsize=12)
        ax.set_ylabel('OOD Score', fontsize=12)
        ax.set_title('Prediction Variance vs OOD Score', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'variance_vs_score.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.close()
    
    def generate_all_plots(self, id_dataset: str):
        """Generate all visualizations."""
        
        print(f"\nGenerating visualizations...")
        print(f"  ID dataset: {id_dataset}")
        
        self.plot_score_distributions(id_dataset)
        self.plot_roc_curves(id_dataset)
        self.plot_metric_comparison(id_dataset)
        self.plot_prediction_variance_vs_score()
        
        print(f"All plots saved to: {self.output_dir}")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
    def save_per_example_csv(self):
        """Save per-example results to CSV."""
        
        filepath = self.output_dir / 'per_example_results.csv'
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = [
                'dataset', 'example_id', 'ood_score', 
                'prediction_mean', 'prediction_variance', 'prediction_std',
                'convergence_iters', 'rate_variance', 'rate_range'
            ]
            # Add per-layer rates
            if self.dataset_results:
                first_dataset = list(self.dataset_results.values())[0]
                n_layers = len(first_dataset.examples[0].optimized_rates)
                header.extend([f'rate_layer_{i}' for i in range(n_layers)])
            
            writer.writerow(header)
            
            # Data
            for dataset_name, results in self.dataset_results.items():
                for example in results.examples:
                    row = [
                        dataset_name,
                        example.example_id,
                        example.ood_score,
                        example.prediction_mean,
                        example.prediction_variance,
                        example.prediction_std,
                        example.convergence_iters,
                        example.rate_variance,
                        example.rate_range
                    ]
                    row.extend(example.optimized_rates)
                    writer.writerow(row)
        
        print(f"Saved: {filepath}")
    
    def save_dataset_summary_csv(self):
        """Save per-dataset summary statistics to CSV."""
        
        filepath = self.output_dir / 'dataset_summary.csv'
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            header = [
                'dataset', 'n_examples',
                'mean_ood_score', 'std_ood_score', 'median_ood_score',
                'min_ood_score', 'max_ood_score',
                'mean_prediction_variance', 'mean_convergence_iters', 'mean_rate_variance'
            ]
            writer.writerow(header)
            
            for dataset_name, results in self.dataset_results.items():
                row = [
                    results.dataset_name,
                    results.n_examples,
                    results.mean_ood_score,
                    results.std_ood_score,
                    results.median_ood_score,
                    results.min_ood_score,
                    results.max_ood_score,
                    results.mean_prediction_variance,
                    results.mean_convergence_iters,
                    results.mean_rate_variance
                ]
                writer.writerow(row)
        
        print(f"Saved: {filepath}")
    
    def save_binary_metrics_csv(self, id_dataset: str):
        """Save binary OOD detection metrics to CSV."""
        
        filepath = self.output_dir / 'binary_metrics.csv'
        
        binary_metrics = self.compute_all_binary_metrics(id_dataset)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            header = [
                'id_dataset', 'ood_dataset', 'n_id', 'n_ood',
                'auroc', 'aupr', 'fpr_at_95tpr',
                'optimal_threshold', 'optimal_tpr', 'optimal_fpr'
            ]
            writer.writerow(header)
            
            for ood_dataset, metrics in binary_metrics.items():
                row = [
                    metrics['id_dataset'],
                    metrics['ood_dataset'],
                    metrics['n_id'],
                    metrics['n_ood'],
                    metrics['auroc'],
                    metrics['aupr'],
                    metrics['fpr_at_95tpr'],
                    metrics['optimal_threshold'],
                    metrics['optimal_tpr'],
                    metrics['optimal_fpr']
                ]
                writer.writerow(row)
        
        print(f"Saved: {filepath}")
    
    def save_detailed_json(self):
        """Save all results to JSON (including per-example details)."""
        
        filepath = self.output_dir / 'detailed_results.json'
        
        # Convert to serializable format
        output = {}
        for dataset_name, results in self.dataset_results.items():
            output[dataset_name] = {
                'summary': {
                    'n_examples': results.n_examples,
                    'mean_ood_score': results.mean_ood_score,
                    'std_ood_score': results.std_ood_score,
                    'median_ood_score': results.median_ood_score,
                    'min_ood_score': results.min_ood_score,
                    'max_ood_score': results.max_ood_score,
                },
                'examples': [asdict(ex) for ex in results.examples]
            }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Saved: {filepath}")
    
    def save_all_results(self, id_dataset: str):
        """Save all results (CSV + JSON)."""
        
        print(f"\nSaving results to: {self.output_dir}")
        
        self.save_per_example_csv()
        self.save_dataset_summary_csv()
        self.save_binary_metrics_csv(id_dataset)
        self.save_detailed_json()
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'id_dataset': id_dataset,
            'ood_datasets': [d for d in self.dataset_results.keys() if d != id_dataset],
            'n_mc_samples': self.ratein.n_mc_samples,
            'target_info_loss': self.ratein.target_info_loss,
            'max_iters': self.ratein.max_iters
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"All results saved!")
    
    # ========================================================================
    # Complete Pipeline
    # ========================================================================
    
    def run_complete_evaluation(
        self,
        datasets: Dict[str, torch.utils.data.DataLoader],
        id_dataset: str,
        max_examples_per_dataset: Optional[int] = None
    ):
        """
        Run complete evaluation pipeline:
          1. Evaluate all datasets
          2. Compute metrics
          3. Generate plots
          4. Save everything
        
        Args:
            datasets: Dict of {dataset_name: dataloader}
            id_dataset: Which dataset is in-distribution
            max_examples_per_dataset: Limit for quick testing
        """
        
        # Step 1: Evaluate all datasets
        self.evaluate_all_datasets(datasets, max_examples_per_dataset)
        
        # Step 2: Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        for dataset_name, results in self.dataset_results.items():
            status = "ID" if dataset_name == id_dataset else "OOD"
            print(f"\n{dataset_name} ({status}):")
            print(f"  Examples: {results.n_examples}")
            print(f"  OOD Score: {results.mean_ood_score:.6f} ± {results.std_ood_score:.6f}")
        
        # Step 3: Binary metrics
        print("\n" + "="*70)
        print(f"BINARY OOD DETECTION METRICS (ID = {id_dataset})")
        print("="*70)
        
        binary_metrics = self.compute_all_binary_metrics(id_dataset)
        for ood_dataset, metrics in binary_metrics.items():
            print(f"\n{id_dataset} vs {ood_dataset}:")
            print(f"  AUROC:       {metrics['auroc']:.4f}")
            print(f"  AUPR:        {metrics['aupr']:.4f}")
            print(f"  FPR@95TPR:   {metrics['fpr_at_95tpr']:.4f}")
        
        # Step 4: Generate visualizations
        self.generate_all_plots(id_dataset)
        
        # Step 5: Save everything
        self.save_all_results(id_dataset)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)
        print(f"All results saved to: {self.output_dir}")


# ============================================================================
# Example Usage
# ============================================================================

def example_multi_dataset_evaluation():
    """
    Complete example with synthetic data.
    """
    from gems import GEMS18d
    from torch_geometric.data import Data, Batch, DataLoader
    import random
    
    print("Multi-Dataset Rate-In Evaluation Example")
    print("="*70)
    
    # Create model
    model = GEMS18d(
        dropout_prob=0.1,
        in_channels=1148,
        edge_dim=20,
        conv_dropout_prob=0.1
    )
    model.eval()
    
    # ========================================================================
    # Create synthetic datasets (i just made them somewhat different)
    # ========================================================================
    
    def create_synthetic_dataset(n_examples, distribution_params):
        """Helper to create synthetic graph dataset"""
        graphs = []
        for _ in range(n_examples):
            num_nodes = random.randint(*distribution_params['nodes'])
            num_edges = num_nodes * 2
            
            x = torch.randn(num_nodes, 1148) * distribution_params['x_scale']
            x += distribution_params['x_shift']
            
            edge_index = torch.randint(0, num_nodes, (2, num_edges))
            edge_attr = torch.randn(num_edges, 20) * distribution_params['edge_scale']
            lig_emb = torch.randn(1, 384) * distribution_params['emb_scale']
            
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, lig_emb=lig_emb)
            graphs.append(graph)
        
        return DataLoader(graphs, batch_size=1, shuffle=False)
    
    # Create datasets with different distributions
    datasets = {
        'HIV_protease_val': create_synthetic_dataset(30, {
            'nodes': (45, 55), 'x_scale': 0.5, 'x_shift': 0.0,
            'edge_scale': 0.5, 'emb_scale': 0.5
        }),
        'Carbonic_Anhydrase': create_synthetic_dataset(30, {
            'nodes': (35, 70), 'x_scale': 1.5, 'x_shift': 2.0,
            'edge_scale': 1.2, 'emb_scale': 1.5
        }),
        'Kinase': create_synthetic_dataset(30, {
            'nodes': (40, 80), 'x_scale': 1.2, 'x_shift': 1.5,
            'edge_scale': 1.0, 'emb_scale': 1.2
        }),
        'CASF': create_synthetic_dataset(30, {
            'nodes': (50, 90), 'x_scale': 1.8, 'x_shift': 3.0,
            'edge_scale': 1.5, 'emb_scale': 2.0
        }),
    }
    
    # ========================================================================
    # Run complete evaluation
    # ========================================================================
    
    evaluator = MultiDatasetRateInEvaluator(
        model=model,
        output_dir="./ratein_multi_dataset_results",
        target_info_loss=0.10,
        n_mc_samples=30,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_mc_predictions=False  # Set True to save all 30 MC predictions
    )
    
    evaluator.run_complete_evaluation(
        datasets=datasets,
        id_dataset='HIV_protease_val',
        max_examples_per_dataset=None  # Set to e.g., 10 for quick test
    )
    
    print("\n✅ Example complete! Check ./ratein_multi_dataset_results/")


def main():

    import os
    from gems import GEMS18d
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader

    def load_model_state(model, state_dict_path):
        model.load_state_dict(torch.load(state_dict_path))
        model.eval()  # Set the model to evaluation mode
        return model

    d = 3 # dropout
    cd = 0 # convolution layer dropout

    model_folder = "/cluster/work/math/dagraber/GEMS/experiments/PLINDER_OOD_DIFF/B6AE_PLINDER_OOD_DIFFPL_kikdic"
    model_path = os.path.join(model_folder, f"GEMS18d/GEMS18d_B6AE_PLINDER_OOD_DIFFPL_kikdic_d0{d}0{cd}_42/Fold0/GEMS18d_B6AE_PLINDER_OOD_DIFFPL_kikdic_d0{d}0{cd}_42_f0_best_stdict.pt")

    datasets_paths = {
        
        
        'casf_2016': os.path.join(model_folder, 'dataset_casf2016.pt'),
        '1sqa': os.path.join(model_folder, 'dataset_1sqa_ood_test.pt'),
        '2p15': os.path.join(model_folder, 'dataset_2p15_ood_test.pt'),
        '2vw5': os.path.join(model_folder, 'dataset_2vw5_ood_test.pt'),
        '3dd0': os.path.join(model_folder, 'dataset_3dd0_ood_test.pt'),
        '3f3e': os.path.join(model_folder, 'dataset_3f3e_ood_test.pt'),
        '3o9i': os.path.join(model_folder, 'dataset_3o9i_ood_test.pt'),
        '1nvq': os.path.join(model_folder, 'dataset_1nvq_ood_test.pt'),
        'train': os.path.join(model_folder, 'dataset_train.pt'),
        'validation': os.path.join(model_folder, 'dataset_validation.pt')
        }
    
    datasets = {}
    for name, path in datasets_paths.items():
        dataset = torch.load(path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        datasets[name] = dataloader

    # Create model
    model = GEMS18d(
        dropout_prob=float(f"0.{d}"),
        in_channels=1148,
        edge_dim=20,
        conv_dropout_prob=float(f"0.{cd}")
    )
    model = load_model_state(model, model_path)



    print("Multi-Dataset Rate-In Evaluation")
    print(f"Dropout 0.{d}")
    print(f"Conv Dropout 0.{cd}")
    print("="*70)
    
    
    # ========================================================================
    # Run complete evaluation
    # ========================================================================
    
    evaluator = MultiDatasetRateInEvaluator(
        model=model,
        output_dir=f"./ratein_multi_dataset_results_{d}",
        target_info_loss=0.10,
        n_mc_samples=30,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_mc_predictions=False  # Set True to save all 30 MC predictions
    )
    
    evaluator.run_complete_evaluation(
        datasets=datasets,
        id_dataset='train',
        max_examples_per_dataset=None  # Set to e.g., 10 for quick test
    )
    
    print("\n✅ Example complete! Check ./ratein_multi_dataset_results/")


if __name__ == '__main__':
    main()
    # example_multi_dataset_evaluation()