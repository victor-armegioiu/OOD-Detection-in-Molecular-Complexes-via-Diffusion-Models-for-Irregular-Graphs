"""
Correct Rate-In Implementation
Computes I(h_in; h_out) at each layer to measure information flow degradation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict


class RateInOptimizer:
    """
    Correct implementation of Rate-In algorithm from the paper.
    
    For each layer l, computes:
        ΔI_l = I_full^(l) - I(h_in^(l); h_out^(l))
    
    Where I(h_in; h_out) measures how well output preserves input information.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_info_loss: float = 0.10,
        convergence_tol: float = 0.02,
        max_iters: int = 20,
        n_mc_samples: int = 30
    ):
        self.model = model
        self.target_info_loss = target_info_loss  # ε in paper
        self.convergence_tol = convergence_tol    # δ in paper
        self.max_iters = max_iters                # N_max in paper
        self.n_mc_samples = n_mc_samples
        
        # Find all dropout layers
        self.dropout_modules = self._find_dropout_modules()
        
        # Map layer indices to hookable modules
        self.layer_modules = self._get_layer_modules()
        
        print(f"Initialized Rate-In:")
        print(f"  Dropout layers: {len(self.dropout_modules)}")
        print(f"  Hookable layers: {len(self.layer_modules)}")
    
    def _find_dropout_modules(self) -> List[nn.Module]:
        """Find all dropout layers"""
        dropouts = []
        
        def find_dropouts(module):
            for child in module.children():
                if isinstance(child, nn.Dropout):
                    dropouts.append(child)
                else:
                    find_dropouts(child)
        
        find_dropouts(self.model)
        return dropouts
    
    def _get_layer_modules(self) -> List[nn.Module]:
        """
        Get modules corresponding to each dropout layer.
        These are the modules we'll hook to capture h_in and h_out.
        """
        layers = []
        
        # For GEMS architecture
        if hasattr(self.model, 'NodeTransform'):
            layers.append(self.model.NodeTransform)
        
        # Layer 1 submodules
        if hasattr(self.model, 'layer1'):
            if hasattr(self.model.layer1, 'edge_model'):
                layers.append(self.model.layer1.edge_model)
            if hasattr(self.model.layer1, 'node_model'):
                layers.append(self.model.layer1.node_model)
            if hasattr(self.model.layer1, 'global_model'):
                layers.append(self.model.layer1.global_model)
        
        # Layer 2 submodules
        if hasattr(self.model, 'layer2'):
            if hasattr(self.model.layer2, 'edge_model'):
                layers.append(self.model.layer2.edge_model)
            if hasattr(self.model.layer2, 'node_model'):
                layers.append(self.model.layer2.node_model)
            if hasattr(self.model.layer2, 'global_model'):
                layers.append(self.model.layer2.global_model)
        
        # Final dropout
        if hasattr(self.model, 'dropout_layer'):
            layers.append(self.model.fc1)  # Hook the layer AFTER dropout
        
        return layers
    
    def set_dropout_rates(self, rates: List[float]):
        """Set dropout rate for each layer"""
        assert len(rates) == len(self.dropout_modules)
        for module, rate in zip(self.dropout_modules, rates):
            module.p = rate
    
    def enable_dropout(self):
        """Enable dropout in eval mode"""
        for module in self.dropout_modules:
            module.train()
    
    def disable_dropout(self):
        """Disable all dropout"""
        for module in self.dropout_modules:
            module.p = 0.0
            module.eval()
    
    # ========================================================================
    # Core: Mutual Information Estimation
    # ========================================================================
    
    def estimate_mutual_information(
        self,
        h_in: torch.Tensor,
        h_out: torch.Tensor
    ) -> float:
        """
        Estimate I(h_in; h_out) using correlation-based approximation.
        
        For Gaussian variables:
            I(X; Y) = -0.5 * log(1 - ρ²)
        
        where ρ is the correlation coefficient.
        
        Args:
            h_in: Input activations [N, D_in]
            h_out: Output activations [N, D_out]
        
        Returns:
            Mutual information estimate (in nats)
        """
        
        # Flatten to vectors
        h_in_flat = h_in.flatten().cpu().numpy()
        h_out_flat = h_out.flatten().cpu().numpy()
        
        # Need at least 2 samples for correlation
        if len(h_in_flat) < 2 or len(h_out_flat) < 2:
            return 0.0
        
        # Compute correlation
        # Note: h_in and h_out may have different dimensions
        # We compute correlation between their flattened versions
        try:
            # Standardize
            h_in_std = (h_in_flat - h_in_flat.mean()) / (h_in_flat.std() + 1e-8)
            h_out_std = (h_out_flat - h_out_flat.mean()) / (h_out_flat.std() + 1e-8)
            
            # If different sizes, truncate to minimum
            min_len = min(len(h_in_std), len(h_out_std))
            h_in_std = h_in_std[:min_len]
            h_out_std = h_out_std[:min_len]
            
            # Correlation
            correlation = np.corrcoef(h_in_std, h_out_std)[0, 1]
            
            # Handle NaN
            if np.isnan(correlation):
                return 0.0
            
            # Convert to MI: I(X;Y) = -0.5 * log(1 - ρ²)
            rho_squared = correlation ** 2
            rho_squared = np.clip(rho_squared, 0.0, 0.9999)  # Avoid log(0)
            
            mi = -0.5 * np.log(1 - rho_squared)
            
            return max(0.0, mi)
            
        except:
            return 0.0
    
    def capture_layer_activations(
        self,
        graph_batch,
        layer_idx: int,
        dropout_rate: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Capture h_in and h_out for a specific layer with given dropout rate.
        
        Returns:
            (h_in, h_out) - input and output activations
        """
        
        # Set dropout rates (only target layer has dropout)
        test_rates = [0.0] * len(self.dropout_modules)
        test_rates[layer_idx] = dropout_rate
        self.set_dropout_rates(test_rates)
        
        if dropout_rate > 0:
            self.enable_dropout()
        else:
            self.disable_dropout()
        
        # Storage for activations
        activations = {'input': None, 'output': None}
        
        # Hook function
        def hook_fn(module, input, output):
            # Store input (take first element if tuple)
            if isinstance(input, tuple):
                activations['input'] = input[0].detach().clone()
            else:
                activations['input'] = input.detach().clone()
            
            # Store output (take first element if tuple, e.g., for MetaLayer)
            if isinstance(output, tuple):
                activations['output'] = output[0].detach().clone()
            else:
                activations['output'] = output.detach().clone()
        
        # Register hook on target layer
        target_module = self.layer_modules[layer_idx]
        handle = target_module.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(graph_batch)
        
        # Remove hook
        handle.remove()
        
        h_in = activations['input']
        h_out = activations['output']
        
        # Handle None case
        if h_in is None or h_out is None:
            # Fallback: return dummy tensors
            h_in = torch.zeros(1, 1)
            h_out = torch.zeros(1, 1)
        
        return h_in, h_out
    
    # ========================================================================
    # Core: Information Loss Computation (Algorithm Line 7-8)
    # ========================================================================
    
    def compute_information_loss(
        self,
        graph_batch,
        layer_idx: int,
        dropout_rate: float
    ) -> float:
        """
        Compute ΔI_l = I_full^(l) - I(h_in^(l); h_out^(l))
        
        This is the core of the Rate-In algorithm (lines 7-8).
        
        Args:
            graph_batch: Input graph
            layer_idx: Which layer to measure
            dropout_rate: Dropout rate to apply
        
        Returns:
            Relative information loss: ΔI / I_full
        """
        
        # Step 1: Compute I_full (without dropout)
        h_in_full, h_out_full = self.capture_layer_activations(
            graph_batch, layer_idx, dropout_rate=0.0
        )
        I_full = self.estimate_mutual_information(h_in_full, h_out_full)
        
        # Step 2: Compute I(h_in; h_out) with dropout
        # Average over multiple samples for stability
        mi_samples = []
        for _ in range(5):  # 5 samples to average
            h_in, h_out = self.capture_layer_activations(
                graph_batch, layer_idx, dropout_rate
            )
            mi = self.estimate_mutual_information(h_in, h_out)
            mi_samples.append(mi)
        
        I_dropout = np.mean(mi_samples)
        
        # Step 3: Compute relative information loss
        if I_full < 1e-6:
            # No information to begin with
            return 0.0
        
        delta_I = I_full - I_dropout
        relative_loss = delta_I / I_full
        
        return np.clip(relative_loss, 0.0, 1.0)
    
    # ========================================================================
    # Rate Optimization (Algorithm Lines 4-15)
    # ========================================================================
    
    def optimize_single_layer(
        self,
        graph_batch,
        layer_idx: int,
        initial_rate: float = 0.1
    ) -> Tuple[float, int]:
        """
        Optimize dropout rate for a single layer.
        
        Implements the repeat-until loop (lines 4-15) for one layer.
        
        Returns:
            (optimized_rate, num_iterations)
        """
        
        p = initial_rate
        n = 0
        
        for n in range(self.max_iters):
            # Line 7-8: Compute information loss
            delta_I = self.compute_information_loss(graph_batch, layer_idx, p)
            
            # Line 9-13: Adjust rate based on loss
            if delta_I > self.target_info_loss:
                # Too much loss → decrease dropout (line 10)
                p *= 0.9
            else:
                # Too little loss → increase dropout (line 12)
                p = min(p * 1.1, 0.9)
            
            # Clip to valid range
            p = np.clip(p, 0.01, 0.9)
            
            # Line 15: Check convergence
            if abs(delta_I - self.target_info_loss) < self.convergence_tol:
                break
        
        return p, n + 1
    
    def optimize_all_layers(
        self,
        graph_batch
    ) -> Tuple[List[float], Dict]:
        """
        Optimize dropout rates for all layers.
        
        Implements the outer for loop (lines 2-16).
        
        Returns:
            (optimized_rates, diagnostics)
        """
        
        optimized_rates = []
        iterations = []
        
        # Line 2: for l = 1 to L do
        for layer_idx in range(len(self.dropout_modules)):
            rate, n_iters = self.optimize_single_layer(graph_batch, layer_idx)
            optimized_rates.append(rate)
            iterations.append(n_iters)
        
        # Compute diagnostics (for OOD score)
        diagnostics = {
            'mean_iters': np.mean(iterations),
            'rate_variance': np.var(optimized_rates),
            'rate_range': max(optimized_rates) - min(optimized_rates)
        }
        
        return optimized_rates, diagnostics
    
    # ========================================================================
    # MC Dropout Prediction (Algorithm Line 17)
    # ========================================================================
    
    def predict_with_mc_dropout(
        self,
        graph_batch,
        optimized_rates: List[float]
    ) -> Dict[str, float]:
        """
        Perform forward pass with adjusted dropout (line 17).
        
        Run multiple times to get MC dropout variance.
        
        Returns:
            Dictionary with prediction statistics
        """
        
        # Set optimized rates
        self.set_dropout_rates(optimized_rates)
        self.enable_dropout()
        
        # Line 17: Perform forward pass with adjusted dropout
        predictions = []
        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                out = self.model(graph_batch)
                predictions.append(out.cpu().item())
        
        predictions = np.array(predictions)
        
        return {
            'mean': predictions.mean(),
            'variance': predictions.var(),
            'std': predictions.std()
        }
    
    # ========================================================================
    # Complete Evaluation (Full Algorithm)
    # ========================================================================
    
    def evaluate_sample(
        self,
        graph_batch
    ) -> Tuple[float, Dict]:
        """
        Complete Rate-In evaluation for one sample.
        
        Implements full Algorithm 1.
        
        Returns:
            (ood_score, statistics)
        """
        
        # Lines 2-16: Optimize dropout rates
        optimized_rates, opt_diagnostics = self.optimize_all_layers(graph_batch)
        
        # Line 17: Get prediction with optimized dropout
        pred_stats = self.predict_with_mc_dropout(graph_batch, optimized_rates)
        
        # Compute OOD score (combine multiple signals)
        ood_score = (
            pred_stats['variance'] +                    # MC dropout uncertainty
            0.3 * opt_diagnostics['rate_variance'] +    # Rate inconsistency
            0.2 * (opt_diagnostics['mean_iters'] / self.max_iters)  # Convergence difficulty
        )
        
        # Package statistics
        statistics = {
            'optimized_rates': optimized_rates,
            'convergence_iters': opt_diagnostics['mean_iters'],
            'rate_variance': opt_diagnostics['rate_variance'],
            'prediction_mean': pred_stats['mean'],
            'prediction_variance': pred_stats['variance'],
            'ood_score': ood_score
        }
        
        return ood_score, statistics


# ============================================================================
# Helper: Batch Evaluation
# ============================================================================

def evaluate_dataset_ratein(
    model: nn.Module,
    dataloader,
    target_info_loss: float = 0.10,
    n_mc_samples: int = 30,
    device: str = 'cuda'
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Evaluate Rate-In on entire dataset.
    """
    from tqdm import tqdm
    
    model.eval()
    model.to(device)
    
    optimizer = RateInOptimizer(
        model=model,
        target_info_loss=target_info_loss,
        n_mc_samples=n_mc_samples
    )
    
    all_scores = []
    all_stats = []
    
    for batch in tqdm(dataloader, desc="Rate-In evaluation"):
        batch = batch.to(device)
        
        try:
            ood_score, stats = optimizer.evaluate_sample(batch)
            all_scores.append(ood_score)
            all_stats.append(stats)
        except Exception as e:
            print(f"Error: {e}")
            all_scores.append(float('inf'))
            all_stats.append(None)
    
    return np.array(all_scores), all_stats


# ============================================================================
# Test Function
# ============================================================================

def test_correct_ratein():
    """Test the correct implementation"""
    from gems import GEMS18d
    from torch_geometric.data import Data, Batch
    
    print("Testing Correct Rate-In Implementation")
    print("=" * 70)
    
    # Create model
    model = GEMS18d(
        dropout_prob=0.1,
        in_channels=1148,
        edge_dim=20,
        conv_dropout_prob=0.1
    )
    model.eval()
    
    # Create test graph
    x = torch.randn(50, 1148)
    edge_index = torch.randint(0, 50, (2, 100))
    edge_attr = torch.randn(100, 20)
    lig_emb = torch.randn(1, 384)
    
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, lig_emb=lig_emb)
    batch = Batch.from_data_list([graph])
    
    # Run Rate-In
    optimizer = RateInOptimizer(
        model=model,
        target_info_loss=0.10,
        n_mc_samples=30
    )
    
    print("\nRunning Rate-In evaluation...")
    ood_score, stats = optimizer.evaluate_sample(batch)
    
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    print(f"OOD Score:           {ood_score:.6f}")
    print(f"Prediction Mean:     {stats['prediction_mean']:.4f}")
    print(f"Prediction Variance: {stats['prediction_variance']:.6f}")
    print(f"Convergence Iters:   {stats['convergence_iters']:.1f}")
    print(f"Rate Variance:       {stats['rate_variance']:.6f}")
    print(f"\nOptimized Rates:")
    for i, rate in enumerate(stats['optimized_rates']):
        print(f"  Layer {i}: {rate:.4f}")
    print(f"{'='*70}")
    print("✅ Test complete!")


if __name__ == '__main__':
    test_correct_ratein()