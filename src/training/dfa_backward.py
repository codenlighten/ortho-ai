"""
DFA Backward Hook for PyTorch Autograd Integration

Copyright (c) 2025 Gregory Ward - SmartLedger.Technology
Licensed under the MIT License (see LICENSE file)

Implements Direct Feedback Alignment by:
1. Storing activations during forward pass
2. Computing local errors: e_l = B_l δ_L
3. Replacing gradients during backward pass: ∇W_l = e_l a_{l-1}^T
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any

try:
    from .dfa_feedback import DFAFeedbackMatrix
except ImportError:
    # For standalone testing
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from dfa_feedback import DFAFeedbackMatrix


class DFABackwardHook:
    """
    PyTorch backward hook for Direct Feedback Alignment.
    
    Intercepts the backward pass and replaces gradients with DFA updates:
    - Standard BP: ∇W_l = δ_l a_{l-1}^T (where δ_l propagates through layers)
    - DFA: ∇W_l = e_l a_{l-1}^T (where e_l = B_l δ_L comes directly from output)
    
    Args:
        feedback_matrix: DFAFeedbackMatrix managing fixed random projections
        enabled: Whether DFA is active (can be disabled for comparison)
        store_statistics: Whether to track gradient statistics for diagnostics
    """
    
    def __init__(
        self,
        feedback_matrix: DFAFeedbackMatrix,
        enabled: bool = True,
        store_statistics: bool = True
    ):
        self.feedback_matrix = feedback_matrix
        self.enabled = enabled
        self.store_statistics = store_statistics
        
        # Storage for activations during forward pass
        self._activations: Dict[int, torch.Tensor] = {}
        
        # Storage for hook handles (for removal)
        self._forward_hooks: List[Any] = []
        self._backward_hooks: List[Any] = []
        
        # Statistics for diagnostics
        self._grad_stats: Dict[str, float] = {}
        self._global_error: Optional[torch.Tensor] = None
    
    def register_hooks(
        self,
        modules: List[nn.Module],
        layer_indices: Optional[List[int]] = None
    ) -> None:
        """
        Register forward and backward hooks on modules.
        
        Args:
            modules: List of modules to hook (e.g., Linear layers in transformer)
            layer_indices: Optional layer indices (if None, uses 0, 1, 2, ...)
        """
        if layer_indices is None:
            layer_indices = list(range(len(modules)))
        
        assert len(modules) == len(layer_indices), \
            "Number of modules must match number of layer indices"
        
        # Clear existing hooks
        self.remove_hooks()
        
        # Register hooks for each module
        for module, layer_idx in zip(modules, layer_indices):
            # Forward hook to store activations
            fwd_hook = module.register_forward_hook(
                lambda mod, inp, out, idx=layer_idx: self._forward_hook(mod, inp, out, idx)
            )
            self._forward_hooks.append(fwd_hook)
            
            # Backward hook to replace gradients
            bwd_hook = module.register_full_backward_hook(
                lambda mod, grad_inp, grad_out, idx=layer_idx: 
                    self._backward_hook(mod, grad_inp, grad_out, idx)
            )
            self._backward_hooks.append(bwd_hook)
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._forward_hooks:
            hook.remove()
        for hook in self._backward_hooks:
            hook.remove()
        
        self._forward_hooks.clear()
        self._backward_hooks.clear()
        self._activations.clear()
    
    def _forward_hook(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
        layer_idx: int
    ) -> None:
        """
        Forward hook to store activations.
        
        Stores input activations a_{l-1} for computing ∇W_l = e_l a_{l-1}^T.
        """
        if not self.enabled:
            return
        
        # Get input tensor (handle tuple input)
        if isinstance(input, tuple):
            activation = input[0]
        else:
            activation = input
        
        # Store detached copy to save memory (don't need gradients)
        # Keep on same device as input
        self._activations[layer_idx] = activation.detach()
    
    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[Optional[torch.Tensor], ...],
        grad_output: Tuple[torch.Tensor, ...],
        layer_idx: int
    ) -> Optional[Tuple[Optional[torch.Tensor], ...]]:
        """
        Backward hook to replace gradients with DFA updates.
        
        Computes:
        1. Local error: e_l = B_l δ_L
        2. DFA gradient: ∇W_l = e_l a_{l-1}^T
        
        Returns:
            Modified grad_input tuple with DFA gradients
        """
        if not self.enabled or self._global_error is None:
            return None  # Don't modify gradients
        
        # Get stored activation
        if layer_idx not in self._activations:
            return None  # No activation stored, skip
        
        activation = self._activations[layer_idx]  # [batch, d_{l-1}]
        
        # Compute local error: e_l = B_l δ_L
        # Ensure global_error is on same device as activation
        global_error_device = self._global_error.to(activation.device)
        local_error = self.feedback_matrix.compute_local_error(
            layer_idx,
            global_error_device
        )  # [batch, d_l]
        
        # Compute DFA gradient: ∇W_l = (1/batch_size) * e_l^T @ a_{l-1}
        # Standard: grad_W = grad_output^T @ input
        # DFA: grad_W = local_error^T @ activation
        
        batch_size = activation.shape[0]
        
        # Handle 2D (batch, features) or 3D (batch, seq, features) tensors
        if activation.dim() == 3:
            # For attention layers with sequence dimension
            # Reshape: [batch, seq, d] -> [batch*seq, d]
            batch_seq, seq_len, d_in = activation.shape
            activation_flat = activation.reshape(-1, d_in)
            
            # local_error should also be [batch*seq, d_out]
            if local_error.dim() == 2 and local_error.shape[0] == batch_seq:
                # Expand to sequence length
                local_error = local_error.unsqueeze(1).expand(-1, seq_len, -1)
            local_error_flat = local_error.reshape(-1, local_error.shape[-1])
            
            # Compute gradient: [d_out, d_in]
            dfa_grad = torch.matmul(
                local_error_flat.T, 
                activation_flat
            ) / (batch_seq * seq_len)
        else:
            # Standard 2D case: [batch, d]
            dfa_grad = torch.matmul(
                local_error.T,
                activation
            ) / batch_size
        
        # Store statistics if requested
        if self.store_statistics:
            self._grad_stats[f"layer_{layer_idx}_grad_norm"] = dfa_grad.norm().item()
            self._grad_stats[f"layer_{layer_idx}_local_error_norm"] = local_error.norm().item()
        
        # Replace gradient in module's weight parameter
        if hasattr(module, 'weight') and module.weight is not None:
            if module.weight.grad is None:
                module.weight.grad = dfa_grad
            else:
                # Replace existing gradient
                module.weight.grad.copy_(dfa_grad)
        
        # Return None to keep standard gradient flow for other parameters
        # (e.g., biases, layer norms)
        return None
    
    def set_global_error(self, global_error: torch.Tensor) -> None:
        """
        Set the global error δ_L from the loss.
        
        Should be called after computing loss, before backward pass.
        
        Args:
            global_error: Global error δ_L [batch, d_final] or [batch, seq, d_final]
        """
        self._global_error = global_error
    
    def clear_global_error(self) -> None:
        """Clear stored global error (e.g., after backward pass)."""
        self._global_error = None
        self._activations.clear()
    
    def get_statistics(self) -> Dict[str, float]:
        """Get gradient statistics for diagnostics."""
        return self._grad_stats.copy()
    
    def enable(self) -> None:
        """Enable DFA gradient replacement."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable DFA gradient replacement (use standard BP)."""
        self.enabled = False
    
    def __repr__(self) -> str:
        num_hooks = len(self._forward_hooks)
        status = "enabled" if self.enabled else "disabled"
        return f"DFABackwardHook(num_layers={num_hooks}, status={status})"


class HybridDFAHook(DFABackwardHook):
    """
    Hybrid DFA/BP backward hook.
    
    Applies DFA between transformer blocks but uses standard BP within blocks.
    This is the recommended strategy from expert review to balance gradient
    quality with computational efficiency.
    
    Args:
        feedback_matrix: DFAFeedbackMatrix for inter-block connections
        block_boundaries: List of layer indices that are block boundaries
        enabled: Whether DFA is active
    """
    
    def __init__(
        self,
        feedback_matrix: DFAFeedbackMatrix,
        block_boundaries: List[int],
        enabled: bool = True,
        store_statistics: bool = True
    ):
        super().__init__(feedback_matrix, enabled, store_statistics)
        self.block_boundaries = set(block_boundaries)
    
    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[Optional[torch.Tensor], ...],
        grad_output: Tuple[torch.Tensor, ...],
        layer_idx: int
    ) -> Optional[Tuple[Optional[torch.Tensor], ...]]:
        """
        Backward hook that only applies DFA at block boundaries.
        
        For layers within blocks, uses standard BP.
        For layers at block boundaries, uses DFA.
        """
        # Only apply DFA at block boundaries
        if layer_idx not in self.block_boundaries:
            return None  # Use standard BP
        
        # Apply DFA for this layer
        return super()._backward_hook(module, grad_input, grad_output, layer_idx)


# Testing code
if __name__ == "__main__":
    print("Testing DFABackwardHook...")
    
    # Test 1: Basic hook registration
    print("\n=== Test 1: Hook Registration ===")
    
    # Create simple network
    layer_dims = [64, 128, 128, 64]
    output_dim = 256
    
    # Create layers where each outputs the dimension for DFA
    layers = []
    layers.append(nn.Linear(64, layer_dims[0]))  # Input layer
    for i in range(1, len(layer_dims)):
        layers.append(nn.Linear(layer_dims[i-1], layer_dims[i]))
    
    feedback = DFAFeedbackMatrix(layer_dims, output_dim)
    hook = DFABackwardHook(feedback)
    
    print(f"Created {len(layers)} layers")
    print(f"Feedback matrix: {len(feedback)} projections")
    
    # Register hooks
    hook.register_hooks(layers, list(range(len(layers))))
    print(f"Registered hooks: {len(hook._forward_hooks)} forward, {len(hook._backward_hooks)} backward")
    
    # Test 2: Forward pass with activation storage
    print("\n=== Test 2: Forward Pass + Activation Storage ===")
    
    batch_size = 4
    x = torch.randn(batch_size, 64)  # Match first layer input
    
    # Forward pass through layers
    for i, layer in enumerate(layers):
        x = layer(x)
    
    # Add final projection to output_dim for loss computation
    final_proj = nn.Linear(layer_dims[-1], output_dim)
    output = final_proj(x)
    
    # Add final projection to output_dim for loss computation
    final_proj = nn.Linear(layer_dims[-1], output_dim)
    output = final_proj(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Stored activations: {len(hook._activations)}")
    for idx, act in hook._activations.items():
        print(f"  Layer {idx}: {act.shape}")
    
    # Test 3: Backward pass with DFA
    print("\n=== Test 3: Backward Pass with DFA ===")
    
    # Create dummy loss
    target = torch.randn_like(output)
    loss = ((output - target) ** 2).mean()
    
    # Compute global error: δ_L = ∂L/∂output
    global_error = 2 * (output - target) / batch_size
    
    print(f"Loss: {loss.item():.6f}")
    print(f"Global error shape: {global_error.shape}")
    
    # Set global error for DFA
    hook.set_global_error(global_error.detach())
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print(f"Layer gradients:")
    for i, layer in enumerate(layers):
        if layer.weight.grad is not None:
            print(f"  Layer {i}: grad shape {layer.weight.grad.shape}, norm {layer.weight.grad.norm().item():.6f}")
    
    # Get statistics
    stats = hook.get_statistics()
    print(f"Statistics: {len(stats)} entries")
    for key, val in sorted(stats.items()):
        print(f"  {key}: {val:.6f}")
    
    # Test 4: Compare DFA vs BP
    print("\n=== Test 4: DFA vs Standard BP Comparison ===")
    
    # Create fresh network
    layers_dfa = [nn.Linear(64, 128), nn.Linear(128, 256)]
    layers_bp = [nn.Linear(64, 128), nn.Linear(128, 256)]
    
    # Copy weights
    layers_bp[0].weight.data.copy_(layers_dfa[0].weight.data)
    layers_bp[1].weight.data.copy_(layers_dfa[1].weight.data)
    
    feedback = DFAFeedbackMatrix([128, 256], 256)
    hook = DFABackwardHook(feedback)
    hook.register_hooks(layers_dfa, [0, 1])
    
    # Forward pass (same input for both)
    x = torch.randn(4, 64)
    
    x_dfa = x.clone()
    for layer in layers_dfa:
        x_dfa = layer(x_dfa)
    
    x_bp = x.clone()
    for layer in layers_bp:
        x_bp = layer(x_bp)
    
    # Same target
    target = torch.randn_like(x_dfa)
    
    # DFA backward
    loss_dfa = ((x_dfa - target) ** 2).mean()
    global_error = 2 * (x_dfa - target) / x.shape[0]
    hook.set_global_error(global_error.detach())
    loss_dfa.backward()
    
    # BP backward
    loss_bp = ((x_bp - target) ** 2).mean()
    loss_bp.backward()
    
    # Compare gradients
    print("Gradient comparison (Layer 0):")
    grad_dfa = layers_dfa[0].weight.grad
    grad_bp = layers_bp[0].weight.grad
    
    grad_diff = (grad_dfa - grad_bp).norm() / grad_bp.norm()
    grad_cosine = torch.nn.functional.cosine_similarity(
        grad_dfa.flatten(), 
        grad_bp.flatten(), 
        dim=0
    )
    
    print(f"  DFA grad norm: {grad_dfa.norm().item():.6f}")
    print(f"  BP grad norm: {grad_bp.norm().item():.6f}")
    print(f"  Relative difference: {grad_diff.item():.4f}")
    print(f"  Cosine similarity: {grad_cosine.item():.4f}")
    
    # Test 5: Hybrid DFA/BP
    print("\n=== Test 5: Hybrid DFA/BP Hook ===")
    
    layer_dims = [64, 128, 128, 64]
    layers = [nn.Linear(layer_dims[i], layer_dims[i]) for i in range(len(layer_dims))]
    
    feedback = DFAFeedbackMatrix(layer_dims, 256)
    
    # Only apply DFA at layers 0 and 2 (block boundaries)
    hybrid_hook = HybridDFAHook(feedback, block_boundaries=[0, 2])
    hybrid_hook.register_hooks(layers, list(range(len(layers))))
    
    print(f"Hybrid hook: DFA at block boundaries {hybrid_hook.block_boundaries}")
    print(f"Registered {len(hybrid_hook._forward_hooks)} hooks")
    
    # Cleanup
    hook.remove_hooks()
    hybrid_hook.remove_hooks()
    
    print("\n✅ All tests completed!")
