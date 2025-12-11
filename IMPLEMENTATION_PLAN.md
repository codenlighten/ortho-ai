# OKADFA Implementation Plan (Revised)

**Incorporating Expert Review Feedback**

---

## Key Insights from Review

### 1. **Hybrid DFA/BP is Critical for v1**
Pure DFA is risky. We will implement:
- **Strategy**: DFA between blocks, standard BP within blocks
- **Rationale**: Maintains stability while reducing graph depth
- **LayerNorm**: Always use standard BP for $(\gamma, \beta)$ parameters

### 2. **Activation Management Strategy**
We need activations for DFA but not full BP graph:
- **Store**: Minimal $(h_{l-1}, z_l)$ per layer
- **Recompute**: Not needed for v1, use storage
- **Memory savings**: No deep autograd chain

### 3. **Diagnostic-First Approach**
Build diagnostics **before** full training:
- Gradient comparison (DFA vs BP) on small network
- Attention approximation quality tracking
- Orthogonality violation monitoring

---

## Phase 1: Core Components (Week 1)

### 1.1 DFA Feedback Matrix (Priority 1)

**File**: `src/training/dfa_feedback.py`

```python
class DFAFeedbackMatrix:
    """
    Manages fixed random feedback matrices B_l for DFA.
    
    B_l ∈ R^{d_l × d_final}
    B_l(i,j) ~ N(0, 1/√d_final)
    """
    
    def __init__(self, layer_dims: List[int], output_dim: int, seed: int = 42):
        """
        Args:
            layer_dims: List of dimensions [d_1, d_2, ..., d_L]
            output_dim: Final output dimension d_final
            seed: Fixed random seed for reproducibility
        """
        
    def get_feedback_matrix(self, layer_idx: int) -> torch.Tensor:
        """Get B_l for layer l"""
        
    def compute_local_error(
        self, 
        layer_idx: int, 
        global_error: torch.Tensor
    ) -> torch.Tensor:
        """Compute e_l = B_l δ_L"""
```

**Tests**:
- [ ] Matrix shapes correct for all layers
- [ ] Initialization scale: std ≈ 1/√d_final
- [ ] Reproducibility with fixed seed
- [ ] GPU memory usage reasonable

---

### 1.2 DFA Backward Hook (Priority 1)

**File**: `src/training/dfa_backward.py`

```python
class DFABackwardHook:
    """
    PyTorch autograd hook for DFA backward pass.
    
    Replaces standard backprop with local DFA updates:
    δ_l = e_l ⊙ φ'(z_l)
    ∇W_l = h_{l-1}^T δ_l
    """
    
    def __init__(self, feedback_matrices: DFAFeedbackMatrix):
        self.feedback_matrices = feedback_matrices
        self.stored_activations = {}  # Store (h_{l-1}, z_l)
        
    def register_hooks(self, model: nn.Module):
        """Register forward and backward hooks on model layers"""
        
    def forward_hook(
        self, 
        module: nn.Module, 
        input: torch.Tensor, 
        output: torch.Tensor
    ):
        """Store activations during forward pass"""
        
    def backward_hook(
        self,
        module: nn.Module,
        grad_input: torch.Tensor,
        grad_output: torch.Tensor
    ) -> torch.Tensor:
        """Replace gradient with DFA local gradient"""
```

**Key Decisions**:
1. **Hook placement**: At block boundaries (not inside blocks)
2. **Activation storage**: Dict keyed by layer name
3. **Gradient replacement**: Use `register_backward_hook`

**Tests**:
- [ ] Activations stored correctly
- [ ] Local gradients computed correctly
- [ ] Gradients flow to optimizer
- [ ] No memory leaks over multiple steps

---

### 1.3 Orthogonality Loss (Priority 1)

**File**: `src/training/orthogonality_loss.py`

```python
class OrthogonalityLoss(nn.Module):
    """
    Computes orthogonality penalty:
    L_ortho = λ Σ (||W_Q^T W_Q - I||_F^2 + ||W_K^T W_K - I||_F^2)
    """
    
    def __init__(
        self,
        lambda_init: float = 1e-4,
        warmup_steps: int = 1000,
        apply_to_v: bool = False
    ):
        """
        Args:
            lambda_init: Initial λ value
            warmup_steps: Steps for linear warmup
            apply_to_v: Whether to include W_V in penalty
        """
        
    def forward(
        self,
        attention_layers: List[nn.Module],
        step: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss and diagnostics.
        
        Returns:
            loss: Orthogonality penalty
            diagnostics: Dict with per-layer violations
        """
        
    def get_lambda(self, step: int) -> float:
        """Get current λ with warmup"""
```

**Implementation Details**:
```python
def compute_orthogonality_penalty(W: torch.Tensor) -> torch.Tensor:
    """
    Compute ||W^T W - I||_F^2 efficiently.
    
    Args:
        W: (d_model, d_k) projection matrix
    Returns:
        penalty: Scalar tensor
    """
    gram = W.T @ W  # (d_k, d_k)
    identity = torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
    diff = gram - identity
    penalty = torch.sum(diff ** 2)  # Frobenius norm squared
    return penalty
```

**Tests**:
- [ ] Loss = 0 for orthogonal matrices
- [ ] Loss > 0 for non-orthogonal matrices
- [ ] Warmup schedule correct
- [ ] Batched computation efficient

---

### 1.4 Gradient Diagnostics (Priority 2)

**File**: `src/diagnostics/gradient_comparison.py`

```python
class GradientComparator:
    """
    Compare DFA vs BP gradients for validation.
    
    Metrics:
    - ||∇_DFA - ∇_BP||_F / ||∇_BP||_F
    - Cosine similarity
    - Component-wise correlation
    """
    
    def __init__(self, sample_frequency: int = 100):
        self.sample_frequency = sample_frequency
        self.history = []
        
    def compare_gradients(
        self,
        model_dfa: nn.Module,
        model_bp: nn.Module,
        batch: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Run forward+backward on both models, compare gradients.
        
        Returns dict with metrics per layer.
        """
        
    def log_comparison(
        self,
        step: int,
        comparisons: Dict[str, float]
    ):
        """Log to W&B/TensorBoard"""
```

**Usage**:
```python
# In training loop
if step % comparator.sample_frequency == 0:
    # Create small BP reference model
    model_bp = create_small_reference_model()
    metrics = comparator.compare_gradients(
        model_dfa=okadfa_model,
        model_bp=model_bp,
        batch=diagnostic_batch,
        targets=diagnostic_targets
    )
    comparator.log_comparison(step, metrics)
```

**Tests**:
- [ ] Identical models → zero error
- [ ] Random models → high error
- [ ] DFA gradients within expected range

---

## Phase 2: Model Architecture (Week 2)

### 2.1 KOA Multi-Head Attention

**File**: `src/models/koa_attention.py`

```python
class KOAMultiHeadAttention(nn.Module):
    """
    Multi-head attention with Favor+ kernels and orthogonality.
    
    Architecture:
    - Q, K, V projections (with orthogonality on Q, K)
    - Favor+ attention per head
    - Output projection
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_features: Optional[int] = None,
        dropout: float = 0.1,
        causal: bool = True
    ):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Projections (per-head for orthogonality tracking)
        self.q_projs = nn.ModuleList([
            nn.Linear(d_model, self.d_k, bias=False)
            for _ in range(num_heads)
        ])
        self.k_projs = nn.ModuleList([
            nn.Linear(d_model, self.d_k, bias=False)
            for _ in range(num_heads)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(d_model, self.d_k, bias=False)
            for _ in range(num_heads)
        ])
        
        # Favor+ attention
        self.attention = FavorPlusAttention(
            d_model=self.d_k,
            num_features=num_features,
            causal=causal
        )
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
    def get_projection_matrices(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get (W_Q, W_K) pairs for orthogonality loss"""
```

**Tests**:
- [ ] Multi-head output shape correct
- [ ] Projection matrices accessible for loss
- [ ] Forward pass with Favor+ works
- [ ] Gradients flow correctly

---

### 2.2 DFA Transformer Block (Hybrid BP/DFA)

**File**: `src/models/dfa_transformer_block.py`

```python
class DFATransformerBlock(nn.Module):
    """
    Transformer block with hybrid DFA/BP.
    
    Strategy:
    - Standard BP WITHIN block (LayerNorm, residuals)
    - DFA BETWEEN blocks
    - This maintains stability while reducing graph depth
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_dfa: bool = True
    ):
        # Pre-LayerNorm architecture for stability
        self.ln1 = nn.LayerNorm(d_model)  # Uses standard BP
        self.attention = KOAMultiHeadAttention(...)
        
        self.ln2 = nn.LayerNorm(d_model)  # Uses standard BP
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.use_dfa = use_dfa
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pre-norm Transformer block with residual connections.
        
        If use_dfa=True, DFA hook is attached at block boundary.
        """
        # Attention sub-block (with residual)
        residual = x
        x = self.ln1(x)  # BP for LayerNorm
        x = self.attention(x)
        x = residual + x  # Residual (BP)
        
        # FFN sub-block (with residual)
        residual = x
        x = self.ln2(x)  # BP for LayerNorm
        x = self.ffn(x)
        x = residual + x  # Residual (BP)
        
        # DFA hook attached HERE (between blocks)
        return x
```

**Tests**:
- [ ] Forward pass matches standard transformer
- [ ] BP within block works
- [ ] DFA hook placement correct
- [ ] Memory usage tracked

---

### 2.3 Full OKADFA Model

**File**: `src/models/okadfa_model.py`

```python
class OKADFAModel(nn.Module):
    """
    Complete OKADFA language model.
    
    Components:
    - Token + position embeddings
    - Stack of DFA transformer blocks
    - LM head
    - DFA hooks between blocks
    - Orthogonality loss computation
    """
    
    def __init__(self, config: OmegaConf):
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        
        self.blocks = nn.ModuleList([
            DFATransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                use_dfa=config.dfa.enabled
            )
            for _ in range(config.num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # DFA components
        if config.dfa.enabled:
            self.dfa_feedback = DFAFeedbackMatrix(...)
            self.dfa_hook = DFABackwardHook(self.dfa_feedback)
            self.dfa_hook.register_hooks(self)
        
        # Orthogonality loss
        self.ortho_loss = OrthogonalityLoss(
            lambda_init=config.koa.orthogonality_weight,
            warmup_steps=config.koa.orthogonality_warmup_steps
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        return_ortho_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional orthogonality loss.
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            ortho_loss: Optional orthogonality penalty
        """
```

**Tests**:
- [ ] End-to-end forward pass
- [ ] Loss computation correct
- [ ] DFA gradients computed
- [ ] Orthogonality penalty computed

---

## Phase 3: Training Infrastructure (Week 3)

### 3.1 Training Script

**File**: `scripts/train.py`

Key features:
- [ ] Data loading (OpenWebText/C4)
- [ ] Training loop with both losses
- [ ] Gradient accumulation
- [ ] Gradient clipping
- [ ] Learning rate scheduling
- [ ] Checkpoint saving/loading
- [ ] W&B logging with all diagnostics

### 3.2 Evaluation Script

**File**: `scripts/evaluate.py`

Key metrics:
- [ ] Perplexity on validation set
- [ ] Memory profiling
- [ ] Throughput (tokens/sec)
- [ ] Attention approximation quality
- [ ] Gradient statistics

---

## Acceptance Criteria (Before Scaling)

### Component-Level
- [ ] **Favor+**: Attention similarity > 0.95 (with M=4×d_k, small inputs)
- [ ] **DFA**: Gradient error < 0.1× ||∇_BP||_F after warmup
- [ ] **Orthogonality**: Violation < 0.1 per head after warmup

### Model-Level (2-layer, WikiText-2)
- [ ] Training loss decreases consistently
- [ ] Validation perplexity < 50
- [ ] No gradient explosions/vanishing
- [ ] Memory usage < baseline

### Diagnostic Checks
- [ ] Gradient comparison logged every 100 steps
- [ ] Attention fidelity logged every 100 steps  
- [ ] Orthogonality violation logged every step
- [ ] No NaN/Inf in any metric

---

## Implementation Order (Next 3 Days)

### Day 1: DFA Core
1. ✅ Favor+ (DONE)
2. ⏭️ `DFAFeedbackMatrix` class
3. ⏭️ Basic tests for feedback matrices
4. ⏭️ `OrthogonalityLoss` class
5. ⏭️ Tests for orthogonality loss

### Day 2: DFA Integration
1. ⏭️ `DFABackwardHook` class
2. ⏭️ Hook registration mechanism
3. ⏭️ Activation storage
4. ⏭️ Gradient replacement logic
5. ⏭️ End-to-end test: single linear layer with DFA

### Day 3: Model Architecture
1. ⏭️ `KOAMultiHeadAttention`
2. ⏭️ `DFATransformerBlock`
3. ⏭️ 2-layer toy model
4. ⏭️ Forward pass test
5. ⏭️ Backward pass test with DFA

---

## Ablation Study Matrix

| ID | Attention | DFA | Ortho | Notes |
|----|-----------|-----|-------|-------|
| A0 | Softmax | BP | No | Baseline |
| A1 | Favor+ | BP | No | Kernel only |
| A2 | Favor+ | BP | Yes | Kernel + ortho |
| A3 | Favor+ | DFA | No | Kernel + DFA |
| A4 | Favor+ | DFA | Yes | **Full OKADFA** |
| A5 | Favor+ | Hybrid | Yes | Hybrid DFA (recommended v1) |

---

## Risk Mitigation

### Risk 1: DFA Instability
**Mitigation**: Start with hybrid DFA/BP, only DFA between blocks

### Risk 2: Poor Approximation
**Mitigation**: Use M=4×d_k, monitor attention fidelity continuously

### Risk 3: Orthogonality Conflicts
**Mitigation**: Run ablation A3 (no ortho) first, tune λ carefully

### Risk 4: Implementation Bugs
**Mitigation**: Extensive unit tests, gradient checks, diagnostic tools

---

*Updated: December 11, 2025 - Post Expert Review*
