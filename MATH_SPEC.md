# OKADFA: Mathematical Specification

**Orthogonalized Kernel Attention with Direct Feedback Alignment**

---

## 1. Core Method Definition

### 1.1 Direct Feedback Alignment (DFA)

Consider a Transformer with layers $l = 1, \ldots, L$. Let the final pre-softmax logits be $y_{\text{final}} \in \mathbb{R}^{d_{\text{final}}}$ and the total loss:

$$L_{\text{total}} = L_{\text{CE}}(y_{\text{final}}, y_{\text{target}}) + L_{\text{ortho}}$$

where $L_{\text{ortho}}$ is the orthogonality penalty (defined in §1.3).

#### Global Error Signal

$$\delta_L = \frac{\partial L_{\text{total}}}{\partial y_{\text{final}}} \in \mathbb{R}^{d_{\text{final}}}$$

#### Fixed Random Feedback Matrices

For each layer $l$, define a **fixed random feedback matrix**:

$$B_l \in \mathbb{R}^{d_l \times d_{\text{final}}}, \quad B_l(i,j) \sim \mathcal{N}\left(0, \frac{1}{\sqrt{d_{\text{final}}}}\right)$$

These matrices are:
- Initialized once with a fixed random seed
- Never updated during training
- Used to compute local error signals

#### Local Error Computation

Local error at layer $l$:

$$e_l = B_l \, \delta_L \in \mathbb{R}^{d_l}$$

For a linear weight matrix $W_l \in \mathbb{R}^{d_{l-1} \times d_l}$, with:

$$z_l = h_{l-1} W_l, \quad h_l = \phi(z_l)$$

we compute:

$$\delta_l = e_l \odot \phi'(z_l)$$

$$\nabla_{W_l} L_{\text{DFA}} = h_{l-1}^\top \delta_l$$

This pattern applies to:
- Attention projections: $W_{Q,h,l}, W_{K,h,l}, W_{V,h,l}$
- FFN layers within each block
- Output projection matrices

#### Implementation Note: Activation Storage

DFA **does not** require the full backpropagation graph through all layers. However, we still need $h_{l-1}$ and $z_l$ (or sufficient information to reconstruct them) at update time. Two strategies:

1. **Store minimal activations**: Keep only $h_{l-1}$ and $z_l$ per layer (cheaper compute, moderate memory)
2. **Recompute with checkpointing**: Recompute forward pass locally per layer without gradient graph (more compute, less memory)

Our implementation uses strategy 1 with optional checkpointing for deeper models.

---

### 1.2 Kernelized Orthogonal Attention (KOA)

#### Notation

- Input sequence: $X \in \mathbb{R}^{T \times d_{\text{model}}}$
- For head $h$ in layer $l$, projection matrices: $W_{Q,h,l}, W_{K,h,l}, W_{V,h,l} \in \mathbb{R}^{d_{\text{model}} \times d_k}$

#### Projections

$$Q_h = X W_{Q,h,l}, \quad K_h = X W_{K,h,l}, \quad V_h = X W_{V,h,l}$$

#### Positive Orthogonal Random Feature Map

We introduce a **Favor+ feature map** $\phi: \mathbb{R}^{d_k} \to \mathbb{R}^{M}$ with $M \in [2d_k, 4d_k]$:

$$\phi(x) = \frac{1}{\sqrt{M}} \exp\left(\frac{x \Omega}{\sqrt{d_k}} - \frac{\|x\|^2}{2}\right)$$

where $\Omega \in \mathbb{R}^{d_k \times M}$ is constructed from orthogonal blocks via QR decomposition with random scaling.

#### Kernelized Attention

Define:

$$\tilde{Q}_h = \phi(Q_h) \in \mathbb{R}^{T \times M}, \quad \tilde{K}_h = \phi(K_h) \in \mathbb{R}^{T \times M}$$

Kernelized attention with normalization:

$$A_h = \text{Attn}_{\text{kernel}}(Q_h, K_h, V_h) = \frac{\tilde{Q}_h \left(\tilde{K}_h^\top V_h\right)}{\tilde{Q}_h \left(\tilde{K}_h^\top \mathbf{1}\right) + \epsilon}$$

where $\mathbf{1} \in \mathbb{R}^{T}$ is a vector of ones and $\epsilon = 10^{-6}$ for numerical stability.

After computing attention for all heads, concatenate and project:

$$\text{MultiHead}(X) = \text{Concat}(A_1, \ldots, A_H) W_O$$

#### Complexity Analysis

**Standard Softmax Attention**: $O(T^2 d_k)$ per layer

**Kernelized Attention**: $O(T M d_k)$ per layer

Since $M = O(d_k)$ and typically $M \ll T$, we achieve **linear complexity in sequence length $T$**.

---

### 1.3 Orthogonality Penalty

For each head $h$ and layer $l$, we enforce approximate orthogonality on $W_Q$ and $W_K$:

$$L_{\text{ortho}} = \lambda \sum_{l=1}^L \sum_{h=1}^{H} \left( \|W_{Q,h,l}^\top W_{Q,h,l} - I\|_F^2 + \|W_{K,h,l}^\top W_{K,h,l} - I\|_F^2 \right)$$

where:
- $\|\cdot\|_F$ denotes the Frobenius norm
- $I \in \mathbb{R}^{d_k \times d_k}$ is the identity matrix
- $\lambda$ is the orthogonality penalty weight

#### Lambda Warmup Schedule

$\lambda$ is linearly warmed up over the first $p\%$ of training steps (default $p = 10$):

$$\lambda(t) = \begin{cases}
\lambda_{\max} \cdot \frac{t}{t_{\text{warmup}}} & \text{if } t < t_{\text{warmup}} \\
\lambda_{\max} & \text{otherwise}
\end{cases}$$

where $\lambda_{\max} \in [10^{-5}, 10^{-3}]$ is tuned empirically.

#### Optional Extension

Include $W_V$ in the penalty (ablation study):

$$L_{\text{ortho}}^{\text{+V}} = L_{\text{ortho}} + \lambda \sum_{l=1}^L \sum_{h=1}^{H} \|W_{V,h,l}^\top W_{V,h,l} - I\|_F^2$$

---

### 1.4 Training Loop (OKADFA)

#### Algorithm: OKADFA Training Step

**Input**: Batch $X$, targets $Y$, model parameters $\Theta$, feedback matrices $\{B_l\}_{l=1}^L$

**Output**: Updated parameters $\Theta'$

1. **Forward Pass**
   - Run Transformer with KOA replacing softmax MHSA
   - Store minimal activations: $h_{l-1}, z_l$ for each layer $l$
   - Compute $L_{\text{CE}}$ from final logits
   - Compute $L_{\text{ortho}}$ from Q, K projection matrices
   - Form $L_{\text{total}} = L_{\text{CE}} + L_{\text{ortho}}$

2. **Global Error**
   - Compute $\delta_L = \frac{\partial L_{\text{total}}}{\partial y_{\text{final}}}$

3. **Local DFA Updates** (for each layer $l = L, \ldots, 1$)
   - Compute local error: $e_l = B_l \delta_L$
   - For each parameterized submodule in layer $l$:
     - Retrieve stored activations $h_{l-1}, z_l$
     - Compute $\delta_l = e_l \odot \phi'(z_l)$
     - Compute gradient: $\nabla_{W_l} = h_{l-1}^\top \delta_l$
   - Accumulate gradients for AdamW optimizer

4. **LayerNorm Handling**
   - LayerNorm parameters $(\gamma, \beta)$ updated with **standard BP** for stability
   - This maintains a small BP graph within each block but avoids deep chains

5. **Parameter Update**
   - Apply AdamW with accumulated DFA gradients
   - Apply gradient clipping if specified

---

## 2. Intuition

**Why OKADFA?**

DFA decouples error propagation across layers, eliminating the need for deep backpropagation chains and reducing memory requirements. KOA with orthogonality constraints produces better-conditioned, linearly-complex attention pathways. Together, they enable each layer to operate as a relatively independent, well-conditioned module trained by a shared global error signal, while maintaining linear complexity in sequence length.

**Key Benefits:**
- **Memory**: No deep computation graph, only local activations per layer
- **Compute**: Linear complexity in sequence length for attention
- **Conditioning**: Orthogonal constraints improve stability and generalization
- **Parallelism**: Layers can potentially be updated more independently

---

## 3. Implementation Strategy

### 3.1 Hybrid DFA/BP (Recommended v1)

Pure DFA from input to output for a deep stack is risky. We recommend:

**Strategy A: Intra-block BP, Inter-block DFA**
- Use standard BP **within each Transformer block** (especially for LayerNorm + FFN)
- Use DFA **between blocks**
- Drastically reduces graph depth while maintaining stability

**Strategy B: Partial DFA**
- Use DFA for bottom $L - k$ layers
- Use full BP for top $k$ layers (e.g., last 2-4 blocks)
- Allows model to learn complex patterns in later layers with full gradients

### 3.2 Activation Management

We do not maintain a full backpropagation graph. Instead:
- **(a) Minimal storage**: Store only $h_{l-1}$ and $z_l$ per layer
- **(b) Activation checkpointing**: Recompute forward pass locally for DFA updates, trading compute for memory

For Phase 1, we use strategy (a) for simplicity.

### 3.3 LayerNorm Integration

LayerNorm's Jacobian couples features, making naive DFA application problematic. Our approach:

1. Apply DFA at the level of **block outputs**, not inside LayerNorm
2. Use standard BP for LayerNorm weights $(\gamma, \beta)$
3. Update major weight matrices (Q/K/V/FFN) with DFA
4. This creates a hybrid system: small local BP graphs + DFA between blocks

---

## 4. Diagnostic Metrics

### 4.1 Gradient Quality (DFA vs BP)

On a small diagnostic sub-network, compute:

$$\text{GradError}_l = \frac{\|\nabla_{\text{DFA}} W_l - \nabla_{\text{BP}} W_l\|_F}{\|\nabla_{\text{BP}} W_l\|_F}$$

**Acceptance criterion**: $\text{GradError}_l < 0.1$ consistently after warm-up (first 10% of steps)

### 4.2 Approximation Quality (Kernel vs Softmax)

For sampled heads and positions, compute:

$$\text{AttnSim} = \frac{\langle \text{vec}(\text{Attn}_{\text{softmax}}), \text{vec}(\text{Attn}_{\text{kernel}}) \rangle}{\|\text{vec}(\text{Attn}_{\text{softmax}})\| \cdot \|\text{vec}(\text{Attn}_{\text{kernel}})\|}$$

Alternatively, Frobenius norm:

$$\text{AttnError} = \|Q K^\top - \phi(Q) \phi(K)^\top\|_F$$

**Target**: Cosine similarity $> 0.95$ for high-quality approximation

### 4.3 Orthogonality Violation

Track over training:

$$\text{OrthoViolation}_{h,l} = \|W_{Q,h,l}^\top W_{Q,h,l} - I\|_F^2 + \|W_{K,h,l}^\top W_{K,h,l} - I\|_F^2$$

**Target**: $\text{OrthoViolation} < 0.1$ per head after warmup

---

## 5. Experimental Ablations

### 5.1 Attention Variants
1. **Baseline**: Softmax + standard BP
2. **KOA only**: Favor+ kernel + standard BP
3. **KOA + Ortho**: Favor+ + BP + orthogonality penalty
4. **KOA + DFA**: Favor+ + DFA (no orthogonality)
5. **OKADFA (full)**: Favor+ + DFA + orthogonality

### 5.2 DFA Scope
1. DFA on all layers
2. DFA on bottom $L - k$ layers only
3. Hybrid: DFA between blocks, BP within blocks (recommended)

### 5.3 Orthogonality Hyperparameters
- $\lambda \in \{0, 10^{-5}, 3 \times 10^{-5}, 10^{-4}, 3 \times 10^{-4}, 10^{-3}\}$
- With/without warmup schedule
- Include/exclude $W_V$ in penalty

### 5.4 Feedback Matrix Variants (Phase 2)
1. Dense, fixed Gaussian (baseline)
2. Sparse $B_l$ (post-initialization pruning)
3. Different initialization scales
4. Block-diagonal structure

### 5.5 Metrics

**Efficiency:**
- Peak VRAM usage (GB)
- Wall-clock time per token (ms)
- Approximate FLOPs per token

**Quality:**
- Training perplexity vs. step
- Validation perplexity
- Final evaluation perplexity

**Diagnostics:**
- $\text{GradError}_l$ (DFA vs BP)
- $\text{AttnSim}$ (kernel vs softmax)
- $\text{OrthoViolation}$ over time

---

## 6. Implementation Checklist

### Phase 1: Core Components
- [ ] `DFAFeedbackMatrix`: Initialize and manage $B_l$ matrices
- [ ] `DFABackwardHook`: PyTorch autograd hook for local gradients
- [ ] `OrthogonalityLoss`: Compute $L_{\text{ortho}}$ efficiently
- [ ] `OrthogonalityScheduler`: Lambda warmup schedule
- [ ] Gradient comparison diagnostics

### Phase 2: Model Architecture
- [ ] `KOAMultiHeadAttention`: Multi-head wrapper with orthogonality
- [ ] `DFATransformerBlock`: Transformer block with DFA hooks
- [ ] `OKADFAModel`: Full model with config-driven instantiation
- [ ] Activation storage/checkpointing utilities

### Phase 3: Training Infrastructure
- [ ] Data loading (OpenWebText/C4)
- [ ] Training loop with diagnostics
- [ ] Checkpoint management
- [ ] W&B/TensorBoard logging
- [ ] Evaluation scripts

---

## 7. Success Criteria

### Component-Level
- [ ] Favor+ achieves $\text{AttnSim} > 0.95$ with $M = 4 \times d_k$
- [ ] DFA gradients: $\text{GradError} < 0.1$ after warmup
- [ ] Orthogonality: $\text{OrthoViolation} < 0.1$ after warmup

### Model-Level
- [ ] 2-layer toy model converges on WikiText-2
- [ ] Training loss decreases consistently over 10K steps
- [ ] Memory usage $< 80\%$ of standard Transformer
- [ ] Throughput $\geq$ baseline for sequences $> 512$

### Full System
- [ ] 6-layer model trains stably for 50K steps
- [ ] Perplexity within 5% of standard Transformer
- [ ] Wall-clock speedup for long sequences ($T > 1024$)
- [ ] GPU memory reduction of $\geq 30\%$

---

## References

1. **Performer (Favor+)**: Choromanski et al., "Rethinking Attention with Performers" (ICLR 2021)
2. **Direct Feedback Alignment**: Nøkland, "Direct Feedback Alignment Provides Learning in Deep Neural Networks" (NeurIPS 2016)
3. **Orthogonal Constraints**: Bansal et al., "Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?" (NeurIPS 2018)

---

*Mathematical Specification v1.0 - December 11, 2025*
