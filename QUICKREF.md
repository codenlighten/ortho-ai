# OKADFA Quick Reference

## Virtual Environment

### Activate
```bash
source .venv/bin/activate
```

### Deactivate
```bash
deactivate
```

### Run Python with venv (without activating)
```bash
.venv/bin/python script.py
```

## Development Commands

### Using the helper script
```bash
./dev.sh verify    # Verify installation
./dev.sh test      # Run tests
./dev.sh format    # Format code
./dev.sh lint      # Lint code
./dev.sh shell     # Python REPL
```

### Manual commands
```bash
# Install/update dependencies
.venv/bin/pip install -e .
.venv/bin/pip install -e ".[dev]"

# Run tests
.venv/bin/pytest tests/ -v
.venv/bin/pytest tests/ -v --cov=src

# Format code
.venv/bin/black src/ tests/
.venv/bin/isort src/ tests/

# Lint code
.venv/bin/flake8 src/ tests/

# Type checking (if using mypy)
.venv/bin/mypy src/
```

## Project Structure

```
src/
├── models/          # Model architectures
│   ├── koa_attention.py      # Kernelized Orthogonal Attention
│   ├── dfa_transformer.py    # DFA-enabled Transformer
│   └── okadfa_model.py       # Combined OKADFA model
├── training/        # Training components
│   ├── dfa_backward.py       # DFA backward pass
│   └── orthogonality_loss.py # Orthogonality constraint
├── kernels/         # Kernel approximations
│   └── favor_plus.py         # Favor+ random features
└── diagnostics/     # Monitoring tools
    └── gradient_comparison.py
```

## Key Concepts

### Direct Feedback Alignment (DFA)
```python
# Instead of BP: δ_l = δ_{l+1} @ W_{l+1}^T
# DFA uses: e_l = B_l^T @ δ_L
# Where B_l is a fixed random feedback matrix
```

### Kernelized Orthogonal Attention (KOA)
```python
# Standard attention: softmax(Q @ K^T) @ V
# KOA: φ(Q) @ φ(K)^T @ V  (linear complexity)
# With constraint: ||W_Q^T W_Q - I||_F^2 + ||W_K^T W_K - I||_F^2
```

### Loss Function
```python
L_total = L_cross_entropy + λ * L_ortho
```

## Configuration

Edit `configs/default.yaml` to change:
- Model architecture (layers, hidden size, heads)
- DFA settings (feedback matrix properties)
- KOA settings (kernel type, num features, λ)
- Training hyperparameters (lr, batch size, etc.)
- Diagnostic settings

## Common Tasks

### Create a new model component
1. Create file in appropriate `src/` subdirectory
2. Add unit tests in `tests/`
3. Import in `src/models/__init__.py` or similar
4. Update documentation

### Run experiments
```bash
# Coming soon
.venv/bin/python scripts/train.py --config configs/default.yaml
```

### Debug with small model
```python
from omegaconf import OmegaConf
config = OmegaConf.load("configs/default.yaml")
config.model.num_layers = 2  # Tiny model
config.model.hidden_size = 128
```

### Check gradients
```python
from src.diagnostics import compare_gradients
# Compare DFA vs BP on small batch
```

## Troubleshooting

### ImportError
```bash
# Reinstall in editable mode
.venv/bin/pip install -e .
```

### CUDA out of memory
- Reduce `batch_size` in config
- Reduce `model.hidden_size` or `model.num_layers`
- Use gradient checkpointing (to be implemented)

### Gradient explosion/vanishing
- Check `orthogonality_weight` (λ)
- Verify layer norm is working
- Check DFA feedback matrix scale
- Reduce learning rate

## Resources

### Papers
- **Performer**: https://arxiv.org/abs/2009.14794
- **DFA**: https://arxiv.org/abs/1609.01596
- **Orthogonal Transformers**: https://arxiv.org/abs/1806.01278

### Docs
- PyTorch: https://pytorch.org/docs
- Transformers: https://huggingface.co/docs/transformers
- W&B: https://docs.wandb.ai

---
*Keep this file updated as the project evolves!*
