# OKADFA: Orthogonalized Kernel Attention with Direct Feedback Alignment

Efficient LLM training through combined Direct Feedback Alignment and Kernelized Orthogonal Attention.

**Copyright (c) 2025 Gregory Ward - SmartLedger.Technology**

## Project Structure

```
ortho-ai-research/
├── src/
│   ├── models/          # Model architectures
│   ├── training/        # Training logic (DFA, orthogonality loss)
│   ├── kernels/         # Kernel approximations (Favor+)
│   └── diagnostics/     # Gradient comparison & monitoring
├── configs/             # Experiment configurations
├── tests/               # Unit tests
├── scripts/             # Training & evaluation scripts
└── requirements.txt     # Dependencies
```

## Setup

### 1. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -e .
# Or for development:
pip install -e ".[dev]"
```

### 3. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

## Quick Start

```python
# Coming soon: Training scripts
```

## Research Protocol

See `orthogonalized_kernel_attention_with_direct_feedback_alignment_for.md` for the full research proposal.

### Key Innovations

1. **Direct Feedback Alignment (DFA)**: Decoupled gradient computation with fixed random feedback matrices
2. **Kernelized Orthogonal Attention (KOA)**: Linear-complexity attention with orthogonality constraints

### Experimental Phases

- **Phase 1**: Validate DFA on small transformer (2-4 layers)
- **Phase 2**: Validate KOA independently
- **Phase 3**: Combine into OKADFA
- **Phase 4**: Scale to 125M parameters

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ tests/
isort src/ tests/
```

## License

MIT License - See [LICENSE](LICENSE) file for details.

**Copyright (c) 2025 Gregory Ward - SmartLedger.Technology**
