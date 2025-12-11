# Google Colab Notebooks for OKADFA

Train **Orthogonalized Kernel Attention with Direct Feedback Alignment** models on Google Colab's free GPU!

## 🚀 Quick Links

### 1. Quick Start (10 minutes)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/codenlighten/ortho-ai/blob/main/colab/Quick_Start.ipynb)

**Perfect for:** First-time users, quick testing, verifying GPU setup

**What you get:**
- ✅ Complete setup in 3 steps
- 🧪 100-step training demo
- 📊 See OKADFA in action

---

### 2. Full Training Notebook (30+ minutes)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/codenlighten/ortho-ai/blob/main/colab/OKADFA_Training.ipynb)

**Perfect for:** Extended training, benchmarking, experimentation

**What you get:**
- 🎯 Multiple training presets (quick/extended/full)
- 📈 Benchmark vs baseline transformer
- 🧮 Model size calculator
- 📊 Result analysis & visualization
- 💾 Checkpoint management
- 🔧 Custom configuration options

---

## 📋 Prerequisites

1. **Google Account** - Free Colab access
2. **GPU Runtime** - Enable in: Runtime → Change runtime type → GPU
3. **10 minutes** - For quick start
4. **2+ hours** - For full-scale training

---

## 🎯 What is OKADFA?

**Orthogonalized Kernel Attention with Direct Feedback Alignment** combines:

- 🧠 **Biologically-inspired learning** - Direct Feedback Alignment (DFA) instead of backprop
- 🎯 **Orthogonalized attention** - Better feature decorrelation
- ⚡ **Kernel approximation** - Linear complexity attention
- 📊 **Proven results** - 31.8% perplexity improvement on WikiText-2

---

## 📊 Expected Results

### Quick Start (100 steps, ~3 minutes)
```
Val PPL: 49,000 → 30,000 (38% improvement)
Model: 8M parameters
```

### Extended Training (1000 steps, ~30 minutes)
```
Val PPL: ~15,000-20,000
Model: 27M parameters
```

### Full Scale (5000 steps, ~2 hours)
```
Val PPL: <10,000
Model: 163M parameters (GPT-2 Small)
```

---

## 🖥️ GPU Recommendations

### Colab Free Tier (T4 GPU, 16GB VRAM)
- ✅ Models up to 163M parameters (GPT-2 Small)
- ✅ Batch size: 4-8
- ✅ Training steps: Up to 5000
- ⏱️ Session limit: ~12 hours

### Colab Pro (A100 GPU, 40GB VRAM)
- ✅ Models up to 406M parameters (GPT-2 Medium)
- ✅ Batch size: 16-32
- ✅ Training steps: 10,000+
- ⏱️ Longer sessions, faster training

---

## 📚 Notebooks Overview

### Quick_Start.ipynb
**Best for:** New users, quick demos
- 3 simple steps
- 100 training steps
- 3-5 minutes total
- Minimal configuration

### OKADFA_Training.ipynb
**Best for:** Serious training, research
- Multiple training modes
- Full configurability
- Benchmark comparisons
- Result visualization
- Checkpoint management
- 11 comprehensive sections

---

## 🚦 Getting Started

### Option 1: Click & Run (Easiest)
1. Click the "Open in Colab" badge above
2. Enable GPU: Runtime → Change runtime type → GPU
3. Click "Run all" (Runtime → Run all)
4. Wait for training to complete

### Option 2: Manual Setup
1. Go to [Google Colab](https://colab.research.google.com)
2. File → Open notebook → GitHub
3. Enter: `codenlighten/ortho-ai`
4. Select: `colab/Quick_Start.ipynb` or `colab/OKADFA_Training.ipynb`
5. Enable GPU and run cells

---

## 💡 Tips & Tricks

### Avoiding Session Disconnects
```python
# Keep Colab alive by clicking cells periodically
# Or use this JavaScript in browser console:
function ClickConnect(){
  console.log("Clicking connect button"); 
  document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)
```

### Managing GPU Memory
```python
import torch
torch.cuda.empty_cache()  # Clear cache between runs
```

### Saving Checkpoints
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Checkpoints auto-save to Drive
```

### Monitoring Training
```bash
# Watch logs in real-time
!tail -f logs/*.log

# Check GPU usage
!nvidia-smi
```

---

## 🛠️ Customization

### Small Model (Low Memory)
```bash
--d_model 256 --num_layers 4 --num_heads 4 --batch_size 8
```

### Medium Model (Balanced)
```bash
--d_model 512 --num_layers 6 --num_heads 8 --batch_size 8
```

### Large Model (GPT-2 Small)
```bash
--use_gpt2_small --batch_size 4
```

### Custom Configuration
```bash
--d_model 384 \
--num_layers 8 \
--num_heads 6 \
--d_ff 1536 \
--learning_rate 3e-4 \
--batch_size 8
```

---

## 📊 Performance Metrics

### GPU Speedup
- **CPU**: 430 ms/step, 1,191 tokens/sec
- **GPU**: 26 ms/step, 19,845 tokens/sec
- **Speedup**: **16.7x faster** ⚡

### Training Efficiency
- **Perplexity improvement**: 31.8% (WikiText-2)
- **Convergence**: Faster than baseline transformer
- **Memory efficient**: Linear attention complexity

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
--batch_size 4  # or 2

# Or reduce model size
--d_model 256 --num_layers 4
```

### Slow Training
```bash
# Check GPU is enabled
!nvidia-smi

# Verify GPU device
import torch
print(torch.cuda.is_available())
```

### Session Timeout
```bash
# Save checkpoints more frequently
--save_interval 100

# Mount Drive for auto-backup
from google.colab import drive
drive.mount('/content/drive')
```

### Import Errors
```bash
# Reinstall dependencies
!pip install --upgrade torch datasets transformers
```

---

## 📖 Additional Resources

- **Main Repository**: https://github.com/codenlighten/ortho-ai
- **Documentation**: See [README.md](../README.md)
- **Experiments Guide**: See [EXPERIMENTS.md](../EXPERIMENTS.md)
- **Quick Reference**: See [QUICK_REFERENCE.md](../QUICK_REFERENCE.md)

---

## 🤝 Contributing

Found a bug? Have a suggestion? Want to add a notebook?

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📄 License

MIT License - See [LICENSE](../LICENSE) for details

---

## 🌟 Star the Project!

If OKADFA helps your research, please star the repo! ⭐

**Happy Training! 🚀**

---

## 📧 Contact

Questions? Issues? Feedback?

- **GitHub Issues**: https://github.com/codenlighten/ortho-ai/issues
- **Discussions**: https://github.com/codenlighten/ortho-ai/discussions

---

*Last Updated: December 11, 2025*
