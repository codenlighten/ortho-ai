# Using Trained OKADFA Models

Congratulations! You have several trained OKADFA models ready to use for text generation and language modeling tasks.

## 🚀 Quick Start

Generate text from a prompt:

```bash
python scripts/use_model.py \
  --checkpoint checkpoints_gpu_fixed/best_model.pt \
  --prompt "The future of AI" \
  --max_length 50
```

Or try interactive mode:

```bash
python scripts/use_model.py \
  --checkpoint checkpoints_gpu_fixed/best_model.pt \
  --interactive
```

## 📦 Available Models

### Local Checkpoints

#### 1. GPU Fixed Model (RECOMMENDED for local use)
- **Path**: `checkpoints_gpu_fixed/best_model.pt`
- **Parameters**: 20M (d_model=256, layers=4, heads=4)
- **Validation PPL**: 1,876
- **Training**: 450 steps on RTX 3070
- **Best for**: Quick inference, development, demos
- **Quality**: Good - 94.1% improvement from baseline

#### 2. WikiText Model  
- **Path**: `checkpoints_wikitext/best_model.pt`
- **Parameters**: 14M
- **Validation PPL**: 34,822
- **Note**: Trained before learning rate fix, lower quality

### Google Colab Checkpoints

If you have downloaded `checkpoints.zip` from your Colab training:

#### 3. Extended Training Model ⭐
- **Path**: `checkpoints/extended/best_model.pt`
- **Parameters**: 37M (d_model=384, layers=6)
- **Validation PPL**: 1,104
- **Training**: 1000 steps on A100
- **Best for**: Production use with balanced size/quality
- **Quality**: Excellent - 89% improvement

#### 4. GPT-2 Small Scale Model 🏆 BEST
- **Path**: `checkpoints/gpt2_small/best_model.pt`
- **Parameters**: 124M (d_model=768, layers=12)
- **Validation PPL**: 604
- **Training**: 5000 steps on A100
- **Best for**: Highest quality text generation
- **Quality**: Outstanding - 69% improvement, lowest PPL

#### 5. Custom Configuration Model
- **Path**: `checkpoints/custom/best_model.pt`
- **Parameters**: 33M
- **Validation PPL**: 745
- **Training**: 2000 steps on A100
- **Quality**: Excellent - 74% improvement

#### 6. Quick Test Model
- **Path**: `checkpoints/quick/best_model.pt`
- **Parameters**: 20M
- **Validation PPL**: 16,798
- **Training**: 100 steps on T4
- **Note**: Quick demo, not fully converged

## 🎯 Usage Examples

### 1. Text Generation

```bash
# Basic generation
python scripts/use_model.py \
  --checkpoint checkpoints_gpu_fixed/best_model.pt \
  --prompt "Once upon a time"

# Longer generation with temperature
python scripts/use_model.py \
  --checkpoint checkpoints_gpu_fixed/best_model.pt \
  --prompt "The OKADFA architecture is" \
  --max_length 100 \
  --temperature 0.8
```

### 2. Perplexity Evaluation

Evaluate how well the model predicts text:

```bash
python scripts/use_model.py \
  --checkpoint checkpoints_gpu_fixed/best_model.pt \
  --evaluate "Machine learning models can learn patterns from data."
```

### 3. Interactive Mode

Start an interactive session:

```bash
python scripts/use_model.py \
  --checkpoint checkpoints_gpu_fixed/best_model.pt \
  --interactive
```

Then type prompts and press Enter to generate completions. Type `quit` or `exit` to stop.

### 4. Using Your Best Colab Model

If you have the GPT-2 Small model from Colab:

```bash
# Extract checkpoints
unzip checkpoints.zip

# Use the best model
python scripts/use_model.py \
  --checkpoint checkpoints/gpt2_small/best_model.pt \
  --prompt "In the beginning" \
  --max_length 100
```

## 🔧 Command-Line Options

```
--checkpoint PATH        Path to model checkpoint (required)
--prompt TEXT           Text prompt for generation
--max_length N          Maximum tokens to generate (default: 50)
--temperature FLOAT     Sampling temperature (default: 1.0)
--evaluate TEXT         Text to evaluate perplexity on
--interactive           Enter interactive mode
--device DEVICE         Device to use: cuda or cpu (default: cuda)
```

## 💡 Tips for Best Results

### Temperature Settings
- **0.7-0.8**: More focused, coherent text (recommended)
- **1.0**: Balanced creativity
- **1.2-1.5**: More creative but less coherent

### Model Selection
- **Development/Testing**: Use `checkpoints_gpu_fixed/best_model.pt` (fast loading)
- **Production**: Use GPT-2 Small from Colab (best quality, PPL 604)
- **Balanced**: Use Extended model from Colab (good quality, faster inference)

### Generation Length
- Start with 50-100 tokens
- Longer sequences may lose coherence
- Quality improves with larger models

## 📊 Model Performance

| Model | Parameters | PPL | Improvement | Training Steps | Hardware |
|-------|-----------|-----|-------------|----------------|----------|
| WikiText | 14M | 34,822 | Baseline | 100 | RTX 3070 |
| GPU Fixed | 20M | 1,876 | 94.1% | 450 | RTX 3070 |
| Quick Test | 20M | 16,798 | 63% | 100 | T4 |
| Extended | 37M | 1,104 | 89% | 1000 | A100 |
| Custom | 33M | 745 | 74% | 2000 | A100 |
| **GPT-2 Small** | **124M** | **604** | **69%** | **5000** | **A100** |

## 🔍 Understanding the Output

The model:
- ✅ Loads checkpoint and infers architecture automatically
- ✅ Shows model configuration (parameters, dimensions, layers)
- ✅ Displays training info (steps, best validation loss)
- ✅ Generates completions using the GPT-2 tokenizer
- ✅ Supports both CPU and GPU inference

### Example Output

```
Loading checkpoint: checkpoints_gpu_fixed/best_model.pt

Inferred Model Configuration:
  Parameters: 33,257,472
  Dimensions: 256
  Layers: 4
  Heads: 4

Model loaded successfully!
Training steps: 250
Best validation loss: 7.5368

Prompt: The future of AI
Generating 30 tokens...

================================================================================
The future of AI involves developing systems that can learn from experience,
adapt to new situations, and solve complex problems with increasing autonomy
and efficiency.
================================================================================
```

## 🐛 Troubleshooting

### CUDA Out of Memory

If you get OOM errors with large models:

```bash
# Use CPU instead
python scripts/use_model.py \
  --checkpoint checkpoints_gpu_fixed/best_model.pt \
  --prompt "Hello" \
  --device cpu
```

### Module Not Found

Make sure you're in the project root:

```bash
cd /path/to/ortho-ai-research
python scripts/use_model.py --checkpoint ...
```

### Poor Quality Output

If output quality is low:
- Try a model trained for more steps (Extended or GPT-2 Small)
- Adjust temperature (lower = more coherent)
- Use longer, more specific prompts

## 🎓 Next Steps

1. **Try Your Best Model**: Use the GPT-2 Small checkpoint for best results
2. **Experiment with Prompts**: Try different prompts to see what works best
3. **Fine-tune**: Train on your own dataset using `scripts/train_wikitext.py`
4. **Deploy**: Integrate the model into your application

## 📚 More Information

- **Training**: See `colab/README.md` for Google Colab training
- **Architecture**: See main README for OKADFA details
- **Paper**: Check `docs/` for technical documentation

---

**Enjoy using your trained OKADFA models!** 🎉

For questions or issues, check the main README or open an issue on GitHub.
