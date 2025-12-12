"""
Quick demo of using trained OKADFA models.

This script shows examples with your available checkpoints:
- checkpoints_gpu_fixed/best_model.pt (20M params, PPL 1,876)
- Colab checkpoints from checkpoints.zip (up to 124M params, PPL 604)

Copyright (c) 2025 Gregory Ward - SmartLedger.Technology
Licensed under the MIT License.
"""

print("=" * 80)
print("OKADFA Trained Model Demo")
print("=" * 80)

# Example 1: Generate text with your best local model
print("\n1. Text Generation Example:")
print("-" * 80)
print("Command:")
print("  python scripts/use_model.py \\")
print("    --checkpoint checkpoints_gpu_fixed/best_model.pt \\")
print("    --prompt 'The future of artificial intelligence is'")
print("\nThis will complete the prompt with 50 tokens of generated text.")

# Example 2: Evaluate perplexity
print("\n\n2. Perplexity Evaluation Example:")
print("-" * 80)
print("Command:")
print("  python scripts/use_model.py \\")
print("    --checkpoint checkpoints_gpu_fixed/best_model.pt \\")
print("    --evaluate 'Machine learning models can learn complex patterns from data.'")
print("\nThis will compute how well the model predicts the given text.")

# Example 3: Interactive mode
print("\n\n3. Interactive Generation:")
print("-" * 80)
print("Command:")
print("  python scripts/use_model.py \\")
print("    --checkpoint checkpoints_gpu_fixed/best_model.pt \\")
print("    --interactive")
print("\nThis starts an interactive session where you can try multiple prompts.")

# Example 4: Using Colab models
print("\n\n4. Using Your Colab Models:")
print("-" * 80)
print("If you have checkpoints.zip from Colab:")
print("  1. Extract: unzip checkpoints.zip")
print("  2. Use the GPT-2 Small model (best performance, PPL 604):")
print("\n  python scripts/use_model.py \\")
print("    --checkpoint checkpoints/gpt2_small/best_model.pt \\")
print("    --prompt 'Once upon a time' \\")
print("    --max_length 100")

# Available models summary
print("\n\n" + "=" * 80)
print("YOUR AVAILABLE MODELS")
print("=" * 80)
print("\nLocal Checkpoints:")
print("  ✓ checkpoints_gpu_fixed/best_model.pt")
print("    - 20M parameters")
print("    - Validation PPL: 1,876")
print("    - Training: 450 steps on RTX 3070")
print("    - Best for: Quick inference, development")
print()
print("  ✓ checkpoints_wikitext/best_model.pt")
print("    - 14M parameters")
print("    - Validation PPL: 34,822")
print("    - Training: Earlier run (before LR fix)")
print("    - Note: Lower quality than gpu_fixed")

print("\n\nColab Checkpoints (in checkpoints.zip):")
print("  ✓ Quick Test Model")
print("    - 20M parameters")
print("    - Validation PPL: 16,798")
print("    - Training: 100 steps on T4")
print()
print("  ✓ Extended Training Model")
print("    - 37M parameters")
print("    - Validation PPL: 1,104")
print("    - Training: 1000 steps on A100")
print("    - Best for: Good quality, reasonable size")
print()
print("  ✓ GPT-2 Small Scale Model ⭐ BEST")
print("    - 124M parameters")
print("    - Validation PPL: 604")
print("    - Training: 5000 steps on A100")
print("    - Best for: Production quality text generation")
print()
print("  ✓ Custom Configuration Model")
print("    - 33M parameters")
print("    - Validation PPL: 745")
print("    - Training: 2000 steps on A100")

print("\n\n" + "=" * 80)
print("RECOMMENDED USAGE")
print("=" * 80)
print("\nFor quick testing:")
print("  Use: checkpoints_gpu_fixed/best_model.pt")
print("  Why: Fast loading, good enough for demos")
print()
print("For best quality:")
print("  Use: GPT-2 Small from checkpoints.zip (PPL 604)")
print("  Why: Lowest perplexity, best text generation")
print()
print("For balanced performance:")
print("  Use: Extended model from checkpoints.zip (PPL 1,104)")
print("  Why: Good quality, smaller size, faster inference")

print("\n\n" + "=" * 80)
print("QUICK START")
print("=" * 80)
print("\n# Try your best local model right now:")
print("python scripts/use_model.py \\")
print("  --checkpoint checkpoints_gpu_fixed/best_model.pt \\")
print("  --prompt 'The OKADFA architecture' \\")
print("  --max_length 50")

print("\n\n# Or try interactive mode:")
print("python scripts/use_model.py \\")
print("  --checkpoint checkpoints_gpu_fixed/best_model.pt \\")
print("  --interactive")

print("\n" + "=" * 80)
