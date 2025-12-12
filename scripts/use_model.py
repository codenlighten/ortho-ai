"""
Load and use trained OKADFA models for text generation.

This script demonstrates how to:
1. Load a trained checkpoint
2. Generate text completions
3. Compute perplexity on new text

Copyright (c) 2025 Gregory Ward - SmartLedger.Technology
Licensed under the MIT License.

Usage:
    # Generate text from prompt
    python scripts/use_model.py --checkpoint checkpoints_gpu_fixed/best_model.pt \
                                 --prompt "The future of AI is"
    
    # Evaluate perplexity on text
    python scripts/use_model.py --checkpoint checkpoints_gpu_fixed/best_model.pt \
                                 --evaluate "This is a test sentence."
    
    # Interactive mode
    python scripts/use_model.py --checkpoint checkpoints_gpu_fixed/best_model.pt \
                                 --interactive
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import torch
import torch.nn.functional as F
from models.okadfa_model import OKADFAModel
from data.tokenizer import get_tokenizer


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load a trained OKADFA model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config - infer from state dict if not saved
    config = checkpoint.get('config', {})
    
    # Infer dimensions from checkpoint if config not available
    if not config or 'vocab_size' not in config:
        state_dict = checkpoint['model_state_dict']
        
        # Get vocab_size from embedding weight
        vocab_size = state_dict['token_embedding.weight'].shape[0]
        d_model = state_dict['token_embedding.weight'].shape[1]
        
        # Count layers
        num_layers = sum(1 for k in state_dict.keys() if k.startswith('layers.') and k.endswith('.norm1.weight'))
        
        # Get other params from first layer
        num_heads = 8  # Default, will try to infer
        d_ff = state_dict['layers.0.ff.fc1.weight'].shape[0]
        
        # Get max_seq_len from positional encoding
        max_seq_len = state_dict['pos_encoding.pe'].shape[1]
        
        # Get num_random_features from projection matrix
        # projection_matrix shape is [d_head, num_random_features]
        proj_matrix = state_dict['layers.0.attention.favor_attention.feature_map.projection_matrix']
        d_head = proj_matrix.shape[0]
        num_random_features = proj_matrix.shape[1]
        num_heads = d_model // d_head
        
        config = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'd_ff': d_ff,
            'num_random_features': num_random_features,
            'max_seq_len': max_seq_len,
            'dropout': 0.1,
        }
        
        print(f"\nInferred Model Configuration:")
    else:
        print(f"\nModel Configuration:")
    
    num_params = checkpoint.get('num_parameters', sum(p.numel() for p in checkpoint['model_state_dict'].values()))
    print(f"  Parameters: {num_params:,}")
    print(f"  Dimensions: {config['d_model']}")
    print(f"  Layers: {config['num_layers']}")
    print(f"  Heads: {config['num_heads']}")
    
    # Create model
    model = OKADFAModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_random_features=config.get('num_random_features', 256),
        max_seq_len=config.get('max_seq_len', 1024),
        dropout=config.get('dropout', 0.1),
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"\nModel loaded successfully!")
    print(f"Training steps: {checkpoint.get('step', 'Unknown')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'Unknown'):.4f}")
    
    return model, config


def generate_text(model, tokenizer, prompt: str, max_length: int = 50, 
                  temperature: float = 1.0, device: str = 'cuda'):
    """Generate text completion from a prompt."""
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generating {max_length} tokens...\n")
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions
            logits = model(generated)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop at end of sequence token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    
    print("=" * 80)
    print(generated_text)
    print("=" * 80)
    
    return generated_text


def evaluate_perplexity(model, tokenizer, text: str, device: str = 'cuda'):
    """Calculate perplexity of text."""
    model.eval()
    
    # Encode text
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    with torch.no_grad():
        # Get predictions
        logits = model(input_ids)
        
        # Calculate loss (shift for next token prediction)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean'
        )
        
        perplexity = torch.exp(loss).item()
    
    print(f"\nText: {text}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    
    return perplexity


def interactive_mode(model, tokenizer, device: str = 'cuda'):
    """Interactive text generation."""
    print("\n" + "=" * 80)
    print("OKADFA Interactive Mode")
    print("=" * 80)
    print("Type your prompt and press Enter to generate text.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 80 + "\n")
    
    while True:
        try:
            prompt = input("\nPrompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            # Generate
            generate_text(model, tokenizer, prompt, max_length=50, device=device)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Use trained OKADFA models")
    
    # Model loading
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    # Generation options
    parser.add_argument('--prompt', type=str, default=None,
                       help='Text prompt for generation')
    parser.add_argument('--max_length', type=int, default=50,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    
    # Evaluation
    parser.add_argument('--evaluate', type=str, default=None,
                       help='Text to evaluate perplexity on')
    
    # Interactive
    parser.add_argument('--interactive', action='store_true',
                       help='Enter interactive mode')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    # Load tokenizer
    tokenizer = get_tokenizer()
    
    # Execute requested action
    if args.interactive:
        interactive_mode(model, tokenizer, device)
    elif args.prompt:
        generate_text(model, tokenizer, args.prompt, args.max_length, 
                     args.temperature, device)
    elif args.evaluate:
        evaluate_perplexity(model, tokenizer, args.evaluate, device)
    else:
        print("\nNo action specified. Use --prompt, --evaluate, or --interactive")
        print("Example: python scripts/use_model.py --checkpoint <path> --prompt 'Hello'")


if __name__ == '__main__':
    main()
