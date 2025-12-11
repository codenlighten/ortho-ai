"""
Training script for OKADFA on WikiText-2 dataset.

This script demonstrates training with real text data using proper tokenization.

Copyright (c) 2025 Gregory Ward - SmartLedger.Technology
Licensed under the MIT License. See LICENSE file in the project root.

Usage:
    # Quick test (small model, short training)
    python scripts/train_wikitext.py --quick_test
    
    # Full WikiText-2 training
    python scripts/train_wikitext.py --dataset wikitext-2 --max_steps 10000
    
    # Multi-GPU training
    python scripts/train_wikitext.py --device cuda --num_gpus 4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
from tqdm import tqdm
from typing import Optional

from models.okadfa_model import OKADFAModel, create_gpt2_small_okadfa
from training.dfa_feedback import DFAFeedbackMatrix
from training.dfa_backward import DFABackwardHook
from training.orthogonality_loss import OrthogonalityLoss
from data.wikitext_loader import create_wikitext_dataloaders
from data.text_dataset import SimpleTextDataset
from data.tokenizer import get_tokenizer


class WikiTextTrainer:
    """Trainer for OKADFA on WikiText dataset."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        max_steps: int = 10000,
        warmup_steps: int = 500,
        grad_clip: float = 1.0,
        ortho_lambda_start: float = 0.0,
        ortho_lambda_max: float = 1e-4,
        ortho_warmup_steps: Optional[int] = None,
        log_interval: int = 100,
        eval_interval: int = 500,
        checkpoint_dir: str = './checkpoints',
        save_interval: int = 1000,
    ):
        """
        Initialize trainer.
        
        Args:
            model: OKADFA model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay
            max_steps: Maximum training steps
            warmup_steps: LR warmup steps
            grad_clip: Gradient clipping threshold
            ortho_lambda_start: Initial orthogonality penalty
            ortho_lambda_max: Maximum orthogonality penalty
            ortho_warmup_steps: Orthogonality warmup steps
            log_interval: Logging interval
            eval_interval: Evaluation interval
            checkpoint_dir: Checkpoint directory
            save_interval: Checkpoint save interval
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Training hyperparameters
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        
        # LR scheduler (cosine with warmup)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_steps - warmup_steps,
            eta_min=learning_rate * 0.1,
        )
        
        # Loss functions
        self.lm_loss_fn = nn.CrossEntropyLoss()
        
        # Orthogonality loss
        if ortho_warmup_steps is None:
            ortho_warmup_steps = int(0.1 * max_steps)  # 10% of training
        
        self.ortho_loss_fn = OrthogonalityLoss(
            lambda_start=ortho_lambda_start,
            lambda_max=ortho_lambda_max,
            warmup_steps=ortho_warmup_steps,
        )
        
        # Setup DFA hooks
        self._setup_dfa_hooks()
        
        # Training state
        self.step = 0
        self.best_val_loss = float('inf')
    
    def _setup_dfa_hooks(self):
        """Setup DFA backward hooks."""
        # Get all linear modules that should use DFA
        dfa_modules = self.model.get_all_linear_modules()
        
        if not dfa_modules:
            print("Warning: No DFA modules found!")
            return
        
        # Get output dimension (vocab size)
        output_dim = self.model.lm_head.out_features
        
        # Create feedback matrices for each module
        self.feedback_matrices = []
        self.dfa_hooks = []
        
        for i, module in enumerate(dfa_modules):
            # Get module output dimension
            if hasattr(module, 'out_features'):
                module_output_dim = module.out_features
            elif hasattr(module, 'embed_dim'):
                module_output_dim = module.embed_dim
            else:
                module_output_dim = module.weight.shape[0]
            
            # Create feedback matrix
            feedback_matrix = DFAFeedbackMatrix(
                output_dim=module_output_dim,
                final_output_dim=output_dim,
            ).to(self.device)
            
            self.feedback_matrices.append(feedback_matrix)
            
            # Create and register hook
            hook = DFABackwardHook(feedback_matrix)
            self.dfa_hooks.append(hook)
        
        # Register all hooks at once
        if self.dfa_hooks:
            self.dfa_hooks[0].register_hooks(dfa_modules, list(range(len(dfa_modules))))
            print(f"Registered {len(self.dfa_hooks)} DFA hooks")
    
    def _warmup_lr(self, step: int) -> float:
        """Calculate learning rate with warmup."""
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        return 1.0
    
    def train_step(self, batch) -> dict:
        """Execute one training step."""
        self.model.train()
        
        # Move batch to device
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Language modeling loss
        lm_loss = self.lm_loss_fn(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1)
        )
        
        # Orthogonality loss
        ortho_loss = self.ortho_loss_fn(self.model, self.step)
        
        # Total loss
        total_loss = lm_loss + ortho_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.grad_clip
        )
        
        # Optimizer step with warmup
        warmup_factor = self._warmup_lr(self.step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * warmup_factor
        
        self.optimizer.step()
        
        # LR scheduler (after warmup)
        if self.step >= self.warmup_steps:
            self.scheduler.step()
        
        self.step += 1
        
        return {
            'loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'ortho_loss': ortho_loss.item(),
            'lr': self.optimizer.param_groups[0]['lr'],
            'grad_norm': grad_norm.item(),
            'ortho_lambda': self.ortho_loss_fn.lambda_max,
        }
    
    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            input_ids, target_ids = batch
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            
            # Loss
            loss = self.lm_loss_fn(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )
            
            total_loss += loss.item() * target_ids.numel()
            total_tokens += target_ids.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(min(avg_loss, 20))  # Cap for numerical stability
        
        return {
            'val_loss': avg_loss,
            'val_ppl': perplexity,
        }
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.max_steps} steps...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        
        train_iter = iter(self.train_loader)
        
        while self.step < self.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Training step
            metrics = self.train_step(batch)
            
            # Logging
            if self.step % self.log_interval == 0:
                print(
                    f"Step {self.step}/{self.max_steps} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"LM: {metrics['lm_loss']:.4f} | "
                    f"Ortho: {metrics['ortho_loss']:.4f} | "
                    f"LR: {metrics['lr']:.2e} | "
                    f"Grad: {metrics['grad_norm']:.2f}"
                )
            
            # Evaluation
            if self.step % self.eval_interval == 0:
                eval_metrics = self.evaluate()
                print(
                    f"\nEvaluation at step {self.step}:"
                    f"\n  Val Loss: {eval_metrics['val_loss']:.4f}"
                    f"\n  Val PPL: {eval_metrics['val_ppl']:.2f}"
                )
                
                # Save best model
                if eval_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = eval_metrics['val_loss']
                    self.save_checkpoint('best_model.pt')
            
            # Periodic checkpoint
            if self.step % self.save_interval == 0:
                self.save_checkpoint(f'checkpoint_step_{self.step}.pt')
        
        # Final checkpoint
        self.save_checkpoint('final_model.pt')
        print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description='Train OKADFA on WikiText')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='wikitext-2-raw-v1',
                      choices=['wikitext-2-raw-v1', 'wikitext-103-raw-v1'],
                      help='Dataset to use')
    parser.add_argument('--seq_length', type=int, default=512,
                      help='Sequence length')
    parser.add_argument('--cache_dir', type=str, default=None,
                      help='Dataset cache directory')
    
    # Model
    parser.add_argument('--use_gpt2_small', action='store_true',
                      help='Use GPT-2 Small config (124M params)')
    parser.add_argument('--vocab_size', type=int, default=50257,
                      help='Vocabulary size (GPT-2 default)')
    parser.add_argument('--d_model', type=int, default=768,
                      help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=6,
                      help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=12,
                      help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=3072,
                      help='Feed-forward dimension')
    parser.add_argument('--num_random_features', type=int, default=256,
                      help='Number of random features for Favor+')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                      help='Weight decay')
    parser.add_argument('--max_steps', type=int, default=10000,
                      help='Maximum training steps')
    parser.add_argument('--warmup_steps', type=int, default=500,
                      help='Warmup steps')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                      help='Gradient clipping threshold')
    
    # Orthogonality
    parser.add_argument('--ortho_lambda_start', type=float, default=0.0,
                      help='Initial orthogonality penalty')
    parser.add_argument('--ortho_lambda_max', type=float, default=1e-4,
                      help='Maximum orthogonality penalty')
    parser.add_argument('--ortho_warmup_steps', type=int, default=None,
                      help='Orthogonality warmup steps')
    
    # Logging & Checkpointing
    parser.add_argument('--log_interval', type=int, default=100,
                      help='Logging interval')
    parser.add_argument('--eval_interval', type=int, default=500,
                      help='Evaluation interval')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_wikitext',
                      help='Checkpoint directory')
    parser.add_argument('--save_interval', type=int, default=1000,
                      help='Checkpoint save interval')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=0,
                      help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                      help='Quick test with small model and data')
    
    args = parser.parse_args()
    
    # Quick test configuration
    if args.quick_test:
        print("Running in quick test mode...")
        args.d_model = 256
        args.num_layers = 2
        args.num_heads = 4
        args.d_ff = 1024
        args.seq_length = 128
        args.batch_size = 4
        args.max_steps = 100
        args.warmup_steps = 20
        args.eval_interval = 50
        args.save_interval = 100
        args.device = 'cpu'
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Get tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer('gpt2')
    args.vocab_size = tokenizer.vocab_size
    
    # Create dataloaders
    print(f"Loading {args.dataset} dataset...")
    train_loader, val_loader = create_wikitext_dataloaders(
        dataset_name=args.dataset,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
        max_train_samples=1000 if args.quick_test else None,
        max_val_samples=200 if args.quick_test else None,
    )
    
    # Create model
    print("Creating model...")
    if args.use_gpt2_small:
        model = create_gpt2_small_okadfa(
            vocab_size=args.vocab_size,
            max_seq_length=args.seq_length,
            num_random_features=args.num_random_features,
        )
    else:
        model = OKADFAModel(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            max_seq_length=args.seq_length,
            num_random_features=args.num_random_features,
        )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = WikiTextTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        ortho_lambda_start=args.ortho_lambda_start,
        ortho_lambda_max=args.ortho_lambda_max,
        ortho_warmup_steps=args.ortho_warmup_steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval,
    )
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
