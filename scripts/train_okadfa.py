"""
Training Script for OKADFA Model

Copyright (c) 2025 Gregory Ward - SmartLedger.Technology
Licensed under the MIT License (see LICENSE file)

Complete training pipeline with:
- DFA feedback alignment
- Orthogonality regularization
- Gradient diagnostics
- Checkpoint management
- WandB/TensorBoard logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import sys
from pathlib import Path
import time
import json
from typing import Optional, Dict, Any
import math

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.okadfa_model import OKADFAModel, create_gpt2_small_okadfa
from src.training.orthogonality_loss import OrthogonalityLoss
from src.training.dfa_feedback import DFAFeedbackMatrix
from src.training.dfa_backward import DFABackwardHook
from src.diagnostics.gradient_compare import (
    GradientComparator,
    AttentionComparator,
    DiagnosticLogger,
)


class SimpleTextDataset(Dataset):
    """
    Simple text dataset for demonstration.
    
    In production, replace with proper tokenized dataset (WikiText-2, etc.)
    """
    
    def __init__(
        self,
        texts: list,
        vocab_size: int = 50257,
        seq_len: int = 128,
        num_samples: int = 1000,
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings (or token sequences)
            vocab_size: Vocabulary size
            seq_len: Sequence length
            num_samples: Number of samples to generate
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
        # For demo: generate random sequences
        # In production: use proper tokenization
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a single training example.
        
        Returns:
            input_ids: [seq_len]
            labels: [seq_len] (shifted input for language modeling)
        """
        sequence = self.data[idx]
        
        # For language modeling: predict next token
        input_ids = sequence[:-1]  # All but last
        labels = sequence[1:]      # All but first
        
        return input_ids, labels


class OKADFATrainer:
    """
    Trainer for OKADFA models with DFA and diagnostics.
    """
    
    def __init__(
        self,
        model: OKADFAModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
        grad_clip: float = 1.0,
        ortho_lambda: float = 0.01,
        ortho_warmup_steps: int = 1000,
        use_dfa: bool = True,
        log_interval: int = 10,
        eval_interval: int = 100,
        save_interval: int = 500,
        checkpoint_dir: str = "./checkpoints",
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: OKADFA model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            learning_rate: Peak learning rate
            weight_decay: AdamW weight decay
            warmup_steps: Learning rate warmup steps
            max_steps: Maximum training steps
            grad_clip: Gradient clipping threshold
            ortho_lambda: Orthogonality loss weight
            ortho_warmup_steps: Orthogonality warmup steps
            use_dfa: Enable DFA training
            log_interval: Steps between logging
            eval_interval: Steps between evaluation
            save_interval: Steps between checkpoints
            checkpoint_dir: Directory for checkpoints
            use_wandb: Enable WandB logging
            use_tensorboard: Enable TensorBoard logging
            device: Training device
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        self.use_dfa = use_dfa
        
        # Logging
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Orthogonality loss
        self.ortho_loss_fn = OrthogonalityLoss(
            lambda_max=ortho_lambda,
            warmup_steps=ortho_warmup_steps,
        )
        
        # DFA components
        self.feedback_matrix = None
        self.dfa_hook = None
        if use_dfa:
            self._setup_dfa()
        
        # Diagnostics
        self.grad_comparator = GradientComparator(model)
        self.attn_comparator = AttentionComparator()
        self.logger = DiagnosticLogger(
            log_dir=str(self.checkpoint_dir / "logs"),
            use_wandb=use_wandb,
            use_tensorboard=use_tensorboard,
            console_log=True,
        )
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_loss_history = []
        self.val_loss_history = []
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup
                return step / self.warmup_steps
            else:
                # Cosine decay
                progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _setup_dfa(self):
        """Setup DFA feedback matrices and hooks."""
        print("Setting up DFA...")
        
        # Create feedback matrix
        self.feedback_matrix = DFAFeedbackMatrix(
            layer_dims=[self.model.d_model] * self.model.num_layers,
            output_dim=self.model.vocab_size,
            device=self.device,
        )
        
        # Create DFA hook
        self.dfa_hook = DFABackwardHook(
            feedback_matrix=self.feedback_matrix,
            enabled=True,
            store_statistics=True,
        )
        
        # Get DFA modules and register hooks
        dfa_modules = self.model.get_dfa_modules()
        layer_indices = list(range(len(dfa_modules)))
        self.dfa_hook.register_hooks(dfa_modules, layer_indices)
        
        print(f"  ✓ Registered DFA hooks on {len(dfa_modules)} modules")
    
    def train_step(self, batch) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: (input_ids, labels)
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        input_ids, labels = batch
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Language modeling loss
        lm_loss = nn.functional.cross_entropy(
            logits.view(-1, self.model.vocab_size),
            labels.view(-1),
            reduction='mean'
        )
        
        # Orthogonality loss
        ortho_loss = self.model.get_orthogonality_loss(self.ortho_loss_fn)
        # Note: OrthogonalityLoss tracks its own step internally via forward calls
        
        # Total loss
        total_loss = lm_loss + ortho_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.grad_clip
        )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        # Metrics
        metrics = {
            'loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'ortho_loss': ortho_loss.item(),
            'grad_norm': grad_norm.item(),
            'lr': self.scheduler.get_last_lr()[0],
            'ortho_lambda': self.ortho_loss_fn.lambda_max,
        }
        
        return metrics
    
    @torch.no_grad()
    def eval_step(self, batch) -> Dict[str, float]:
        """
        Single evaluation step.
        
        Args:
            batch: (input_ids, labels)
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        input_ids, labels = batch
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, self.model.vocab_size),
            labels.view(-1),
            reduction='mean'
        )
        
        # Perplexity
        perplexity = torch.exp(loss)
        
        return {
            'val_loss': loss.item(),
            'val_perplexity': perplexity.item(),
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Run full evaluation on validation set.
        
        Returns:
            Dictionary of averaged metrics
        """
        if self.val_loader is None:
            return {}
        
        print("Running evaluation...")
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            metrics = self.eval_step(batch)
            total_loss += metrics['val_loss']
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_perplexity = math.exp(avg_loss)
        
        return {
            'val_loss': avg_loss,
            'val_perplexity': avg_perplexity,
        }
    
    def save_checkpoint(self, path: Optional[Path] = None):
        """Save training checkpoint."""
        if path is None:
            path = self.checkpoint_dir / f"checkpoint_step_{self.current_step}.pt"
        
        checkpoint = {
            'step': self.current_step,
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
        }
        
        torch.save(checkpoint, path)
        print(f"✓ Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_step = checkpoint['step']
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_loss_history = checkpoint['train_loss_history']
        self.val_loss_history = checkpoint['val_loss_history']
        
        print(f"✓ Loaded checkpoint from step {self.current_step}")
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("OKADFA TRAINING START")
        print("="*60)
        print(f"Model: {self.model.num_parameters:,} parameters")
        print(f"Device: {self.device}")
        print(f"DFA: {'Enabled' if self.use_dfa else 'Disabled'}")
        print(f"Max steps: {self.max_steps}")
        print("="*60 + "\n")
        
        start_time = time.time()
        epoch = 0
        
        while self.current_step < self.max_steps:
            epoch += 1
            self.current_epoch = epoch
            
            for batch_idx, batch in enumerate(self.train_loader):
                if self.current_step >= self.max_steps:
                    break
                
                # Training step
                step_start = time.time()
                metrics = self.train_step(batch)
                step_time = time.time() - step_start
                
                self.current_step += 1
                self.train_loss_history.append(metrics['loss'])
                
                # Logging
                if self.current_step % self.log_interval == 0:
                    metrics['step'] = self.current_step
                    metrics['epoch'] = epoch
                    metrics['step_time'] = step_time
                    metrics['samples_per_sec'] = batch[0].size(0) / step_time
                    
                    self.logger.log_metrics(metrics, step=self.current_step)
                    
                    # Console log
                    print(f"Step {self.current_step}/{self.max_steps} | "
                          f"Loss: {metrics['loss']:.4f} | "
                          f"LM: {metrics['lm_loss']:.4f} | "
                          f"Ortho: {metrics['ortho_loss']:.6f} | "
                          f"LR: {metrics['lr']:.2e}")
                
                # Evaluation
                if self.current_step % self.eval_interval == 0:
                    eval_metrics = self.evaluate()
                    if eval_metrics:
                        self.logger.log_metrics(eval_metrics, step=self.current_step)
                        self.val_loss_history.append(eval_metrics['val_loss'])
                        
                        # Save best model
                        if eval_metrics['val_loss'] < self.best_val_loss:
                            self.best_val_loss = eval_metrics['val_loss']
                            self.save_checkpoint(self.checkpoint_dir / "best_model.pt")
                        
                        print(f"  Eval | Loss: {eval_metrics['val_loss']:.4f} | "
                              f"PPL: {eval_metrics['val_perplexity']:.2f}")
                
                # Checkpointing
                if self.current_step % self.save_interval == 0:
                    self.save_checkpoint()
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Steps: {self.current_step}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print("="*60 + "\n")
        
        # Save final checkpoint
        self.save_checkpoint(self.checkpoint_dir / "final_model.pt")
        self.logger.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train OKADFA model")
    
    # Model args
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=128)
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # DFA args
    parser.add_argument("--use_dfa", action="store_true", default=True)
    parser.add_argument("--ortho_lambda", type=float, default=0.01)
    parser.add_argument("--ortho_warmup_steps", type=int, default=1000)
    
    # Logging args
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_tensorboard", action="store_true")
    
    # Data args
    parser.add_argument("--num_train_samples", type=int, default=10000)
    parser.add_argument("--num_val_samples", type=int, default=1000)
    
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Create model
    print("Creating model...")
    model = OKADFAModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        use_causal=True,
        orthogonal_init=True,
    )
    print(f"✓ Model created: {model.num_parameters:,} parameters")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SimpleTextDataset(
        texts=[],
        vocab_size=args.vocab_size,
        seq_len=args.max_seq_len,
        num_samples=args.num_train_samples,
    )
    val_dataset = SimpleTextDataset(
        texts=[],
        vocab_size=args.vocab_size,
        seq_len=args.max_seq_len,
        num_samples=args.num_val_samples,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val: {len(val_dataset)} samples")
    
    # Create trainer
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    trainer = OKADFATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        grad_clip=args.grad_clip,
        ortho_lambda=args.ortho_lambda,
        ortho_warmup_steps=args.ortho_warmup_steps,
        use_dfa=args.use_dfa,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb,
        use_tensorboard=args.use_tensorboard,
        device=device,
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
