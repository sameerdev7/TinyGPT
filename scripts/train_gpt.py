#!/usr/bin/env python3
"""
Train GPT on TinyStories with W&B tracking.
Colab-ready version.

Usage:
    python scripts/train_gpt.py --config config/gpt_tinystories.yaml
    python scripts/train_gpt.py --config config/gpt_tinystories.yaml --no-wandb
"""

import argparse
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.language_model import TransformerLM
from training.optimizer import AdamW
from training.loss import cross_entropy
from training.utils import get_batch, gradient_clipping, get_lr_cosine_schedule

# Try to import W&B logger, but make it optional
try:
    from training.wandb_logger import WandBLogger
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# Try to import tokenizer for sample generation
try:
    from tokenization.bpe_trainer import load_tokenizer
    from tokenization.tokenizer import Tokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("Warning: Could not import tokenizer")


def load_config(config_path):
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, device):
    """Estimate train and val loss."""
    model.eval()
    losses = {}
    
    for split, data in [("train", train_data), ("val", val_data)]:
        split_losses = []
        for _ in range(config['training']['eval_iters']):
            x, y = get_batch(
                data,
                config['training']['batch_size'],
                config['model']['context_length'],
                device
            )
            logits = model(x, apply_softmax=False)
            loss = cross_entropy(logits, y)
            split_losses.append(loss.item())
        
        losses[split] = np.mean(split_losses)
    
    model.train()
    return losses


@torch.no_grad()
def generate_samples(model, tokenizer, prompts, max_tokens, device, temperature=0.8):
    """Generate text samples for logging."""
    if not TOKENIZER_AVAILABLE or tokenizer is None:
        return []
    
    model.eval()
    samples = []
    
    for prompt in prompts:
        try:
            # Encode prompt
            input_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            
            # Generate
            for _ in range(max_tokens):
                context = input_tensor[:, -256:]  # Context window
                logits = model(context, apply_softmax=False)
                
                if logits.dim() == 3:
                    next_token_logits = logits[0, -1, :] / temperature
                else:
                    next_token_logits = logits[0] / temperature
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
            
            # Decode
            generated_text = tokenizer.decode(input_tensor[0].tolist())
            samples.append(generated_text)
        except Exception as e:
            print(f"Warning: Sample generation failed: {e}")
            samples.append(f"{prompt} [generation failed]")
    
    model.train()
    return samples


def train(config_path, use_wandb=True, wandb_entity=None):
    """Main training function."""
    # Load config
    config = load_config(config_path)
    print("="*70)
    print(f"Training: {config['experiment_name']}")
    print("="*70)
    
    # Set device
    device = get_device()
    print(f"Device: {device}")
    
    # Set seed
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    
    # Initialize W&B
    logger = None
    if use_wandb and WANDB_AVAILABLE:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{config['experiment_name']}_{timestamp}"
            
            logger = WandBLogger(
                project="tinystories-gpt",
                name=run_name,
                config=config,
                entity=wandb_entity,
                tags=["transformer", "language-model", "tinystories"],
                notes=config.get('description', '')
            )
            
            # Log config file
            if os.path.exists(config_path):
                logger.log_config_file(Path(config_path))
        except Exception as e:
            print(f"Warning: W&B initialization failed: {e}")
            logger = None
    elif use_wandb and not WANDB_AVAILABLE:
        print("Warning: W&B requested but not available. Install with: pip install wandb")
    
    # Load data
    print("Loading data...")
    processed_dir = config['data']['processed_dir']

    train_data = np.memmap(
        f"{processed_dir}/train.bin",
        dtype='uint16',
        mode='r'
    )
    val_data = np.memmap(
        f"{processed_dir}/val.bin",
        dtype='uint16',
        mode='r'
    )     
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    
    # Load tokenizer for sample generation
    tokenizer = None
    if TOKENIZER_AVAILABLE and logger:
        try:
            tokenizer_dir = config['data']['tokenizer_dir']
            vocab_path = Path(tokenizer_dir) / "vocab.pkl"
            merges_path = Path(tokenizer_dir) / "merges.pkl"
            
            if vocab_path.exists() and merges_path.exists():
                vocab, merges = load_tokenizer(str(vocab_path), str(merges_path))
                tokenizer = Tokenizer(vocab, merges)
                print("✓ Tokenizer loaded for sample generation")
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
    
    # Create model
    print("Creating model...")
    model = TransformerLM(
        vocab_size=config['model']['vocab_size'],
        context_length=config['model']['context_length'],
        d_model=config['model']['d_model'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        rope_theta=config['model']['rope_theta'],
        device=device,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Watch model with W&B
    if logger:
        logger.watch_model(model, log_freq=100)
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(config['training']['beta1'], config['training']['beta2']),
        weight_decay=config['training']['weight_decay'],
    )
    
    # Create checkpoint dir
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Sample prompts for generation
    sample_prompts = [
        "Once upon a time",
        "One day, a little girl",
        "There was a friendly dog"
    ]
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    model.train()
    
    pbar = tqdm(range(config['training']['max_iters']), desc="Training")
    best_val_loss = float('inf')
    
    for iter_num in pbar:
        # Get learning rate
        lr = get_lr_cosine_schedule(
            iter_num,
            config['training']['learning_rate'],
            config['training']['min_lr'],
            config['training']['warmup_iters'],
            config['training']['max_iters'],
        )
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Log learning rate
        if logger:
            logger.log_learning_rate(lr, iter_num)
        
        # Evaluate
        if iter_num % config['training']['eval_interval'] == 0:
            losses = estimate_loss(model, train_data, val_data, config, device)
            perplexity = np.exp(losses['val'])
            
            print(f"\n[{iter_num:5d}] train: {losses['train']:.4f}, val: {losses['val']:.4f}, "
                  f"ppl: {perplexity:.2f}, lr: {lr:.2e}")
            
            # Log to W&B
            if logger:
                logger.log({
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "val/perplexity": perplexity,
                    "train/tokens_seen": iter_num * config['training']['batch_size'] * config['model']['context_length'],
                }, step=iter_num)
                
                # Generate samples periodically
                if iter_num % (config['training']['eval_interval'] * 2) == 0 and tokenizer:
                    try:
                        samples = generate_samples(
                            model, tokenizer, sample_prompts,
                            max_tokens=50, device=device
                        )
                        if samples:
                            logger.log_text_samples(sample_prompts, samples, iter_num)
                    except Exception as e:
                        print(f"Warning: Sample generation failed: {e}")
            
            # Check for best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if logger:
                    logger.log({"val/best_loss": best_val_loss}, step=iter_num)
        
        # Training step
        x, y = get_batch(
            train_data,
            config['training']['batch_size'],
            config['model']['context_length'],
            device
        )
        
        logits = model(x, apply_softmax=False)
        loss = cross_entropy(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Log gradient norm
        if logger and iter_num % config['training']['log_interval'] == 0:
            logger.log_gradient_norm(model, iter_num)
        
        if config['training']['grad_clip'] > 0:
            gradient_clipping(model.parameters(), config['training']['grad_clip'])
        
        optimizer.step()
        
        # Log training metrics
        if logger and iter_num % config['training']['log_interval'] == 0:
            logger.log({
                "train/loss_step": loss.item(),
                "train/perplexity_step": np.exp(loss.item()),
            }, step=iter_num, commit=False)
            logger.log_system_metrics(iter_num)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{lr:.2e}',
            'best_val': f'{best_val_loss:.4f}'
        })
        
        # Checkpoint
        if iter_num > 0 and iter_num % config['training']['checkpoint_interval'] == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iter_num}.pt"
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': iter_num,
                'config': config,
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")
            
            # Log checkpoint to W&B
            if logger:
                logger.log_checkpoint(
                    checkpoint_path,
                    iter_num,
                    metadata={
                        "val_loss": best_val_loss,
                        "train_loss": loss.item(),
                    }
                )
    
    # Final checkpoint
    final_path = checkpoint_dir / "checkpoint_final.pt"
    final_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': config['training']['max_iters'],
        'config': config,
        'best_val_loss': best_val_loss,
    }
    torch.save(final_data, final_path)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print(f"Final checkpoint: {final_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best perplexity: {np.exp(best_val_loss):.2f}")
    
    # Log final checkpoint
    if logger:
        logger.log_checkpoint(
            final_path,
            config['training']['max_iters'],
            metadata={
                "best_val_loss": best_val_loss,
                "final": True,
            }
        )
        logger.finish()


def main():
    parser = argparse.ArgumentParser(description="Train GPT on TinyStories")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (username/team)")
    args = parser.parse_args()
    
    train(args.config, use_wandb=not args.no_wandb, wandb_entity=args.wandb_entity)


if __name__ == "__main__":
    main()
