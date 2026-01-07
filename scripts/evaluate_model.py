#!/usr/bin/env python3
"""
Comprehensive model evaluation script.
Colab-ready version.

Usage:
    python scripts/evaluate_model.py --checkpoint checkpoints/checkpoint_final.pt
    python scripts/evaluate_model.py --checkpoint checkpoints/checkpoint_final.pt --save-report
"""

import argparse
import sys
from pathlib import Path
import json
from datetime import datetime
import os

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.language_model import TransformerLM
from tokenization.bpe_trainer import load_tokenizer
from tokenization.tokenizer import Tokenizer
from evaluation.metrics import (
    calculate_perplexity,
    calculate_token_accuracy,
    evaluate_generation_quality,
)


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model_and_tokenizer(checkpoint_path, tokenizer_dir, device):
    """Load model and tokenizer."""
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    if not config:
        raise ValueError("Checkpoint does not contain config!")
    
    model_config = config['model']
    
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=model_config['vocab_size'],
        context_length=model_config['context_length'],
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        rope_theta=model_config['rope_theta'],
        device=device,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Loading tokenizer...")
    vocab_path = Path(tokenizer_dir) / "vocab.pkl"
    merges_path = Path(tokenizer_dir) / "merges.pkl"
    
    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(f"Tokenizer files not found in {tokenizer_dir}")
    
    vocab, merges = load_tokenizer(str(vocab_path), str(merges_path))
    tokenizer = Tokenizer(vocab, merges)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded: {total_params:,} parameters")
    
    return model, tokenizer, config


def generate_text(model, tokenizer, prompt, max_tokens, device, temperature=0.8):
    """Generate text from prompt."""
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Use context window
            context = input_tensor[:, -256:]
            logits = model(context, apply_softmax=False)
            
            if logits.dim() == 3:
                next_token_logits = logits[0, -1, :] / temperature
            else:
                next_token_logits = logits[0] / temperature
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
    
    return tokenizer.decode(input_tensor[0].tolist())


def evaluate_model(
    checkpoint_path,
    tokenizer_dir="data/tokenizer",
    data_dir="data/processed",
    num_samples=50,
    save_report=False,
    output_dir="evaluation_results"
):
    """Run comprehensive model evaluation."""
    
    print("="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    device = get_device()
    print(f"Device: {device}\n")
    
    # Load model and tokenizer
    model, tokenizer, config = load_model_and_tokenizer(
        checkpoint_path, tokenizer_dir, device
    )
    
    # Load test data
    print("\nLoading data...")
    val_path = os.path.join(data_dir, "val.bin")
    
    if not os.path.exists(val_path):
        print(f"Warning: Validation data not found at {val_path}")
        print("Skipping perplexity evaluation...")
        val_data = None
    else:
        val_data = np.memmap(val_path, dtype='uint16', mode='r')
        print(f"Validation tokens: {len(val_data):,}")
    
    results = {}
    
    # 1. Perplexity Evaluation
    if val_data is not None:
        print("\n" + "="*70)
        print("1. PERPLEXITY EVALUATION")
        print("="*70)
        
        batch_size = config['training']['batch_size']
        context_length = config['model']['context_length']
        
        try:
            perplexity, avg_loss = calculate_perplexity(
                model, val_data, batch_size, context_length, device, num_batches=100
            )
            
            print(f"Validation Loss: {avg_loss:.4f}")
            print(f"Validation Perplexity: {perplexity:.2f}")
            
            results['perplexity'] = {
                'loss': float(avg_loss),
                'perplexity': float(perplexity),
            }
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            results['perplexity'] = {
                'loss': None,
                'perplexity': None,
                'error': str(e)
            }
    
    # 2. Token Accuracy
    if val_data is not None:
        print("\n" + "="*70)
        print("2. TOKEN ACCURACY")
        print("="*70)
        
        try:
            accuracy = calculate_token_accuracy(
                model, val_data, batch_size, context_length, device, num_batches=50
            )
            
            print(f"Token-level Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            results['accuracy'] = {
                'token_accuracy': float(accuracy),
            }
        except Exception as e:
            print(f"Error calculating accuracy: {e}")
            results['accuracy'] = {
                'token_accuracy': None,
                'error': str(e)
            }
    
    # 3. Generation Quality
    print("\n" + "="*70)
    print("3. GENERATION QUALITY")
    print("="*70)
    
    test_prompts = [
        "Once upon a time, there was a",
        "One day, a little girl",
        "There was a friendly dog who",
        "In a big forest, lived a",
        "A curious boy found a",
    ]
    
    print(f"Generating {num_samples} samples...")
    all_generated = []
    
    samples_per_prompt = max(1, num_samples // len(test_prompts))
    
    for prompt in tqdm(test_prompts, desc="Generating"):
        for _ in range(samples_per_prompt):
            try:
                generated = generate_text(
                    model, tokenizer, prompt, max_tokens=100, device=device
                )
                all_generated.append(generated)
            except Exception as e:
                print(f"Warning: Generation failed for prompt '{prompt}': {e}")
                continue
    
    # Calculate generation metrics
    if all_generated:
        print("\nCalculating generation metrics...")
        try:
            gen_metrics = evaluate_generation_quality(all_generated)
            
            print("\nGeneration Metrics:")
            for key, value in gen_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            results['generation'] = gen_metrics
        except Exception as e:
            print(f"Error calculating generation metrics: {e}")
            results['generation'] = {'error': str(e)}
    else:
        print("Warning: No samples generated successfully")
        results['generation'] = {}
    
    # 4. Sample Outputs
    print("\n" + "="*70)
    print("4. SAMPLE OUTPUTS")
    print("="*70)
    
    sample_outputs = []
    for i, prompt in enumerate(test_prompts[:3]):
        try:
            print(f"\nPrompt {i+1}: {prompt}")
            generated = generate_text(
                model, tokenizer, prompt, max_tokens=80, device=device, temperature=0.8
            )
            print(f"Generated: {generated}")
            print("-" * 70)
            sample_outputs.append({
                'prompt': prompt,
                'generated': generated,
            })
        except Exception as e:
            print(f"Error generating sample: {e}")
            sample_outputs.append({
                'prompt': prompt,
                'generated': None,
                'error': str(e)
            })
    
    results['samples'] = sample_outputs
    
    # 5. Model Statistics
    print("\n" + "="*70)
    print("5. MODEL STATISTICS")
    print("="*70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    stats = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'vocab_size': config['model']['vocab_size'],
        'context_length': config['model']['context_length'],
        'd_model': config['model']['d_model'],
        'num_layers': config['model']['num_layers'],
        'num_heads': config['model']['num_heads'],
        'd_ff': config['model']['d_ff'],
    }
    
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: ~{total_params * 4 / 1024**2:.2f} MB (FP32)")
    print(f"Vocabulary Size: {config['model']['vocab_size']:,}")
    print(f"Context Length: {config['model']['context_length']}")
    print(f"Hidden Dimension: {config['model']['d_model']}")
    print(f"Layers: {config['model']['num_layers']}")
    print(f"Attention Heads: {config['model']['num_heads']}")
    
    results['model_stats'] = stats
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    if 'perplexity' in results and results['perplexity'].get('perplexity'):
        print(f"✓ Perplexity: {results['perplexity']['perplexity']:.2f}")
    
    if 'accuracy' in results and results['accuracy'].get('token_accuracy'):
        print(f"✓ Token Accuracy: {results['accuracy']['token_accuracy']*100:.2f}%")
    
    if 'generation' in results and 'diversity/avg_length' in results['generation']:
        print(f"✓ Avg Text Length: {results['generation']['diversity/avg_length']:.1f} tokens")
        print(f"✓ Unique Unigrams: {results['generation']['diversity/unique_unigrams']*100:.1f}%")
        print(f"✓ Coherence Score: {results['generation']['coherence/avg']*100:.1f}%")
    
    # Save report
    if save_report:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"evaluation_report_{timestamp}.json"
        
        results['metadata'] = {
            'checkpoint': str(checkpoint_path),
            'timestamp': timestamp,
            'device': str(device),
        }
        
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Report saved to: {report_file}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="data/tokenizer",
        help="Tokenizer directory"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Data directory"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to generate"
    )
    
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save evaluation report to file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for reports"
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        checkpoint_path=args.checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        save_report=args.save_report,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
