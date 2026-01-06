"""
Text generation script using trained Transformer checkpoints.

This script loads a trained model checkpoint and generates text using various
sampling strategies including greedy decoding, temperature sampling, top-k, and top-p.

Usage:
    # Greedy decoding (deterministic)
    uv run generate_text.py --checkpoint checkpoints/checkpoint_final.pt \
        --prompt "Once upon a time"

    # With temperature sampling
    uv run generate_text.py --checkpoint checkpoints/checkpoint_final.pt \
        --prompt "Once upon a time" \
        --temperature 0.8 \
        --max-tokens 100

    # With top-k sampling
    uv run generate_text.py --checkpoint checkpoints/checkpoint_final.pt \
        --prompt "Once upon a time" \
        --top-k 50

    # Multiple samples
    uv run generate_text.py --checkpoint checkpoints/checkpoint_final.pt \
        --prompt "Once upon a time" \
        --num-samples 5 \
        --temperature 0.9
"""

#!/usr/bin/env python3
"""
Text generation script...
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # ← THIS LINE FIXES IT

import argparse
# ... rest of your imports
from typing import Optional, List
import torch
import torch.nn.functional as F
from tokenization.bpe_trainer import load_tokenizer
from tokenization.tokenizer import Tokenizer
from models.language_model import TransformerLM


class TextGenerator:
    """Text generator using a trained Transformer language model.

    Attributes:
        model: Trained TransformerLM
        tokenizer: BPE tokenizer
        device: torch device (cuda/mps/cpu)
        config: Model configuration from checkpoint
    """

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_dir: str = "tokenizer",
        device: Optional[str] = None
    ):
        """Initialize the text generator.

        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            tokenizer_dir: Directory containing vocab.pkl and merges.pkl
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
        """
        self.device = self._get_device(device)
        print(f"Using device: {self.device}")

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint.get('config', {})

        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_dir}...")
        vocab_path = Path(tokenizer_dir) / "vocab.pkl"
        merges_path = Path(tokenizer_dir) / "merges.pkl"

        if not vocab_path.exists() or not merges_path.exists():
            raise FileNotFoundError(
                f"Tokenizer files not found in {tokenizer_dir}. "
                f"Expected vocab.pkl and merges.pkl"
            )

        vocab, merges = load_tokenizer(str(vocab_path), str(merges_path))
        self.tokenizer = Tokenizer(vocab, merges)
        print(f"Loaded tokenizer with vocab size: {len(vocab)}")

        # Initialize model using config from checkpoint
        print("Initializing model from checkpoint config...")
        full_config = checkpoint.get('config', {})
        if not full_config:
            raise ValueError("Checkpoint does not contain config! Cannot initialize model.")

        model_config = full_config['model']  # ← This is the key fix!

        print(f"Model config: {model_config}")

        self.model = TransformerLM(
            vocab_size=model_config['vocab_size'],
            context_length=model_config['context_length'],
            d_model=model_config['d_model'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            d_ff=model_config['d_ff'],
            rope_theta=model_config['rope_theta'],
            device=self.device,
        ).to(self.device)

        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded successfully!")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Context length: {self.config.get('context_length', 256)}")
        print(f"  Model dimension: {self.config.get('d_model', 512)}")
        print(f"  Layers: {self.config.get('num_layers', 4)}")
        print()

    def _get_device(self, device: Optional[str]) -> torch.device:
        """Get the device for inference."""
        if device:
            return torch.device(device)

        # Auto-detect
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_token: Optional[str] = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text to continue
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
                        Use 1.0 for standard sampling, 0.0 for greedy
            top_k: Keep only top k tokens with highest probability (None = no filtering)
            top_p: Keep tokens with cumulative probability >= top_p (None = no filtering)
            stop_token: Stop generation if this token is generated

        Returns:
            Generated text (prompt + generated continuation)
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate tokens
        generated_ids = input_ids.copy()

        for step in range(max_tokens):
            # Get context window (last context_length tokens)
            context_length = self.config.get('context_length', 256)
            context = input_tensor[:, -context_length:]

            # Forward pass
            try:
                with torch.no_grad():
                    logits = self.model(context, apply_softmax=False)
            except Exception as e:
                print(f"\nError in forward pass at step {step}")
                print(f"Context shape: {context.shape}")
                print(f"Context device: {context.device}")
                print(f"Error: {e}")
                raise

            # Handle different output shapes
            if logits.dim() == 2:
                # Model returned [batch_size, vocab_size] - already at last position
                next_token_logits = logits[0]
            elif logits.dim() == 3:
                # Model returned [batch_size, seq_len, vocab_size]
                next_token_logits = logits[0, -1, :]
            else:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")

            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                # Get the k-th largest value
                top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                kth_value = top_k_values[-1]
                # Set all values below the k-th largest to -inf
                indices_to_remove = next_token_logits < kth_value
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[0] = False

                # Map back to original indices
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            if temperature == 0:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)
                next_token_id = next_token.item()
            else:
                # Sample from distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_token_id = next_token.item()

            # Append to generated sequence
            generated_ids.append(next_token_id)

            # Add new token to input tensor
            # Create a 2D tensor of shape [1, 1] to concatenate
            new_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
            input_tensor = torch.cat([input_tensor, new_token_tensor], dim=1)

            # Check for stop token
            if stop_token:
                next_token_text = self.tokenizer.decode([next_token_id])
                if stop_token in next_token_text:
                    break

        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids)
        return generated_text

    def generate_multiple(
        self,
        prompt: str,
        num_samples: int = 3,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> List[str]:
        """Generate multiple samples from a prompt.

        Args:
            prompt: Input text to continue
            num_samples: Number of samples to generate
            max_tokens: Maximum tokens per sample
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering

        Returns:
            List of generated texts
        """
        samples = []
        for i in range(num_samples):
            print(f"Generating sample {i+1}/{num_samples}...")
            sample = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            samples.append(sample)
        return samples


def main():
    """Main function for command-line text generation."""
    parser = argparse.ArgumentParser(
        description="Generate text using a trained Transformer model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )

    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="tokenizer",
        help="Directory containing tokenizer files (vocab.pkl, merges.pkl)"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input text prompt to continue"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (0.0 = greedy, higher = more random)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Keep only top k tokens (None = no filtering)"
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling threshold (None = no filtering)"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=['cuda', 'mps', 'cpu'],
        help="Device to use (None = auto-detect)"
    )

    parser.add_argument(
        "--stop-token",
        type=str,
        default=None,
        help="Stop generation if this token appears"
    )

    args = parser.parse_args()

    # Initialize generator
    try:
        generator = TextGenerator(
            checkpoint_path=args.checkpoint,
            tokenizer_dir=args.tokenizer_dir,
            device=args.device
        )
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate text
    print("=" * 80)
    print("GENERATING TEXT")
    print("=" * 80)
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    if args.top_k:
        print(f"Top-k: {args.top_k}")
    if args.top_p:
        print(f"Top-p: {args.top_p}")
    print("=" * 80)
    print()

    try:
        if args.num_samples == 1:
            # Generate single sample
            generated_text = generator.generate(
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                stop_token=args.stop_token,
            )

            print("Generated text:")
            print("-" * 80)
            print(generated_text)
            print("-" * 80)
        else:
            # Generate multiple samples
            samples = generator.generate_multiple(
                prompt=args.prompt,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

            for i, sample in enumerate(samples, 1):
                print(f"\nSample {i}:")
                print("-" * 80)
                print(sample)
                print("-" * 80)

    except Exception as e:
        print(f"Error during generation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

