import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import sys
from pathlib import Path
import pickle
from typing import Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.language_model import TransformerLM
from tokenization.tokenizer import Tokenizer

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 600
    temperature: float = 0.85
    top_k: Optional[int] = 50

app = FastAPI(title="TinyStories GPT Inference API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

@app.on_event("startup")
async def load_model():
    """Load model and tokenizer at startup."""
    checkpoint_path = "checkpoints/checkpoint_final.pt"
    tokenizer_dir = "data/tokenizer"
    
    print("Loading tokenizer...")
    vocab_path = Path(tokenizer_dir) / "vocab.pkl"
    merges_path = Path(tokenizer_dir) / "merges.pkl"
    
    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(
            f"Tokenizer files not found in {tokenizer_dir}. "
            "Expected vocab.pkl and merges.pkl"
        )
    
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_path, "rb") as f:
        merges = pickle.load(f)
    
    app.state.tokenizer = Tokenizer(vocab, merges)
    print(f"Tokenizer loaded! Vocab size: {len(vocab)}")
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    device = get_device()
    print(f"Using device: {device}")
    
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    config = checkpoint.get('config', {})
    
    if not config:
        raise ValueError("Checkpoint does not contain config!")
    
    model_config = config['model']
    
    app.state.model = TransformerLM(
        vocab_size=model_config['vocab_size'],
        context_length=model_config['context_length'],
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        rope_theta=model_config['rope_theta'],
        device=device,
    ).to(device)
    
    app.state.model.load_state_dict(checkpoint['model_state_dict'])
    app.state.model.eval()
    app.state.device = device
    app.state.config = config
    
    total_params = sum(p.numel() for p in app.state.model.parameters())
    print(f"Model loaded successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Context length: {model_config['context_length']}")

async def generate_tokens(request: GenerateRequest):
    """Generate tokens one at a time for streaming."""
    tokenizer = app.state.tokenizer
    model = app.state.model
    device = app.state.device
    config = app.state.config
    
    # Encode prompt
    input_ids = tokenizer.encode(request.prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    context_length = config['model']['context_length']
    
    # Generation loop
    for _ in range(request.max_tokens):
        # Get context window (last context_length tokens)
        context = input_tensor[:, -context_length:]
        
        # Forward pass
        with torch.no_grad():
            logits = model(context, apply_softmax=False)
        
        # Get logits for last position
        if logits.dim() == 2:
            next_token_logits = logits[0]
        elif logits.dim() == 3:
            next_token_logits = logits[0, -1, :]
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")
        
        # Apply temperature
        if request.temperature > 0:
            next_token_logits = next_token_logits / request.temperature
        
        # Apply top-k filtering
        if request.top_k is not None:
            top_k_values, _ = torch.topk(next_token_logits, min(request.top_k, next_token_logits.size(-1)))
            kth_value = top_k_values[-1]
            indices_to_remove = next_token_logits < kth_value
            next_token_logits[indices_to_remove] = float('-inf')
        
        # Sample next token
        if request.temperature == 0:
            next_token = torch.argmax(next_token_logits, dim=-1)
        else:
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        
        next_token_id = next_token.item()
        
        # Decode token
        token_text = tokenizer.decode([next_token_id])
        
        # Yield token text
        yield token_text
        
        # Add to input tensor
        new_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        input_tensor = torch.cat([input_tensor, new_token_tensor], dim=1)
        
        # Small delay for streaming effect
        await asyncio.sleep(0)

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text endpoint with streaming response."""
    if not hasattr(app.state, 'model') or app.state.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return StreamingResponse(
        generate_tokens(request),
        media_type="text/plain"
    )

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, 'model') and app.state.model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
