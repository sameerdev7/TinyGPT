"""Evaluation metrics for language models."""

import torch
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
import math


def calculate_perplexity(model, data, batch_size, context_length, device, num_batches=100):
    """
    Calculate perplexity on a dataset.
    
    Args:
        model: Language model
        data: Token data (numpy array or memmap)
        batch_size: Batch size for evaluation
        context_length: Context window length
        device: Device to run on
        num_batches: Number of batches to evaluate
        
    Returns:
        perplexity: Model perplexity
        loss: Average cross-entropy loss
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    # Limit number of batches if dataset is too small
    max_possible_batches = (len(data) - context_length) // (batch_size * context_length)
    num_batches = min(num_batches, max_possible_batches)
    
    if num_batches == 0:
        print("Warning: Dataset too small for evaluation")
        return float('inf'), float('inf')
    
    with torch.no_grad():
        for i in range(num_batches):
            # Sample batch
            start_idx = i * batch_size * context_length
            batch_data = []
            
            for j in range(batch_size):
                idx = start_idx + j * context_length
                if idx + context_length + 1 < len(data):
                    batch_data.append(data[idx:idx + context_length + 1])
            
            if len(batch_data) < batch_size:
                break
            
            batch_data = np.stack(batch_data)
            x = torch.from_numpy(batch_data[:, :-1]).long().to(device)
            y = torch.from_numpy(batch_data[:, 1:]).long().to(device)
            
            # Forward pass
            logits = model(x, apply_softmax=False)
            
            # Calculate loss
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            
            loss = torch.nn.functional.cross_entropy(logits_flat, y_flat, reduction='sum')
            
            total_loss += loss.item()
            total_tokens += y_flat.size(0)
    
    if total_tokens == 0:
        return float('inf'), float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    model.train()
    return perplexity, avg_loss


def calculate_token_accuracy(model, data, batch_size, context_length, device, num_batches=50):
    """
    Calculate token-level accuracy (top-1).
    
    Args:
        model: Language model
        data: Token data
        batch_size: Batch size
        context_length: Context length
        device: Device
        num_batches: Number of batches to evaluate
        
    Returns:
        accuracy: Token-level accuracy (0-1)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(num_batches):
            try:
                # Sample random positions
                max_start = len(data) - context_length - 1
                if max_start <= 0:
                    break
                
                start_indices = np.random.randint(0, max_start, size=batch_size)
                
                x = np.stack([data[idx:idx + context_length] for idx in start_indices])
                y = np.stack([data[idx + 1:idx + context_length + 1] for idx in start_indices])
                
                x = torch.from_numpy(x).long().to(device)
                y = torch.from_numpy(y).long().to(device)
                
                logits = model(x, apply_softmax=False)
                predictions = torch.argmax(logits, dim=-1)
                
                correct += (predictions == y).sum().item()
                total += y.numel()
            except Exception as e:
                print(f"Warning: Batch {i} failed: {e}")
                continue
    
    if total == 0:
        return 0.0
    
    accuracy = correct / total
    model.train()
    return accuracy


def calculate_diversity_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Calculate text diversity metrics.
    
    Args:
        texts: List of generated texts
        
    Returns:
        Dictionary with diversity metrics:
        - unique_unigrams: Ratio of unique unigrams
        - unique_bigrams: Ratio of unique bigrams
        - unique_trigrams: Ratio of unique trigrams
        - avg_length: Average text length in tokens
        - vocab_size: Number of unique tokens
    """
    if not texts:
        return {
            "unique_unigrams": 0.0,
            "unique_bigrams": 0.0,
            "unique_trigrams": 0.0,
            "avg_length": 0.0,
            "vocab_size": 0,
        }
    
    all_tokens = []
    all_bigrams = []
    all_trigrams = []
    
    for text in texts:
        tokens = text.lower().split()
        all_tokens.extend(tokens)
        
        # Bigrams
        for i in range(len(tokens) - 1):
            all_bigrams.append((tokens[i], tokens[i + 1]))
        
        # Trigrams
        for i in range(len(tokens) - 2):
            all_trigrams.append((tokens[i], tokens[i + 1], tokens[i + 2]))
    
    metrics = {
        "unique_unigrams": len(set(all_tokens)) / max(len(all_tokens), 1),
        "unique_bigrams": len(set(all_bigrams)) / max(len(all_bigrams), 1),
        "unique_trigrams": len(set(all_trigrams)) / max(len(all_trigrams), 1),
        "avg_length": np.mean([len(text.split()) for text in texts]) if texts else 0.0,
        "vocab_size": len(set(all_tokens)),
    }
    
    return metrics


def calculate_repetition_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Calculate repetition metrics.
    
    Args:
        texts: List of generated texts
        
    Returns:
        Dictionary with repetition metrics:
        - rep_2: Ratio of repeated 2-grams
        - rep_3: Ratio of repeated 3-grams
        - rep_4: Ratio of repeated 4-grams
    """
    if not texts:
        return {"rep_2": 0.0, "rep_3": 0.0, "rep_4": 0.0}
    
    def get_ngram_repetition(tokens: List[str], n: int) -> float:
        if len(tokens) < n:
            return 0.0
        
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0
        
        counter = Counter(ngrams)
        repeated = sum(count - 1 for count in counter.values() if count > 1)
        return repeated / len(ngrams)
    
    rep_scores = {2: [], 3: [], 4: []}
    
    for text in texts:
        tokens = text.lower().split()
        for n in [2, 3, 4]:
            rep_scores[n].append(get_ngram_repetition(tokens, n))
    
    return {
        "rep_2": np.mean(rep_scores[2]) if rep_scores[2] else 0.0,
        "rep_3": np.mean(rep_scores[3]) if rep_scores[3] else 0.0,
        "rep_4": np.mean(rep_scores[4]) if rep_scores[4] else 0.0,
    }


def calculate_bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    Calculate BLEU score (simplified version).
    
    Args:
        reference: Reference text
        hypothesis: Generated text
        max_n: Maximum n-gram order
        
    Returns:
        BLEU score (0-1)
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    if len(hyp_tokens) == 0:
        return 0.0
    
    # Brevity penalty
    bp = 1.0 if len(hyp_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(hyp_tokens))
    
    # Calculate precision for each n-gram order
    precisions = []
    for n in range(1, max_n + 1):
        if len(hyp_tokens) < n:
            precisions.append(0.0)
            continue
        
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)])
        hyp_ngrams = Counter([tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1)])
        
        overlap = sum((ref_ngrams & hyp_ngrams).values())
        total = sum(hyp_ngrams.values())
        
        precision = overlap / total if total > 0 else 0.0
        precisions.append(precision)
    
    # Geometric mean
    if any(p == 0 for p in precisions):
        return 0.0
    
    geo_mean = math.exp(sum(math.log(p + 1e-10) for p in precisions) / len(precisions))
    bleu = bp * geo_mean
    
    return bleu


def calculate_coherence_score(text: str) -> float:
    """
    Simple coherence score based on sentence structure.
    
    Args:
        text: Generated text
        
    Returns:
        Coherence score (0-1)
    """
    if not text.strip():
        return 0.0
    
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 2:
        return 0.5
    
    # Check for reasonable sentence lengths
    lengths = [len(s.split()) for s in sentences]
    avg_length = np.mean(lengths)
    std_length = np.std(lengths)
    
    # Penalize very short or very long sentences
    length_score = 1.0 if 5 <= avg_length <= 20 else 0.5
    
    # Penalize high variance in length
    variance_score = 1.0 if std_length < 10 else 0.7
    
    # Check for proper capitalization
    capital_score = sum(1 for s in sentences if s and s[0].isupper()) / len(sentences)
    
    coherence = (length_score + variance_score + capital_score) / 3
    return coherence


def evaluate_generation_quality(
    generated_texts: List[str],
    reference_texts: List[str] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation of generated texts.
    
    Args:
        generated_texts: List of generated texts
        reference_texts: Optional list of reference texts for BLEU
        
    Returns:
        Dictionary with all metrics
    """
    if not generated_texts:
        return {}
    
    metrics = {}
    
    # Diversity metrics
    diversity = calculate_diversity_metrics(generated_texts)
    metrics.update({f"diversity/{k}": v for k, v in diversity.items()})
    
    # Repetition metrics
    repetition = calculate_repetition_metrics(generated_texts)
    metrics.update({f"repetition/{k}": v for k, v in repetition.items()})
    
    # Coherence
    coherence_scores = [calculate_coherence_score(text) for text in generated_texts]
    metrics["coherence/avg"] = np.mean(coherence_scores) if coherence_scores else 0.0
    
    # BLEU scores if references provided
    if reference_texts and len(reference_texts) == len(generated_texts):
        bleu_scores = [
            calculate_bleu_score(ref, gen)
            for ref, gen in zip(reference_texts, generated_texts)
        ]
        metrics["bleu/avg"] = np.mean(bleu_scores) if bleu_scores else 0.0
    
    return metrics
