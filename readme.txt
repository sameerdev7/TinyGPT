## Training Summary

This model was trained from scratch on the TinyStories dataset using a GPT-style Transformer.

- **Model size:** 18M parameters  
- **Architecture:** Decoder-only Transformer (GPT)  
- **Dataset:** TinyStories  
- **Training tokens:** ~77.8M  
- **Context length:** <your block size>  
- **Tokenizer:** Custom BPE tokenizer  
- **Optimizer:** AdamW  
- **Training platform:** Google Colab (GPU)  
- **Experiment tracking:** Weights & Biases (W&B)

### Final Metrics
- **Best validation loss:** 1.906  
- **Best perplexity:** 6.73  


### Experiment Tracking

All training metrics, checkpoints, and configurations were logged using Weights & Biases.

🔗 **W&B Run:**  
https://wandb.ai/sameer7sayyad-siddhant-college-of-engg/tinystories-gpt/runs/sghwrfsg

