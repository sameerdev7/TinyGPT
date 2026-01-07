"""Weights & Biases integration for experiment tracking."""

import wandb
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any


class WandBLogger:
    """Weights & Biases experiment tracker."""
    
    def __init__(
        self,
        project: str,
        name: str,
        config: Dict[str, Any],
        entity: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        resume: bool = False,
    ):
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name
            name: Run name (experiment name)
            config: Configuration dictionary
            entity: W&B entity (username/team)
            tags: List of tags for organizing runs
            notes: Notes about this run
            resume: Whether to resume previous run
        """
        self.enabled = True
        
        try:
            # Initialize wandb
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                entity=entity,
                tags=tags,
                notes=notes,
                resume="allow" if resume else False,
            )
            
            # Watch model (will be set later)
            self.model_watched = False
            
            print(f"✓ W&B initialized: {self.run.url}")
            
        except Exception as e:
            print(f"Warning: W&B initialization failed: {e}")
            print("Continuing without W&B logging...")
            self.enabled = False
            self.run = None
    
    def watch_model(self, model: torch.nn.Module, log_freq: int = 100):
        """
        Watch model gradients and parameters.
        
        Args:
            model: PyTorch model to watch
            log_freq: How often to log histograms
        """
        if not self.enabled or self.model_watched:
            return
        
        try:
            wandb.watch(model, log="all", log_freq=log_freq)
            self.model_watched = True
            print("✓ Model gradients and parameters being tracked")
        except Exception as e:
            print(f"Warning: Failed to watch model: {e}")
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Global step number
            commit: Whether to commit immediately
        """
        if not self.enabled:
            return
        
        try:
            wandb.log(metrics, step=step, commit=commit)
        except Exception as e:
            print(f"Warning: Failed to log metrics: {e}")
    
    def log_text_samples(
        self,
        prompts: list,
        generated_texts: list,
        step: int,
        table_name: str = "generated_samples"
    ):
        """
        Log generated text samples to W&B.
        
        Args:
            prompts: List of input prompts
            generated_texts: List of generated texts
            step: Current training step
            table_name: Name for the W&B table
        """
        if not self.enabled:
            return
        
        try:
            # Create a W&B table
            columns = ["step", "prompt", "generated_text", "length"]
            data = []
            
            for prompt, text in zip(prompts, generated_texts):
                data.append([
                    step,
                    prompt,
                    text,
                    len(text.split())
                ])
            
            table = wandb.Table(columns=columns, data=data)
            wandb.log({table_name: table}, step=step)
            
        except Exception as e:
            print(f"Warning: Failed to log text samples: {e}")
    
    def log_learning_rate(self, lr: float, step: int):
        """Log current learning rate."""
        self.log({"learning_rate": lr}, step=step, commit=False)
    
    def log_gradient_norm(self, model: torch.nn.Module, step: int):
        """
        Log gradient norms.
        
        Args:
            model: PyTorch model
            step: Current step
        """
        if not self.enabled:
            return
        
        try:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            self.log({"gradient_norm": total_norm}, step=step, commit=False)
        except Exception as e:
            print(f"Warning: Failed to log gradient norm: {e}")
    
    def log_system_metrics(self, step: int):
        """Log system metrics (GPU memory, etc.)."""
        if not self.enabled:
            return
        
        try:
            metrics = {}
            
            # GPU metrics
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    metrics[f"gpu_{i}_memory_allocated_gb"] = torch.cuda.memory_allocated(i) / 1e9
                    metrics[f"gpu_{i}_memory_reserved_gb"] = torch.cuda.memory_reserved(i) / 1e9
            
            if metrics:
                self.log(metrics, step=step, commit=False)
                
        except Exception as e:
            print(f"Warning: Failed to log system metrics: {e}")
    
    def log_checkpoint(self, checkpoint_path: Path, step: int, metadata: Optional[Dict] = None):
        """
        Log model checkpoint as W&B artifact.
        
        Args:
            checkpoint_path: Path to checkpoint file
            step: Training step
            metadata: Additional metadata
        """
        if not self.enabled:
            return
        
        try:
            artifact = wandb.Artifact(
                name=f"model-checkpoint-{step}",
                type="model",
                metadata=metadata or {}
            )
            artifact.add_file(str(checkpoint_path))
            self.run.log_artifact(artifact)
            print(f"✓ Checkpoint logged to W&B: {checkpoint_path.name}")
        except Exception as e:
            print(f"Warning: Failed to log checkpoint: {e}")
    
    def log_config_file(self, config_path: Path):
        """Log configuration file as artifact."""
        if not self.enabled:
            return
        
        try:
            artifact = wandb.Artifact(name="config", type="config")
            artifact.add_file(str(config_path))
            self.run.log_artifact(artifact)
        except Exception as e:
            print(f"Warning: Failed to log config: {e}")
    
    def finish(self):
        """Finish W&B run."""
        if self.enabled and self.run is not None:
            try:
                wandb.finish()
                print("✓ W&B run finished")
            except Exception as e:
                print(f"Warning: Failed to finish W&B run: {e}")
    
    def alert(self, title: str, text: str, level: str = "INFO"):
        """
        Send W&B alert.
        
        Args:
            title: Alert title
            text: Alert message
            level: Alert level (INFO, WARN, ERROR)
        """
        if not self.enabled:
            return
        
        try:
            wandb.alert(title=title, text=text, level=getattr(wandb.AlertLevel, level))
        except Exception as e:
            print(f"Warning: Failed to send alert: {e}")
