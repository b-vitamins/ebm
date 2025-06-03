"""Callback system for training energy-based models.

This module provides a flexible callback system that allows users to
hook into various points of the training process for logging,
checkpointing, early stopping, and custom behaviors.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from ..core.logging import logger
from ..models.base import EnergyBasedModel


class Callback(ABC):
    """Abstract base class for training callbacks."""
    
    def on_train_begin(self, trainer: Any) -> None:
        """Called at the beginning of training."""
        pass
        
    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        pass
        
    def on_epoch_start(self, trainer: Any, model: EnergyBasedModel) -> None:
        """Called at the start of each epoch."""
        pass
        
    def on_epoch_end(
        self,
        trainer: Any,
        model: EnergyBasedModel,
        metrics: Dict[str, float]
    ) -> None:
        """Called at the end of each epoch."""
        pass
        
    def on_batch_start(
        self,
        trainer: Any,
        model: EnergyBasedModel,
        batch: Tensor
    ) -> None:
        """Called before processing each batch."""
        pass
        
    def on_batch_end(
        self,
        trainer: Any,
        model: EnergyBasedModel,
        loss: float
    ) -> None:
        """Called after processing each batch."""
        pass
        
    def on_validation_start(self, trainer: Any, model: EnergyBasedModel) -> None:
        """Called before validation."""
        pass
        
    def on_validation_end(
        self,
        trainer: Any,
        model: EnergyBasedModel,
        metrics: Dict[str, float]
    ) -> None:
        """Called after validation."""
        pass


class CallbackList:
    """Container for multiple callbacks."""
    
    def __init__(self, callbacks: List[Callback]):
        """Initialize callback list.
        
        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks
        self._should_stop = False
        
    @property
    def should_stop(self) -> bool:
        """Check if any callback requested stopping."""
        return self._should_stop
        
    def stop_training(self) -> None:
        """Signal that training should stop."""
        self._should_stop = True
        
    def __getattr__(self, name: str):
        """Delegate method calls to all callbacks."""
        def method(*args, **kwargs):
            for callback in self.callbacks:
                if hasattr(callback, name):
                    getattr(callback, name)(*args, **kwargs)
        return method


class LoggingCallback(Callback):
    """Callback for logging training progress."""
    
    def __init__(
        self,
        log_every: int = 100,
        log_gradients: bool = False,
        log_weights: bool = False
    ):
        """Initialize logging callback.
        
        Args:
            log_every: Frequency of logging (in steps)
            log_gradients: Whether to log gradient statistics
            log_weights: Whether to log weight statistics
        """
        self.log_every = log_every
        self.log_gradients = log_gradients
        self.log_weights = log_weights
        self.step_count = 0
        self.epoch_start_time = None
        
    def on_epoch_start(self, trainer: Any, model: EnergyBasedModel) -> None:
        """Log epoch start."""
        self.epoch_start_time = time.time()
        logger.info(
            f"Starting epoch {trainer.current_epoch}",
            lr=trainer.optimizer.param_groups[0]['lr']
        )
        
    def on_epoch_end(
        self,
        trainer: Any,
        model: EnergyBasedModel,
        metrics: Dict[str, float]
    ) -> None:
        """Log epoch end."""
        epoch_time = time.time() - self.epoch_start_time
        
        # Format metrics for logging
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        
        logger.info(
            f"Epoch {trainer.current_epoch} completed",
            metrics=metric_str,
            epoch_time=f"{epoch_time:.1f}s"
        )
        
    def on_batch_end(
        self,
        trainer: Any,
        model: EnergyBasedModel,
        loss: float
    ) -> None:
        """Log batch statistics."""
        self.step_count += 1
        
        if self.step_count % self.log_every == 0:
            log_dict = {
                'step': trainer.global_step,
                'loss': loss,
            }
            
            if self.log_gradients:
                # Log gradient statistics
                grad_norms = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
                        
                if grad_norms:
                    log_dict['grad_norm_mean'] = np.mean(grad_norms)
                    log_dict['grad_norm_max'] = np.max(grad_norms)
                    
            if self.log_weights:
                # Log weight statistics
                weight_norms = []
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        weight_norms.append(param.norm().item())
                        
                if weight_norms:
                    log_dict['weight_norm_mean'] = np.mean(weight_norms)
                    
            logger.debug("Training step", **log_dict)


class MetricsCallback(Callback):
    """Callback for tracking and storing metrics."""
    
    def __init__(self, save_path: Optional[Path] = None):
        """Initialize metrics callback.
        
        Args:
            save_path: Optional path to save metrics
        """
        self.save_path = save_path
        self.train_metrics = []
        self.val_metrics = []
        
    def on_epoch_end(
        self,
        trainer: Any,
        model: EnergyBasedModel,
        metrics: Dict[str, float]
    ) -> None:
        """Store training metrics."""
        metrics['epoch'] = trainer.current_epoch
        metrics['timestamp'] = time.time()
        self.train_metrics.append(metrics)
        
        if self.save_path:
            self._save_metrics()
            
    def on_validation_end(
        self,
        trainer: Any,
        model: EnergyBasedModel,
        metrics: Dict[str, float]
    ) -> None:
        """Store validation metrics."""
        metrics['epoch'] = trainer.current_epoch
        metrics['timestamp'] = time.time()
        self.val_metrics.append(metrics)
        
        if self.save_path:
            self._save_metrics()
            
    def _save_metrics(self) -> None:
        """Save metrics to file."""
        data = {
            'train': self.train_metrics,
            'val': self.val_metrics
        }
        
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)


class CheckpointCallback(Callback):
    """Callback for saving model checkpoints."""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        save_every: int = 10,
        save_best: bool = True,
        monitor: str = 'val_loss',
        mode: str = 'min'
    ):
        """Initialize checkpoint callback.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_every: Save frequency (in epochs)
            save_best: Whether to save best model
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' for monitored metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_every = save_every
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
    def on_epoch_end(
        self,
        trainer: Any,
        model: EnergyBasedModel,
        metrics: Dict[str, float]
    ) -> None:
        """Save checkpoint if needed."""
        epoch = trainer.current_epoch
        
        # Regular checkpoint
        if epoch % self.save_every == 0:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            trainer.save_checkpoint(path)
            
        # Best model checkpoint
        if self.save_best and self.monitor in metrics:
            current_value = metrics[self.monitor]
            
            is_better = (
                (self.mode == 'min' and current_value < self.best_value) or
                (self.mode == 'max' and current_value > self.best_value)
            )
            
            if is_better:
                self.best_value = current_value
                path = self.checkpoint_dir / "best_model.pt"
                trainer.save_checkpoint(path)
                logger.info(
                    f"Saved best model",
                    metric=self.monitor,
                    value=current_value
                )


class EarlyStoppingCallback(Callback):
    """Callback for early stopping based on validation metrics."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        monitor: str = 'val_loss',
        mode: str = 'min'
    ):
        """Initialize early stopping callback.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor
            mode: 'min' or 'max' for monitored metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.patience_counter = 0
        self.trainer = None
        
    def on_train_begin(self, trainer: Any) -> None:
        """Store trainer reference."""
        self.trainer = trainer
        
    def on_epoch_end(
        self,
        trainer: Any,
        model: EnergyBasedModel,
        metrics: Dict[str, float]
    ) -> None:
        """Check for improvement."""
        if self.monitor not in metrics:
            return
            
        current_value = metrics[self.monitor]
        
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
            
        if improved:
            self.best_value = current_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.patience:
            logger.info(
                f"Early stopping triggered",
                monitor=self.monitor,
                patience=self.patience,
                best_value=self.best_value
            )
            trainer.callbacks.stop_training()


class VisualizationCallback(Callback):
    """Callback for visualizing model during training."""
    
    def __init__(
        self,
        visualize_every: int = 10,
        num_samples: int = 64,
        save_dir: Optional[Path] = None
    ):
        """Initialize visualization callback.
        
        Args:
            visualize_every: Visualization frequency (epochs)
            num_samples: Number of samples to generate
            save_dir: Directory to save visualizations
        """
        self.visualize_every = visualize_every
        self.num_samples = num_samples
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
    def on_epoch_end(
        self,
        trainer: Any,
        model: EnergyBasedModel,
        metrics: Dict[str, float]
    ) -> None:
        """Generate and save visualizations."""
        if trainer.current_epoch % self.visualize_every != 0:
            return
            
        # Import here to avoid circular imports
        from ..utils.visualization import visualize_samples, visualize_filters
        
        with torch.no_grad():
            # Generate samples
            if hasattr(model, 'sample_fantasy_particles'):
                samples = model.sample_fantasy_particles(
                    num_samples=self.num_samples,
                    num_steps=1000
                )
                
                if self.save_dir:
                    path = self.save_dir / f"samples_epoch_{trainer.current_epoch}.png"
                    visualize_samples(samples, save_path=path)
                    
            # Visualize filters for RBMs
            if hasattr(model, 'W'):
                if self.save_dir:
                    path = self.save_dir / f"filters_epoch_{trainer.current_epoch}.png"
                    visualize_filters(model.W, save_path=path)


class LearningRateSchedulerCallback(Callback):
    """Callback for custom learning rate scheduling."""
    
    def __init__(
        self,
        schedule_fn: callable,
        update_every: str = 'epoch'
    ):
        """Initialize LR scheduler callback.
        
        Args:
            schedule_fn: Function that takes (epoch, step) and returns lr
            update_every: 'epoch' or 'step'
        """
        self.schedule_fn = schedule_fn
        self.update_every = update_every
        
    def on_epoch_start(self, trainer: Any, model: EnergyBasedModel) -> None:
        """Update learning rate at epoch start."""
        if self.update_every == 'epoch':
            lr = self.schedule_fn(trainer.current_epoch, trainer.global_step)
            self._update_lr(trainer, lr)
            
    def on_batch_start(
        self,
        trainer: Any,
        model: EnergyBasedModel,
        batch: Tensor
    ) -> None:
        """Update learning rate at batch start."""
        if self.update_every == 'step':
            lr = self.schedule_fn(trainer.current_epoch, trainer.global_step)
            self._update_lr(trainer, lr)
            
    def _update_lr(self, trainer: Any, lr: float) -> None:
        """Update optimizer learning rate."""
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = lr


class WarmupCallback(Callback):
    """Callback for learning rate warmup."""
    
    def __init__(
        self,
        warmup_steps: int,
        start_lr: float = 1e-6,
        end_lr: float = 1e-3
    ):
        """Initialize warmup callback.
        
        Args:
            warmup_steps: Number of warmup steps
            start_lr: Starting learning rate
            end_lr: Target learning rate after warmup
        """
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        
    def on_batch_start(
        self,
        trainer: Any,
        model: EnergyBasedModel,
        batch: Tensor
    ) -> None:
        """Update learning rate during warmup."""
        if trainer.global_step < self.warmup_steps:
            # Linear warmup
            progress = trainer.global_step / self.warmup_steps
            lr = self.start_lr + progress * (self.end_lr - self.start_lr)
            
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = lr