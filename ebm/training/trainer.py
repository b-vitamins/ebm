"""Training orchestration for energy-based models.

This module provides a flexible trainer class that handles the training
loop, optimization, logging, and callback management.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..core.config import TrainingConfig
from ..core.device import DeviceManager
from ..core.logging import LoggerMixin, log_context
from ..models.base import EnergyBasedModel
from ..sampling.base import GradientEstimator
from .callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    MetricsCallback,
)
from .metrics import MetricsTracker


class Trainer(LoggerMixin):
    """Trainer for energy-based models.

    This class orchestrates the training process, handling:
    - Training loops and optimization
    - Gradient estimation via sampling
    - Metrics tracking and logging
    - Checkpointing and early stopping
    - Callback management
    """

    def __init__(
        self,
        model: EnergyBasedModel,
        config: TrainingConfig,
        gradient_estimator: GradientEstimator,
        callbacks: list[Callback] | None = None
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            config: Training configuration
            gradient_estimator: Method for estimating gradients
            callbacks: Optional list of callbacks
        """
        super().__init__()
        self.model = model
        self.config = config
        self.gradient_estimator = gradient_estimator

        # Setup device manager
        self.device_manager = DeviceManager(model.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Setup metrics
        self.metrics = MetricsTracker()

        # Setup callbacks
        self.callbacks = self._setup_callbacks(callbacks)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')

        # Setup mixed precision if requested
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

        # Compile model if requested (PyTorch 2.0+)
        if config.compile_model and hasattr(torch, 'compile'):
            self.log_info("Compiling model with torch.compile")
            self.model = torch.compile(self.model)

    def _create_optimizer(self) -> Optimizer:
        """Create optimizer from configuration."""
        opt_config = self.config.optimizer
        opt_class = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop,
            'lbfgs': torch.optim.LBFGS,
        }[opt_config.name]

        # Build optimizer arguments
        opt_args = {
            'lr': opt_config.lr,
            'weight_decay': opt_config.weight_decay,
        }

        if opt_config.name in ['adam', 'adamw']:
            opt_args.update({
                'betas': opt_config.betas,
                'eps': opt_config.eps,
            })
        elif opt_config.name == 'sgd':
            opt_args.update({
                'momentum': opt_config.momentum,
                'nesterov': opt_config.nesterov,
            })

        return opt_class(self.model.parameters(), **opt_args)

    def _create_scheduler(self) -> object | None:
        """Create learning rate scheduler if configured."""
        if not self.config.optimizer.scheduler:
            return None

        scheduler_type = self.config.optimizer.scheduler
        scheduler_params = self.config.optimizer.scheduler_params

        if scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_params.get('step_size', 10),
                gamma=scheduler_params.get('gamma', 0.1)
            )
        elif scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=scheduler_params.get('eta_min', 0)
            )
        elif scheduler_type == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_params.get('factor', 0.1),
                patience=scheduler_params.get('patience', 10)
            )
        else:
            self.log_warning(f"Unknown scheduler type: {scheduler_type}")
            return None

    def _setup_callbacks(self, callbacks: list[Callback] | None) -> CallbackList:
        """Setup default and user callbacks."""
        default_callbacks = [
            MetricsCallback(),
            LoggingCallback(log_every=self.config.log_every),
        ]

        # Add checkpoint callback
        if self.config.checkpoint_every > 0:
            default_callbacks.append(
                CheckpointCallback(
                    checkpoint_dir=self.config.checkpoint_dir,
                    save_every=self.config.checkpoint_every
                )
            )

        # Add early stopping
        if self.config.early_stopping:
            default_callbacks.append(
                EarlyStoppingCallback(
                    patience=self.config.patience,
                    min_delta=self.config.min_delta
                )
            )

        # Combine with user callbacks
        all_callbacks = default_callbacks + (callbacks or [])

        return CallbackList(all_callbacks)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        num_epochs: int | None = None
    ) -> dict[str, Any]:
        """Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs (overrides config)

        Returns:
            Training history and final metrics
        """
        num_epochs = num_epochs or self.config.epochs

        self.log_info(
            "Starting training",
            epochs=num_epochs,
            train_batches=len(train_loader),
            val_batches=len(val_loader) if val_loader else 0,
            device=str(self.model.device),
            optimizer=self.config.optimizer.name
        )

        # Initialize model from data if needed
        if hasattr(self.model, 'init_from_data'):
            self.log_info("Initializing model from data statistics")
            self.model.init_from_data(train_loader)

        # Training loop
        history = {'train': [], 'val': []}

        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch

                # Train epoch
                with log_context(epoch=epoch, phase='train'):
                    train_metrics = self._train_epoch(train_loader)
                    history['train'].append(train_metrics)

                # Validation
                if val_loader and (epoch + 1) % self.config.eval_every == 0:
                    with log_context(epoch=epoch, phase='validation'):
                        val_metrics = self._validate(val_loader)
                        history['val'].append(val_metrics)
                else:
                    val_metrics = {}

                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        metric = val_metrics.get('loss', train_metrics.get('loss', 0))
                        self.scheduler.step(metric)
                    else:
                        self.scheduler.step()

                # Check early stopping
                if self.callbacks.should_stop:
                    self.log_info(f"Early stopping at epoch {epoch}")
                    break

        except KeyboardInterrupt:
            self.log_warning("Training interrupted by user")

        # Final logging
        self.log_info(
            "Training completed",
            final_epoch=self.current_epoch,
            total_steps=self.global_step,
            best_metric=self.best_metric
        )

        return {
            'history': history,
            'final_metrics': history['train'][-1] if history['train'] else {},
            'best_metric': self.best_metric,
        }

    def _train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """Train one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Epoch metrics
        """
        self.model.train()
        self.metrics.reset()

        # Callback
        self.callbacks.on_epoch_start(self, self.model)

        # Progress bar
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch}",
            disable=not self.config.get('show_progress', True)
        )

        epoch_start = time.time()

        for _batch_idx, batch in enumerate(pbar):
            # Handle different batch formats
            if isinstance(batch, list | tuple):
                data = batch[0]
            else:
                data = batch

            # Move to device
            data = self.device_manager.to_device(data)

            # Callback
            self.callbacks.on_batch_start(self, self.model, data)

            # Training step
            with self.device_manager.autocast(self.config.mixed_precision):
                loss, batch_metrics = self._training_step(data)

            # Update metrics
            self.metrics.update(batch_metrics)

            # Update progress bar
            pbar.set_postfix(self.metrics.get_current())

            # Callback
            self.callbacks.on_batch_end(self, self.model, loss)

            self.global_step += 1

        # Compute epoch metrics
        epoch_metrics = self.metrics.compute()
        epoch_metrics['epoch_time'] = time.time() - epoch_start
        epoch_metrics['lr'] = self.optimizer.param_groups[0]['lr']

        # Callback
        self.callbacks.on_epoch_end(self, self.model, epoch_metrics)

        return epoch_metrics

    def _training_step(self, data: torch.Tensor) -> tuple[float, dict[str, float]]:
        """Single training step.

        Args:
            data: Batch of training data

        Returns:
            Loss value and metrics dictionary
        """
        # Zero gradients
        self.optimizer.zero_grad()

        # Estimate gradients
        gradients = self.gradient_estimator.estimate_gradient(
            self.model, data
        )

        # Convert to standard PyTorch gradients
        for name, param in self.model.named_parameters():
            if name in gradients:
                param.grad = -gradients[name]  # Negative for gradient ascent

        # Gradient clipping
        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )

        # Optimization step
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Compute metrics
        metrics = self.gradient_estimator.compute_metrics(
            self.model,
            data,
            self.gradient_estimator.last_negative_samples
        )

        # Add gradient norms
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        metrics['grad_norm'] = total_norm ** 0.5

        # Pseudo-loss for tracking
        loss = metrics.get('energy_gap', 0.0)

        return loss, metrics

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> dict[str, float]:
        """Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Validation metrics
        """
        self.model.eval()
        val_metrics = MetricsTracker()

        for batch in tqdm(val_loader, desc="Validation", leave=False):
            if isinstance(batch, list | tuple):
                data = batch[0]
            else:
                data = batch

            data = self.device_manager.to_device(data)

            # Compute validation metrics
            # For RBMs, we typically look at reconstruction error
            if hasattr(self.model, 'reconstruct'):
                recon = self.model.reconstruct(data)
                recon_error = (data - recon).pow(2).mean()

                metrics = {
                    'val_reconstruction_error': recon_error.item(),
                    'val_free_energy': self.model.free_energy(data).mean().item(),
                }
            else:
                metrics = {
                    'val_free_energy': self.model.free_energy(data).mean().item(),
                }

            val_metrics.update(metrics)

        return val_metrics.compute()

    def save_checkpoint(self, path: Path | None = None) -> Path:
        """Save training checkpoint.

        Args:
            path: Checkpoint path (auto-generated if None)

        Returns:
            Path where checkpoint was saved
        """
        if path is None:
            path = self.config.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config.dict(),
            'gradient_estimator': self.gradient_estimator.__class__.__name__,
        }

        torch.save(checkpoint, path)
        self.log_info(f"Saved checkpoint to {path}")

        return path

    def load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.model.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint.get('best_metric', float('inf'))

        self.log_info(
            f"Loaded checkpoint from {path}",
            epoch=self.current_epoch,
            step=self.global_step
        )
