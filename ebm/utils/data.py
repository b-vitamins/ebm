"""Data utilities for energy-based models.

This module provides utilities for data loading, preprocessing,
and dataset creation for training EBMs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from ebm.core.logging import logger


class BinaryTransform:
    """Transform to binarize data using various methods."""

    def __init__(self, method: str = "threshold", threshold: float = 0.5):
        """Initialize binary transform.

        Args:
            method: Binarization method ('threshold', 'bernoulli', 'median')
            threshold: Threshold value for 'threshold' method
        """
        self.method = method
        self.threshold = threshold

    def __call__(self, x: Tensor) -> Tensor:
        """Apply binarization.

        Args:
            x: Input tensor

        Returns
        -------
            Binarized tensor
        """
        if self.method == "threshold":
            return (x > self.threshold).float()
        if self.method == "bernoulli":
            return torch.bernoulli(x)
        if self.method == "median":
            return (x > x.median()).float()
        raise ValueError(f"Unknown binarization method: {self.method}")


class AddNoise:
    """Transform to add noise to data."""

    def __init__(
        self,
        noise_type: str = "gaussian",
        noise_level: float = 0.1,
        clip: bool = True,
    ):
        """Initialize noise transform.

        Args:
            noise_type: Type of noise ('gaussian', 'uniform', 'salt_pepper')
            noise_level: Noise strength
            clip: Whether to clip output to [0, 1]
        """
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.clip = clip

    def __call__(self, x: Tensor) -> Tensor:
        """Add noise to input.

        Args:
            x: Input tensor

        Returns
        -------
            Noisy tensor
        """
        if self.noise_type == "gaussian":
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
        elif self.noise_type == "uniform":
            noise = (torch.rand_like(x) - 0.5) * 2 * self.noise_level
            x = x + noise
        elif self.noise_type == "salt_pepper":
            mask = torch.rand_like(x) < self.noise_level
            salt_pepper = torch.randint_like(x, 0, 2).float()
            x = torch.where(mask, salt_pepper, x)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

        if self.clip:
            x = torch.clamp(x, 0, 1)

        return x


class DequantizeTransform:
    """Transform to dequantize discrete data."""

    def __init__(self, levels: int = 256, scale: float = 1.0):
        """Initialize dequantization transform.

        Args:
            levels: Number of quantization levels
            scale: Scale factor for dequantization noise
        """
        self.levels = levels
        self.scale = scale

    def __call__(self, x: Tensor) -> Tensor:
        """Apply dequantization.

        Args:
            x: Input tensor (assumed to be in [0, 1])

        Returns
        -------
            Dequantized tensor
        """
        # Add uniform noise
        noise = torch.rand_like(x) * self.scale / self.levels
        return x + noise


class EnergyDataset(Dataset):
    """Dataset wrapper that adds energy computation functionality."""

    def __init__(
        self,
        base_dataset: Dataset,
        model: Any | None = None,
        transform: Callable | None = None,
    ):
        """Initialize energy dataset.

        Args:
            base_dataset: Base dataset
            model: Energy model for computing energies
            transform: Additional transform to apply
        """
        self.base_dataset = base_dataset
        self.model = model
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the base dataset."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tensor | tuple[Tensor, float]:
        """Get item with optional energy computation.

        Args:
            idx: Item index

        Returns
        -------
            Data tensor or (data, energy) tuple
        """
        # Get base item
        item = self.base_dataset[idx]

        # Handle (data, label) format
        data = item[0] if isinstance(item, tuple) else item

        # Apply transform
        if self.transform is not None:
            data = self.transform(data)

        # Compute energy if model is provided
        if self.model is not None:
            with torch.no_grad():
                energy = self.model.free_energy(data.unsqueeze(0)).squeeze()
            return data, energy.item()

        return data


def get_mnist_datasets(
    data_dir: str | Path = "./data",
    binary: bool = True,
    flatten: bool = True,
    train_val_split: float = 0.9,
    download: bool = True,
) -> tuple[Dataset, Dataset, Dataset]:
    """Get MNIST datasets for EBM training.

    Args:
        data_dir: Data directory
        binary: Whether to binarize the data
        flatten: Whether to flatten images
        train_val_split: Fraction of training data to use for training
        download: Whether to download if not present

    Returns
    -------
        (train_dataset, val_dataset, test_dataset)
    """
    data_dir = Path(data_dir)

    # Define transforms
    transform_list = [transforms.ToTensor()]

    if binary:
        transform_list.append(BinaryTransform("bernoulli"))

    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))

    transform = transforms.Compose(transform_list)

    # Load datasets
    full_train = datasets.MNIST(
        data_dir, train=True, transform=transform, download=download
    )
    test = datasets.MNIST(
        data_dir, train=False, transform=transform, download=download
    )

    # Split train/val
    n_train = int(len(full_train) * train_val_split)
    n_val = len(full_train) - n_train
    train, val = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    logger.info(
        f"Loaded MNIST: {n_train} train, {n_val} val, {len(test)} test samples"
    )

    return train, val, test


def get_fashion_mnist_datasets(
    data_dir: str | Path = "./data", **kwargs
) -> tuple[Dataset, Dataset, Dataset]:
    """Get Fashion-MNIST datasets.

    Args:
        data_dir: Data directory
        **kwargs: Arguments passed to get_mnist_datasets

    Returns
    -------
        (train_dataset, val_dataset, test_dataset)
    """
    data_dir = Path(data_dir)

    # Same transforms as MNIST
    binary = kwargs.get("binary", True)
    flatten = kwargs.get("flatten", True)
    train_val_split = kwargs.get("train_val_split", 0.9)
    download = kwargs.get("download", True)

    transform_list = [transforms.ToTensor()]
    if binary:
        transform_list.append(BinaryTransform("bernoulli"))
    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))

    transform = transforms.Compose(transform_list)

    # Load datasets
    full_train = datasets.FashionMNIST(
        data_dir, train=True, transform=transform, download=download
    )
    test = datasets.FashionMNIST(
        data_dir, train=False, transform=transform, download=download
    )

    # Split train/val
    n_train = int(len(full_train) * train_val_split)
    n_val = len(full_train) - n_train
    train, val = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    return train, val, test


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing and debugging."""

    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int = 100,
        pattern: str = "random",
        sparsity: float = 0.1,
        seed: int = 42,
    ):
        """Initialize synthetic dataset.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            pattern: Data pattern ('random', 'structured', 'correlated')
            sparsity: Sparsity level for binary data
            seed: Random seed
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.pattern = pattern
        self.sparsity = sparsity

        # Generate data
        torch.manual_seed(seed)
        self.data = self._generate_data()

    def _generate_data(self) -> Tensor:
        """Generate synthetic data based on pattern."""
        if self.pattern == "random":
            # Random binary data
            return torch.bernoulli(
                torch.full((self.n_samples, self.n_features), self.sparsity)
            )

        if self.pattern == "structured":
            # Data with structure (e.g., bars and stripes)
            data = torch.zeros(self.n_samples, self.n_features)

            # Add horizontal stripes
            for i in range(0, self.n_features, 10):
                mask = torch.rand(self.n_samples) < 0.5
                data[mask, i : i + 5] = 1

            # Add vertical patterns
            for i in range(self.n_samples):
                if torch.rand(1) < 0.3:
                    pattern_idx = torch.randperm(self.n_features)[:10]
                    data[i, pattern_idx] = 1

            return data

        if self.pattern == "correlated":
            # Data with correlations
            # Generate latent factors
            n_factors = max(10, self.n_features // 10)
            factors = torch.randn(self.n_samples, n_factors)

            # Generate mixing matrix
            mixing_matrix = torch.randn(n_factors, self.n_features) * 0.5

            # Generate data
            data = torch.sigmoid(
                factors @ mixing_matrix
                + torch.randn(self.n_samples, self.n_features) * 0.1
            )
            return torch.bernoulli(data)

        raise ValueError(f"Unknown pattern: {self.pattern}")

    def __len__(self) -> int:
        """Return total number of generated samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tensor:
        """Return a data sample by index."""
        return self.data[idx]


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    """Create data loaders from datasets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU
        shuffle_train: Whether to shuffle training data

    Returns
    -------
        (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader


def compute_data_statistics(
    data_loader: DataLoader, device: torch.device | None = None
) -> dict[str, Tensor]:
    """Compute statistics of a dataset.

    Args:
        data_loader: Data loader
        device: Device to use for computation

    Returns
    -------
        Dictionary with 'mean', 'std', 'min', 'max' statistics
    """
    if device is None:
        device = torch.device("cpu")

    # Initialize accumulators
    n_samples = 0
    sum_x = None
    sum_x2 = None
    min_x = None
    max_x = None

    for batch in data_loader:
        # Handle (data, label) format
        if isinstance(batch, list | tuple):
            batch = batch[0]

        batch = batch.to(device)
        n_samples += batch.shape[0]

        # Flatten batch
        batch_flat = batch.view(batch.shape[0], -1)

        if sum_x is None:
            sum_x = batch_flat.sum(dim=0)
            sum_x2 = (batch_flat**2).sum(dim=0)
            min_x = batch_flat.min(dim=0)[0]
            max_x = batch_flat.max(dim=0)[0]
        else:
            sum_x += batch_flat.sum(dim=0)
            sum_x2 += (batch_flat**2).sum(dim=0)
            min_x = torch.minimum(min_x, batch_flat.min(dim=0)[0])
            max_x = torch.maximum(max_x, batch_flat.max(dim=0)[0])

    # Compute statistics
    mean = sum_x / n_samples
    var = (sum_x2 / n_samples) - (mean**2)
    std = torch.sqrt(torch.clamp(var, min=0))

    return {
        "mean": mean,
        "std": std,
        "min": min_x,
        "max": max_x,
        "n_samples": n_samples,
    }
