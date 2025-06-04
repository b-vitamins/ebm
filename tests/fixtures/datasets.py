"""Dataset fixtures for testing."""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def synthetic_binary_data() -> dict[str, object]:
    """Provide synthetic binary data for testing."""
    n_samples = 1000
    n_features = 100

    # Create data with some structure
    torch.manual_seed(42)

    # Generate correlated binary data
    n_factors = 10
    weight_matrix = torch.randn(n_factors, n_features) * 0.5
    factors = torch.randn(n_samples, n_factors)
    logits = factors @ weight_matrix
    data = torch.bernoulli(torch.sigmoid(logits))

    dataset = TensorDataset(data)

    return {
        "data": data,
        "dataset": dataset,
        "n_samples": n_samples,
        "n_features": n_features,
        "sparsity": data.mean().item(),
    }


@pytest.fixture
def synthetic_continuous_data() -> dict[str, object]:
    """Provide synthetic continuous data for testing."""
    n_samples = 1000
    n_features = 50

    torch.manual_seed(42)

    # Generate data from a mixture of Gaussians
    weights = torch.tensor([0.3, 0.5, 0.2])

    data = []
    for _i in range(n_samples):
        # Sample component
        component = torch.multinomial(weights, 1).item()

        # Sample from component
        mean = torch.randn(n_features) * component
        std = 0.5 + 0.2 * component
        sample = mean + torch.randn(n_features) * std
        data.append(sample)

    data = torch.stack(data)

    # Normalize to [0, 1]
    data = (data - data.min()) / (data.max() - data.min())

    dataset = TensorDataset(data)

    return {
        "data": data,
        "dataset": dataset,
        "n_samples": n_samples,
        "n_features": n_features,
        "mean": data.mean(dim=0),
        "std": data.std(dim=0),
    }


@pytest.fixture
def mini_mnist_dataset() -> dict[str, object]:
    """Provide a mini MNIST-like dataset for testing."""
    n_samples = 500
    image_size = 28
    n_features = image_size * image_size

    torch.manual_seed(42)

    # Generate MNIST-like patterns
    images = []

    for i in range(n_samples):
        # Create a blank image
        img = torch.zeros(image_size, image_size)

        # Add a random digit-like pattern
        pattern_type = i % 10  # 10 different patterns

        if pattern_type == 0:  # Vertical line
            col = torch.randint(5, 23, (1,)).item()
            img[:, col : col + 3] = 1
        elif pattern_type == 1:  # Horizontal line
            row = torch.randint(5, 23, (1,)).item()
            img[row : row + 3, :] = 1
        elif pattern_type == 2:  # Square
            r, c = torch.randint(5, 20, (2,))
            img[r : r + 8, c : c + 8] = 1
        elif pattern_type == 3:  # Circle (approximation)
            center = torch.tensor([14, 14])
            y, x = torch.meshgrid(
                torch.arange(28), torch.arange(28), indexing="ij"
            )
            dist = ((x - center[0]) ** 2 + (y - center[1]) ** 2).sqrt()
            img[dist < 8] = 1
        else:  # Random patterns
            img = torch.rand(image_size, image_size)
            img = (img > 0.7).float()

        # Add noise
        noise = torch.rand(image_size, image_size)
        img = torch.where(noise < 0.1, 1 - img, img)

        images.append(img.flatten())

    data = torch.stack(images)
    dataset = TensorDataset(data)

    return {
        "data": data,
        "dataset": dataset,
        "n_samples": n_samples,
        "n_features": n_features,
        "image_size": image_size,
    }


@pytest.fixture
def make_structured_data() -> Callable[[int, int, str, float], TensorDataset]:
    """Create structured test data."""

    def _make_structured_data(
        n_samples: int = 100,
        n_features: int = 50,
        structure_type: str = "bars",
        noise_level: float = 0.1,
    ) -> TensorDataset:
        """Create structured data with specific patterns."""
        torch.manual_seed(42)

        if structure_type == "bars":
            # Horizontal and vertical bars
            data = torch.zeros(n_samples, n_features)
            bar_width = max(1, n_features // 10)

            for i in range(n_samples):
                if i % 2 == 0:  # Horizontal bar
                    pos = torch.randint(0, n_features - bar_width, (1,)).item()
                    data[i, pos : pos + bar_width] = 1
                else:  # Vertical bar pattern (if 2D interpretation)
                    # Assuming square layout
                    side = int(np.sqrt(n_features))
                    if side * side == n_features:
                        img = torch.zeros(side, side)
                        col = torch.randint(0, side - 1, (1,)).item()
                        img[:, col] = 1
                        data[i] = img.flatten()
                    else:
                        # Fallback to random pattern
                        pos = torch.randint(0, n_features, (bar_width,))
                        data[i, pos] = 1

        elif structure_type == "clusters":
            # Clustered patterns
            n_clusters = 5
            cluster_centers = torch.rand(n_clusters, n_features)

            for i in range(n_samples):
                cluster = i % n_clusters
                center = cluster_centers[cluster]
                data[i] = (torch.rand(n_features) < center).float()

        elif structure_type == "sparse":
            # Sparse random patterns
            sparsity = 0.1
            data = (torch.rand(n_samples, n_features) < sparsity).float()

        else:
            # Random data
            data = torch.rand(n_samples, n_features)
            data = (data > 0.5).float()

        # Add noise
        if noise_level > 0:
            noise_mask = torch.rand_like(data) < noise_level
            data = torch.where(noise_mask, 1 - data, data)

        return TensorDataset(data)

    return _make_structured_data


@pytest.fixture
def make_data_loader() -> Callable[[TensorDataset, int, bool, int], DataLoader]:
    """Create simple data loaders."""

    def _make_data_loader(
        dataset: TensorDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """Create a data loader from a dataset."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
        )

    return _make_data_loader


@pytest.fixture
def data_statistics() -> Callable[[torch.Tensor], dict[str, object]]:
    """Fixture providing data statistics computation."""

    def _compute_statistics(data: torch.Tensor) -> dict[str, object]:
        """Compute comprehensive statistics for data."""
        return {
            "mean": data.mean().item(),
            "std": data.std().item(),
            "min": data.min().item(),
            "max": data.max().item(),
            "sparsity": (data > 0).float().mean().item(),
            "feature_means": data.mean(dim=0),
            "feature_stds": data.std(dim=0),
            "sample_means": data.mean(dim=1),
            "sample_stds": data.std(dim=1),
            "correlation_matrix": torch.corrcoef(data.T)
            if data.shape[1] < 100
            else None,
        }

    return _compute_statistics


@pytest.fixture
def simple_data_loader() -> DataLoader:
    """Provide a simple data loader used in multiple tests."""
    data = torch.randn(100, 10)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
