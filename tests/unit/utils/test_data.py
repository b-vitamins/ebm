"""Unit tests for data utilities."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ebm.utils.data import (
    AddNoise,
    BinaryTransform,
    DequantizeTransform,
    EnergyDataset,
    SyntheticDataset,
    compute_data_statistics,
    create_data_loaders,
    get_fashion_mnist_datasets,
    get_mnist_datasets,
)


class TestBinaryTransform:
    """Test BinaryTransform class."""

    def test_threshold_method(self) -> None:
        """Test threshold binarization."""
        transform = BinaryTransform(method='threshold', threshold=0.5)

        # Test with known values
        x = torch.tensor([0.3, 0.5, 0.7, 0.0, 1.0])
        result = transform(x)
        expected = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0])
        assert torch.equal(result, expected)

        # Test with 2D tensor
        x = torch.rand(10, 20)
        result = transform(x)
        assert result.shape == x.shape
        assert torch.all((result == 0) | (result == 1))
        assert result.dtype == torch.float32

    def test_bernoulli_method(self) -> None:
        """Test Bernoulli binarization."""
        transform = BinaryTransform(method='bernoulli')

        # Test with probabilities
        torch.manual_seed(42)
        x = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0])
        result = transform(x)

        # Check binary values
        assert torch.all((result == 0) | (result == 1))

        # Test statistical properties with larger sample
        x = torch.full((10000,), 0.7)
        results = transform(x)
        empirical_prob = results.mean()
        assert abs(empirical_prob - 0.7) < 0.02

    def test_median_method(self) -> None:
        """Test median binarization."""
        transform = BinaryTransform(method='median')

        # Test with known values
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = transform(x)
        # Median is 3.0, so values > 3.0 become 1
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
        assert torch.equal(result, expected)

        # Test that roughly half values are 1
        x = torch.randn(1000)
        result = transform(x)
        proportion_ones = result.mean()
        assert 0.45 < proportion_ones < 0.55

    def test_invalid_method(self) -> None:
        """Test error on invalid method."""
        with pytest.raises(ValueError, match="Unknown binarization method"):
            transform = BinaryTransform(method='invalid')
            transform(torch.randn(10))


class TestAddNoise:
    """Test AddNoise transform."""

    def test_gaussian_noise(self) -> None:
        """Test Gaussian noise addition."""
        transform = AddNoise(noise_type='gaussian', noise_level=0.1, clip=True)

        # Test with zeros
        x = torch.zeros(1000, 10)
        result = transform(x)

        # Should have mean close to 0, std close to noise_level
        assert abs(result.mean()) < 0.01
        assert abs(result.std() - 0.1) < 0.01

        # Test clipping
        assert result.min() >= 0.0
        assert result.max() <= 1.0

        # Test without clipping
        transform_no_clip = AddNoise(noise_type='gaussian', noise_level=0.5, clip=False)
        x = torch.full((100,), 0.5)
        result = transform_no_clip(x)
        # Some values should be outside [0, 1]
        assert (result < 0).any() or (result > 1).any()

    def test_uniform_noise(self) -> None:
        """Test uniform noise addition."""
        transform = AddNoise(noise_type='uniform', noise_level=0.2, clip=True)

        # Test with constant value
        x = torch.full((1000,), 0.5)
        result = transform(x)

        # Should be in range [0.3, 0.7] before clipping
        assert result.min() >= 0.3 - 0.01
        assert result.max() <= 0.7 + 0.01

        # Test distribution
        assert abs(result.mean() - 0.5) < 0.02

    def test_salt_pepper_noise(self) -> None:
        """Test salt and pepper noise."""
        transform = AddNoise(noise_type='salt_pepper', noise_level=0.1)

        # Test with mid-range values
        torch.manual_seed(42)
        x = torch.full((10000,), 0.5)
        result = transform(x)

        # About 10% should be 0 or 1
        corrupted = ((result == 0) | (result == 1)).float().mean()
        assert abs(corrupted - 0.1) < 0.02

        # Uncorrupted values should remain 0.5
        unchanged = result[~((result == 0) | (result == 1))]
        if len(unchanged) > 0:
            assert torch.all(unchanged == 0.5)

    def test_invalid_noise_type(self) -> None:
        """Test error on invalid noise type."""
        transform = AddNoise(noise_type='invalid')

        with pytest.raises(ValueError, match="Unknown noise type"):
            transform(torch.randn(10))


class TestDequantizeTransform:
    """Test DequantizeTransform."""

    def test_dequantization(self) -> None:
        """Test basic dequantization."""
        transform = DequantizeTransform(levels=256, scale=1.0)

        # Test with quantized values
        x = torch.tensor([0.0, 0.5, 1.0])

        # Apply multiple times to check randomness
        results = []
        for _ in range(100):
            result = transform(x)
            results.append(result)

        results = torch.stack(results)

        # Check that noise is added
        assert results.std(dim=0).min() > 0

        # Check noise scale
        noise_scale = 1.0 / 256
        assert results.std(dim=0).max() < 2 * noise_scale

        # Mean should be close to original
        assert torch.allclose(results.mean(dim=0), x, atol=noise_scale)

    def test_different_scales(self) -> None:
        """Test with different noise scales."""
        # Small scale
        transform_small = DequantizeTransform(levels=256, scale=0.5)
        x = torch.full((1000,), 0.5)
        result = transform_small(x)

        # Noise should be smaller
        noise = result - x
        assert noise.std() < 1.0 / 256

        # Large scale
        transform_large = DequantizeTransform(levels=256, scale=2.0)
        result = transform_large(x)
        noise = result - x
        assert noise.std() > 1.0 / 256


class TestEnergyDataset:
    """Test EnergyDataset wrapper."""

    def test_basic_wrapping(self) -> None:
        """Test basic dataset wrapping."""
        # Create base dataset
        data = torch.randn(100, 10)
        base_dataset = TensorDataset(data)

        # Wrap it
        energy_dataset = EnergyDataset(base_dataset)

        assert len(energy_dataset) == 100

        # Get item
        item = energy_dataset[0]
        assert torch.equal(item, data[0])

    def test_with_transform(self) -> None:
        """Test with additional transform."""
        data = torch.rand(50, 10)
        base_dataset = TensorDataset(data)

        # Add binarization transform
        transform = BinaryTransform(method='threshold')
        energy_dataset = EnergyDataset(base_dataset, transform=transform)

        # Check transform is applied
        item = energy_dataset[0]
        assert torch.all((item == 0) | (item == 1))

    def test_with_energy_computation(self) -> None:
        """Test energy computation."""
        data = torch.randn(20, 10)
        base_dataset = TensorDataset(data)

        # Mock model
        mock_model = Mock()
        mock_model.free_energy.return_value = torch.tensor([1.5])

        energy_dataset = EnergyDataset(base_dataset, model=mock_model)

        # Get item with energy
        item, energy = energy_dataset[0]

        assert torch.equal(item, data[0])
        assert energy == 1.5

        # Check model was called correctly
        mock_model.free_energy.assert_called_once()
        call_args = mock_model.free_energy.call_args[0][0]
        assert torch.equal(call_args[0], data[0])

    def test_with_labeled_dataset(self) -> None:
        """Test with dataset that returns (data, label) tuples."""
        data = torch.randn(30, 10)
        labels = torch.randint(0, 2, (30,))
        base_dataset = TensorDataset(data, labels)

        energy_dataset = EnergyDataset(base_dataset)

        # Should extract just data
        item = energy_dataset[0]
        assert torch.equal(item, data[0])

        # With model
        mock_model = Mock()
        mock_model.free_energy.return_value = torch.tensor([2.0])

        energy_dataset_model = EnergyDataset(base_dataset, model=mock_model)
        item, energy = energy_dataset_model[0]

        assert torch.equal(item, data[0])
        assert energy == 2.0


class TestSyntheticDataset:
    """Test SyntheticDataset."""

    def test_random_pattern(self) -> None:
        """Test random binary data generation."""
        dataset = SyntheticDataset(
            n_samples=1000,
            n_features=100,
            pattern='random',
            sparsity=0.3,
            seed=42
        )

        assert len(dataset) == 1000

        # Check data properties
        data = dataset.data
        assert data.shape == (1000, 100)
        assert torch.all((data == 0) | (data == 1))

        # Check sparsity
        actual_sparsity = data.mean().item()
        assert abs(actual_sparsity - 0.3) < 0.05

        # Test __getitem__
        item = dataset[0]
        assert torch.equal(item, data[0])

    def test_structured_pattern(self) -> None:
        """Test structured data generation."""
        dataset = SyntheticDataset(
            n_samples=100,
            n_features=100,
            pattern='structured',
            seed=42
        )

        data = dataset.data

        # Should have some structure (not completely random)
        # Check that some features are correlated
        corr_matrix = torch.corrcoef(data.T)

        # Remove diagonal
        corr_matrix.fill_diagonal_(0)

        # Should have some high correlations
        max_corr = corr_matrix.abs().max().item()
        assert max_corr > 0.3  # Some structure exists

    def test_correlated_pattern(self) -> None:
        """Test correlated data generation."""
        dataset = SyntheticDataset(
            n_samples=200,
            n_features=50,
            pattern='correlated',
            seed=42
        )

        data = dataset.data
        assert data.shape == (200, 50)

        # Should be binary
        assert torch.all((data == 0) | (data == 1))

        # Check that features have correlations
        # Due to latent factor model
        corr_matrix = torch.corrcoef(data.T)
        corr_matrix.fill_diagonal_(0)

        # Should have meaningful correlations
        assert corr_matrix.abs().max() > 0.2

    def test_invalid_pattern(self) -> None:
        """Test error on invalid pattern."""
        with pytest.raises(ValueError, match="Unknown pattern"):
            SyntheticDataset(pattern='invalid')


class TestMNISTDatasets:
    """Test MNIST dataset functions."""

    @patch('torchvision.datasets.MNIST')
    def test_get_mnist_datasets(self, mock_mnist) -> None:
        """Test MNIST dataset loading."""
        # Mock dataset
        mock_data = torch.randn(100, 1, 28, 28)
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset.__getitem__.side_effect = lambda i: (mock_data[i], i % 10)
        mock_mnist.return_value = mock_dataset

        # Get datasets
        train, val, test = get_mnist_datasets(
            data_dir='./data',
            binary=True,
            flatten=True,
            train_val_split=0.8,
            download=False
        )

        # Check calls
        assert mock_mnist.call_count == 2  # train and test

        # Check splits
        # Note: actual split done by random_split, so we check the mock was set up
        assert mock_mnist.call_args_list[0][1]['train'] is True
        assert mock_mnist.call_args_list[1][1]['train'] is False

        # Check transforms were set
        transform = mock_mnist.call_args_list[0][1]['transform']
        assert transform is not None

    @patch('torchvision.datasets.FashionMNIST')
    def test_get_fashion_mnist_datasets(self, mock_fashion) -> None:
        """Test Fashion-MNIST dataset loading."""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 50
        mock_fashion.return_value = mock_dataset

        # Get datasets
        train, val, test = get_fashion_mnist_datasets(
            data_dir='./data',
            binary=False,
            flatten=False,
            download=True
        )

        # Check calls
        assert mock_fashion.call_count == 2

        # Check download flag passed
        assert mock_fashion.call_args_list[0][1]['download'] is True


class TestDataLoaderCreation:
    """Test data loader creation."""

    def test_create_data_loaders(self) -> None:
        """Test creating data loaders from datasets."""
        # Create datasets
        train_data = torch.randn(100, 10)
        val_data = torch.randn(20, 10)
        test_data = torch.randn(30, 10)

        train_dataset = TensorDataset(train_data)
        val_dataset = TensorDataset(val_data)
        test_dataset = TensorDataset(test_data)

        # Create loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=16,
            num_workers=0,
            pin_memory=False,
            shuffle_train=True
        )

        # Check train loader
        assert isinstance(train_loader, DataLoader)
        assert train_loader.batch_size == 16
        assert len(train_loader) == 7  # ceil(100/16)

        # Check val loader
        assert isinstance(val_loader, DataLoader)
        assert val_loader.batch_size == 16
        assert len(val_loader) == 2  # ceil(20/16)

        # Check test loader
        assert isinstance(test_loader, DataLoader)
        assert test_loader.batch_size == 16
        assert len(test_loader) == 2  # ceil(30/16)

    def test_create_data_loaders_minimal(self) -> None:
        """Test creating only train loader."""
        train_dataset = TensorDataset(torch.randn(50, 10))

        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset=train_dataset,
            batch_size=32
        )

        assert train_loader is not None
        assert val_loader is None
        assert test_loader is None


class TestDataStatistics:
    """Test data statistics computation."""

    def test_compute_data_statistics(self) -> None:
        """Test computing dataset statistics."""
        # Create known data
        data = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ])
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=2)

        stats = compute_data_statistics(loader)

        # Check computed statistics
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'n_samples' in stats

        # Check values
        expected_mean = torch.tensor([5.5, 6.5, 7.5])
        assert torch.allclose(stats['mean'], expected_mean)

        expected_min = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(stats['min'], expected_min)

        expected_max = torch.tensor([10.0, 11.0, 12.0])
        assert torch.allclose(stats['max'], expected_max)

        assert stats['n_samples'] == 4

    def test_compute_data_statistics_with_labels(self) -> None:
        """Test statistics computation with labeled data."""
        data = torch.randn(50, 20)
        labels = torch.randint(0, 10, (50,))
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=10)

        stats = compute_data_statistics(loader)

        # Should compute stats only for data, not labels
        assert stats['mean'].shape == (20,)
        assert stats['n_samples'] == 50

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_compute_data_statistics_cuda(self) -> None:
        """Test statistics computation on CUDA."""
        data = torch.randn(30, 10)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=10)

        device = torch.device("cuda")
        stats = compute_data_statistics(loader, device=device)

        # Results should be on CPU (moved back)
        assert stats['mean'].device.type == "cpu"
        assert stats['std'].device.type == "cpu"


class TestEdgeCases:
    """Test edge cases for data utilities."""

    def test_empty_dataset(self) -> None:
        """Test handling of empty datasets."""
        dataset = SyntheticDataset(n_samples=0, n_features=10)
        assert len(dataset) == 0
        assert dataset.data.shape == (0, 10)

        # Create loader
        loader = DataLoader(dataset, batch_size=32)
        assert len(loader) == 0

    def test_single_sample_dataset(self) -> None:
        """Test dataset with single sample."""
        dataset = SyntheticDataset(n_samples=1, n_features=10)
        assert len(dataset) == 1

        item = dataset[0]
        assert item.shape == (10,)

    def test_transform_chaining(self) -> None:
        """Test chaining multiple transforms."""
        # Create transform pipeline
        binarize = BinaryTransform(method='threshold')
        add_noise = AddNoise(noise_type='gaussian', noise_level=0.1)

        # Manual chaining
        def chained_transform(x):
            x = binarize(x)
            return add_noise(x)

        # Test
        x = torch.rand(100, 20)
        result = chained_transform(x)

        # Should be mostly binary with some noise
        assert result.min() >= -0.5  # Some negative due to noise
        assert result.max() <= 1.5   # Some above 1 due to noise
