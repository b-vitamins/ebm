"""Unit tests for tensor utilities."""

import numpy as np
import pytest
import torch
from torch import Tensor

from ebm.utils.tensor import (
    TensorStatistics,
    batch_mv,
    batch_outer_product,
    batch_quadratic_form,
    concat_tensors,
    create_causal_mask,
    create_padding_mask,
    ensure_tensor,
    expand_dims_like,
    log_sum_exp,
    masked_fill_inf,
    safe_log,
    safe_sqrt,
    shape_for_broadcast,
    split_tensor,
    stack_tensors,
)


class TestBasicUtilities:
    """Test basic tensor utility functions."""

    def test_ensure_tensor(self) -> None:
        """Test tensor conversion."""
        # From list
        result = ensure_tensor([1, 2, 3])
        assert isinstance(result, Tensor)
        assert result.tolist() == [1, 2, 3]

        # From scalar
        result = ensure_tensor(3.14)
        assert isinstance(result, Tensor)
        assert result.shape == (1,)
        assert result.item() == pytest.approx(3.14)

        # From numpy array
        np_array = np.array([1.0, 2.0, 3.0])
        result = ensure_tensor(np_array)
        assert isinstance(result, Tensor)
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

        # Already a tensor
        tensor = torch.randn(5, 5)
        result = ensure_tensor(tensor)
        assert result is tensor  # Should be same object

        # With dtype conversion
        result = ensure_tensor([1, 2, 3], dtype=torch.float32)
        assert result.dtype == torch.float32

        # With device conversion
        if torch.cuda.is_available():
            result = ensure_tensor([1, 2, 3], device="cuda")
            assert result.device.type == "cuda"

    def test_safe_log(self) -> None:
        """Test numerically stable logarithm."""
        # Normal values
        x = torch.tensor([1.0, 2.0, 3.0])
        result = safe_log(x)
        expected = torch.log(x)
        assert torch.allclose(result, expected)

        # Zero values
        x = torch.tensor([0.0, 1.0, 0.0])
        result = safe_log(x)
        assert torch.all(torch.isfinite(result))
        assert result[0] == torch.log(torch.tensor(1e-10))

        # Negative values (should add eps)
        x = torch.tensor([-1.0, -2.0])
        result = safe_log(x)
        assert torch.all(torch.isfinite(result))

        # Custom epsilon
        x = torch.tensor([0.0])
        result = safe_log(x, eps=1e-5)
        assert result[0] == torch.log(torch.tensor(1e-5))

    def test_safe_sqrt(self) -> None:
        """Test numerically stable square root."""
        # Normal values
        x = torch.tensor([1.0, 4.0, 9.0])
        result = safe_sqrt(x)
        expected = torch.sqrt(x)
        assert torch.allclose(result, expected)

        # Zero values
        x = torch.tensor([0.0, 1.0, 0.0])
        result = safe_sqrt(x)
        assert torch.all(torch.isfinite(result))
        assert result[0] == torch.sqrt(torch.tensor(1e-10))

        # Negative values (should add eps)
        x = torch.tensor([-1.0, -4.0])
        result = safe_sqrt(x)
        assert torch.all(torch.isfinite(result))

    def test_log_sum_exp(self) -> None:
        """Test log-sum-exp computation."""
        # 1D case
        x = torch.tensor([1.0, 2.0, 3.0])
        result = log_sum_exp(x)
        expected = torch.log(torch.exp(x).sum())
        assert torch.allclose(result, expected)

        # 2D case with dimension
        x = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]])

        # Sum over dim 1
        result = log_sum_exp(x, dim=1)
        expected = torch.log(torch.exp(x).sum(dim=1))
        assert torch.allclose(result, expected)
        assert result.shape == (2,)

        # Sum over dim 0 with keepdim
        result = log_sum_exp(x, dim=0, keepdim=True)
        expected = torch.log(torch.exp(x).sum(dim=0, keepdim=True))
        assert torch.allclose(result, expected)
        assert result.shape == (1, 3)

        # Test numerical stability with large values
        x = torch.tensor([100.0, 101.0, 102.0])
        result = log_sum_exp(x)
        assert torch.isfinite(result)
        assert result < 103  # Should be log(e^100 + e^101 + e^102) ≈ 102.4


class TestBatchOperations:
    """Test batch tensor operations."""

    def test_batch_outer_product(self) -> None:
        """Test batch outer product."""
        a = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0]])
        b = torch.tensor([[5.0, 6.0, 7.0],
                          [8.0, 9.0, 10.0]])

        result = batch_outer_product(a, b)

        assert result.shape == (2, 2, 3)

        # Check first batch
        expected_0 = torch.outer(a[0], b[0])
        assert torch.allclose(result[0], expected_0)

        # Check second batch
        expected_1 = torch.outer(a[1], b[1])
        assert torch.allclose(result[1], expected_1)

        # Single batch
        a_single = torch.randn(1, 10)
        b_single = torch.randn(1, 20)
        result_single = batch_outer_product(a_single, b_single)
        assert result_single.shape == (1, 10, 20)

    def test_batch_quadratic_form(self) -> None:
        """Test batch quadratic form computation."""
        # Single matrix for all batches
        x = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0]])
        A = torch.tensor([[2.0, 1.0],
                          [1.0, 3.0]])

        result = batch_quadratic_form(x, A)

        assert result.shape == (2,)

        # Check manually: x^T A x
        expected_0 = x[0] @ A @ x[0]
        expected_1 = x[1] @ A @ x[1]
        assert torch.allclose(result, torch.tensor([expected_0, expected_1]))

        # Different matrix for each batch
        x = torch.randn(5, 3)
        A = torch.randn(5, 3, 3)

        result = batch_quadratic_form(x, A)
        assert result.shape == (5,)

        # Check one example
        expected_0 = x[0] @ A[0] @ x[0]
        assert torch.allclose(result[0], expected_0)

    def test_batch_mv(self) -> None:
        """Test batched matrix-vector multiplication."""
        # 2D matrix, 1D vector
        A = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0]])
        x = torch.tensor([5.0, 6.0])

        result = batch_mv(A, x)
        expected = torch.mv(A, x)
        assert torch.allclose(result, expected)

        # 2D matrix, 2D vector (batched)
        A = torch.randn(3, 4)
        x = torch.randn(5, 4)

        result = batch_mv(A, x)
        assert result.shape == (5, 3)

        # Check one example
        expected_0 = A @ x[0]
        assert torch.allclose(result[0], expected_0)

        # 3D matrix, 1D vector
        A = torch.randn(5, 3, 4)
        x = torch.randn(4)

        result = batch_mv(A, x)
        assert result.shape == (5, 3)

        # 3D matrix, 2D vector (full batch)
        A = torch.randn(5, 3, 4)
        x = torch.randn(5, 4)

        result = batch_mv(A, x)
        assert result.shape == (5, 3)

        # Check one example
        expected_0 = A[0] @ x[0]
        assert torch.allclose(result[0], expected_0)


class TestShapeManipulation:
    """Test shape manipulation utilities."""

    def test_shape_for_broadcast(self) -> None:
        """Test reshaping for broadcasting."""
        # Basic case
        tensor = torch.randn(5)
        target_shape = (10, 5, 3)

        result = shape_for_broadcast(tensor, target_shape)
        assert result.shape == (1, 5, 1)

        # Scalar
        tensor = torch.tensor(3.14)
        result = shape_for_broadcast(tensor, (2, 3, 4))
        assert result.shape == ()  # Scalar remains scalar

        # Already correct shape
        tensor = torch.randn(2, 3, 4)
        result = shape_for_broadcast(tensor, (2, 3, 4))
        assert result is tensor  # Same object

        # With specific dimension
        tensor = torch.randn(5)
        result = shape_for_broadcast(tensor, (2, 5, 3), dim=1)
        assert result.shape == (1, 5, 1)

        # Multiple dimensions
        tensor = torch.randn(3, 4)
        result = shape_for_broadcast(tensor, (2, 3, 4, 5))
        assert result.shape == (1, 1, 3, 4)

    def test_expand_dims_like(self) -> None:
        """Test dimension expansion."""
        # Add dimensions
        tensor = torch.randn(5)
        reference = torch.randn(2, 3, 5)

        result = expand_dims_like(tensor, reference)
        assert result.shape == (5, 1, 1)

        # Same dimensions
        tensor = torch.randn(2, 3)
        reference = torch.randn(4, 5)

        result = expand_dims_like(tensor, reference)
        assert result.shape == tensor.shape

        # Multiple expansions
        tensor = torch.randn(3)
        reference = torch.randn(2, 3, 4, 5, 6)

        result = expand_dims_like(tensor, reference)
        assert result.shape == (3, 1, 1, 1, 1)


class TestMasking:
    """Test masking operations."""

    def test_masked_fill_inf(self) -> None:
        """Test filling masked positions."""
        tensor = torch.randn(3, 4)
        mask = torch.tensor([[True, False, False, True],
                             [False, True, False, False],
                             [True, True, False, False]])

        result = masked_fill_inf(tensor, mask)

        # Check masked positions
        assert torch.isinf(result[0, 0])
        assert torch.isinf(result[0, 3])
        assert torch.isinf(result[1, 1])
        assert torch.isinf(result[2, 0])
        assert torch.isinf(result[2, 1])

        # Check unmasked positions unchanged
        assert result[0, 1] == tensor[0, 1]
        assert result[1, 0] == tensor[1, 0]

        # Custom value
        result = masked_fill_inf(tensor, mask, value=999.0)
        assert result[0, 0] == 999.0
        assert result[1, 1] == 999.0

    def test_create_causal_mask(self) -> None:
        """Test causal mask creation."""
        # Small mask
        mask = create_causal_mask(4)
        expected = torch.tensor([
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True]
        ])
        assert torch.equal(mask, expected)

        # Check device
        if torch.cuda.is_available():
            mask_cuda = create_causal_mask(3, device="cuda")
            assert mask_cuda.device.type == "cuda"
            assert mask_cuda.shape == (3, 3)

    def test_create_padding_mask(self) -> None:
        """Test padding mask creation."""
        # Basic case
        lengths = torch.tensor([3, 2, 4])
        mask = create_padding_mask(lengths, max_length=5)

        expected = torch.tensor([
            [True, True, True, False, False],
            [True, True, False, False, False],
            [True, True, True, True, False]
        ])
        assert torch.equal(mask, expected)

        # Auto max length
        lengths = torch.tensor([2, 3, 1])
        mask = create_padding_mask(lengths)
        assert mask.shape == (3, 3)  # max_length = 3

        # Check values
        assert torch.all(mask[0, :2])
        assert not mask[0, 2]
        assert torch.all(mask[1, :3])
        assert mask[2, 0]
        assert torch.all(~mask[2, 1:])


class TestTensorSplitConcat:
    """Test tensor splitting and concatenation."""

    def test_split_tensor(self) -> None:
        """Test tensor splitting."""
        # Equal splits
        tensor = torch.randn(12, 5)
        splits = split_tensor(tensor, split_size=4, dim=0)

        assert len(splits) == 3
        assert all(s.shape == (4, 5) for s in splits)

        # Unequal splits
        tensor = torch.randn(10, 5)
        splits = split_tensor(tensor, split_sizes=[3, 3, 4], dim=0)

        assert len(splits) == 3
        assert splits[0].shape == (3, 5)
        assert splits[1].shape == (3, 5)
        assert splits[2].shape == (4, 5)

        # Split along different dimension
        tensor = torch.randn(5, 12)
        splits = split_tensor(tensor, split_size=3, dim=1)

        assert len(splits) == 4
        assert all(s.shape == (5, 3) for s in splits)

    def test_concat_tensors(self) -> None:
        """Test tensor concatenation."""
        tensors = [torch.randn(3, 5), torch.randn(2, 5), torch.randn(4, 5)]

        result = concat_tensors(tensors, dim=0)
        assert result.shape == (9, 5)

        # Check content
        assert torch.equal(result[:3], tensors[0])
        assert torch.equal(result[3:5], tensors[1])
        assert torch.equal(result[5:], tensors[2])

        # Concatenate along different dimension
        tensors = [torch.randn(5, 3), torch.randn(5, 2), torch.randn(5, 4)]
        result = concat_tensors(tensors, dim=1)
        assert result.shape == (5, 9)

    def test_stack_tensors(self) -> None:
        """Test tensor stacking."""
        tensors = [torch.randn(3, 4) for _ in range(5)]

        # Stack along new first dimension
        result = stack_tensors(tensors, dim=0)
        assert result.shape == (5, 3, 4)
        assert torch.equal(result[0], tensors[0])
        assert torch.equal(result[4], tensors[4])

        # Stack along different dimension
        result = stack_tensors(tensors, dim=1)
        assert result.shape == (3, 5, 4)

        # Stack along last dimension
        result = stack_tensors(tensors, dim=2)
        assert result.shape == (3, 4, 5)


class TestTensorStatistics:
    """Test TensorStatistics class."""

    def test_initialization(self) -> None:
        """Test statistics initialization."""
        stats = TensorStatistics()

        assert stats.count == 0
        assert stats.sum is None
        assert stats.sum_sq is None
        assert stats.min is None
        assert stats.max is None

    def test_single_update(self) -> None:
        """Test updating with single tensor."""
        stats = TensorStatistics()

        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        stats.update(tensor)

        assert stats.count == 5
        assert stats.mean == 3.0
        assert stats.std == pytest.approx(np.std([1, 2, 3, 4, 5]))
        assert stats.min_value == 1.0
        assert stats.max_value == 5.0

    def test_multiple_updates(self) -> None:
        """Test updating with multiple tensors."""
        stats = TensorStatistics()

        # Update with batches
        stats.update(torch.tensor([1.0, 2.0]))
        stats.update(torch.tensor([3.0, 4.0, 5.0]))
        stats.update(torch.tensor([6.0]))

        assert stats.count == 6
        assert stats.mean == 3.5
        assert stats.min_value == 1.0
        assert stats.max_value == 6.0

    def test_reset(self) -> None:
        """Test resetting statistics."""
        stats = TensorStatistics()

        stats.update(torch.randn(100))
        assert stats.count > 0

        stats.reset()

        assert stats.count == 0
        assert stats.sum is None
        assert stats.mean is None

    def test_empty_stats(self) -> None:
        """Test properties with no data."""
        stats = TensorStatistics()

        assert stats.mean is None
        assert stats.std is None
        assert stats.min_value is None
        assert stats.max_value is None

    def test_summary(self) -> None:
        """Test summary generation."""
        stats = TensorStatistics()

        # Add some data
        stats.update(torch.randn(50))
        stats.update(torch.randn(30, 20))

        summary = stats.summary()

        assert "count" in summary
        assert "mean" in summary
        assert "std" in summary
        assert "min" in summary
        assert "max" in summary

        assert summary["count"] == 650  # 50 + 600
        assert isinstance(summary["mean"], float)
        assert isinstance(summary["std"], float)

    def test_numerical_stability(self) -> None:
        """Test numerical stability with large values."""
        stats = TensorStatistics()

        # Large values
        stats.update(torch.tensor([1e8, 1e8 + 1, 1e8 + 2]))

        # Should compute correct statistics
        assert stats.mean == pytest.approx(1e8 + 1, rel=1e-6)
        assert stats.std == pytest.approx(np.std([0, 1, 2]), rel=1e-6)

        # Very small values
        stats2 = TensorStatistics()
        stats2.update(torch.tensor([1e-8, 2e-8, 3e-8]))

        assert stats2.mean == pytest.approx(2e-8, rel=1e-6)


class TestEdgeCases:
    """Test edge cases for tensor utilities."""

    def test_empty_tensors(self) -> None:
        """Test handling of empty tensors."""
        # Empty tensor operations
        empty = torch.empty(0, 5)

        result = safe_log(empty)
        assert result.shape == (0, 5)

        result = log_sum_exp(empty, dim=0)
        assert result.shape == (5,)
        assert torch.all(result == float('-inf'))

        # Empty batch operations
        a_empty = torch.empty(0, 3)
        b_empty = torch.empty(0, 4)
        result = batch_outer_product(a_empty, b_empty)
        assert result.shape == (0, 3, 4)

    def test_scalar_operations(self) -> None:
        """Test operations on scalars."""
        scalar = torch.tensor(3.14)

        # Safe operations
        result = safe_log(scalar)
        assert result.dim() == 0
        assert result.item() == pytest.approx(np.log(3.14))

        result = safe_sqrt(scalar)
        assert result.dim() == 0
        assert result.item() == pytest.approx(np.sqrt(3.14))

    def test_large_tensors(self) -> None:
        """Test with large tensors."""
        # Large tensor operations should work
        large = torch.randn(10000, 100)

        result = log_sum_exp(large, dim=1)
        assert result.shape == (10000,)
        assert torch.all(torch.isfinite(result))

        # Statistics should handle large tensors
        stats = TensorStatistics()
        stats.update(large)

        assert stats.count == 1_000_000
        assert abs(stats.mean) < 0.1  # Should be close to 0 for randn
        assert 0.9 < stats.std < 1.1  # Should be close to 1 for randn

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_consistency(self) -> None:
        """Test operations maintain device consistency."""
        # Create CUDA tensors
        x_cuda = torch.randn(5, 5, device="cuda")
        y_cuda = torch.randn(5, 3, device="cuda")

        # Operations should preserve device
        result = safe_log(x_cuda)
        assert result.device.type == "cuda"

        result = batch_outer_product(x_cuda, y_cuda)
        assert result.device.type == "cuda"

        # Mask creation with device
        mask = create_causal_mask(5, device="cuda")
        assert mask.device.type == "cuda"
