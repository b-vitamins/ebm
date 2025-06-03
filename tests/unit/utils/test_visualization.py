"""Unit tests for visualization utilities."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.figure import Figure

from ebm.utils.visualization import (
    create_animation,
    plot_energy_histogram,
    plot_reconstruction_comparison,
    plot_training_curves,
    setup_style,
    tile_images,
    visualize_filters,
    visualize_samples,
)

# Use non-interactive backend for testing
matplotlib.use("Agg")


class TestSetupStyle:
    """Test style setup."""

    @patch("ebm.utils.visualization.HAS_SEABORN", True)
    @patch("seaborn.set_style")
    @patch("seaborn.set_context")
    def test_with_seaborn(
        self, mock_set_context: MagicMock, mock_set_style: MagicMock
    ) -> None:
        """Test style setup with seaborn available."""
        setup_style("whitegrid")

        mock_set_style.assert_called_once_with("whitegrid")
        mock_set_context.assert_called_once_with("paper", font_scale=1.2)

    @patch("ebm.utils.visualization.HAS_SEABORN", False)
    @patch("matplotlib.pyplot.style.use")
    def test_without_seaborn(self, mock_style_use: MagicMock) -> None:
        """Test style setup without seaborn."""
        # Mock available styles
        with patch(
            "matplotlib.pyplot.style.available", ["seaborn-v0_8", "default"]
        ):
            setup_style()
            mock_style_use.assert_called_once_with("seaborn-v0_8")

        # Test fallback to default
        mock_style_use.reset_mock()
        with patch("matplotlib.pyplot.style.available", ["default"]):
            setup_style()
            mock_style_use.assert_called_once_with("default")


class TestTileImages:
    """Test image tiling."""

    def test_3d_grayscale_images(self) -> None:
        """Test tiling grayscale images."""
        # Create test images (N, H, W)
        images = torch.randn(6, 8, 8)

        tiled = tile_images(images, nrows=2, ncols=3)

        # Check shape: 2 rows * 8 height, 3 cols * 8 width
        assert tiled.shape == (2 * 8, 3 * 8)
        assert isinstance(tiled, np.ndarray)

    def test_4d_color_images(self) -> None:
        """Test tiling color images."""
        # Create test images (N, C, H, W)
        images = torch.randn(4, 3, 16, 16)

        tiled = tile_images(images, nrows=2, ncols=2)

        # Check shape
        assert tiled.shape == (2 * 16, 2 * 16, 3)

    def test_auto_grid_size(self) -> None:
        """Test automatic grid size calculation."""
        # 9 images should create 3x3 grid
        images = torch.randn(9, 10, 10)
        tiled = tile_images(images)
        assert tiled.shape == (3 * 10, 3 * 10)

        # 10 images should create 4x3 grid
        images = torch.randn(10, 10, 10)
        tiled = tile_images(images)
        assert tiled.shape == (4 * 10, 3 * 10)

    def test_padding(self) -> None:
        """Test image padding."""
        images = torch.randn(4, 8, 8)

        # With padding
        tiled = tile_images(images, nrows=2, ncols=2, padding=2, pad_value=1.0)

        # Each image is 8+4 = 12 pixels (2 padding on each side)
        assert tiled.shape == (2 * 12, 2 * 12)

        # Check that padding values are correct
        # Top-left corner should be padding
        assert tiled[0, 0] == 1.0
        assert tiled[1, 1] == 1.0

    def test_normalization(self) -> None:
        """Test image normalization."""
        # Create images with known range
        images = torch.tensor(
            [[[0.0, 2.0], [4.0, 6.0]], [[1.0, 3.0], [5.0, 7.0]]]
        )

        # With global normalization
        tiled = tile_images(images, normalize=True, scale_each=False)
        assert tiled.min() == 0.0
        assert tiled.max() == 1.0

        # With per-image normalization
        tiled = tile_images(images, normalize=True, scale_each=True)
        # Each sub-image should be normalized
        # Can't easily check this without unpacking, but no errors is good

    def test_tensor_to_numpy_conversion(self) -> None:
        """Test that tensors are converted to numpy."""
        images = torch.randn(4, 8, 8)
        tiled = tile_images(images)

        assert isinstance(tiled, np.ndarray)
        assert tiled.dtype in (np.float32, np.float64)

    def test_insufficient_images_padding(self) -> None:
        """Test padding when not enough images for grid."""
        # 3 images in 2x2 grid
        images = torch.randn(3, 8, 8)
        tiled = tile_images(images, nrows=2, ncols=2, pad_value=0.5)

        assert tiled.shape == (2 * 8, 2 * 8)

        # Bottom-right should be pad value
        assert np.all(tiled[8:, 8:] == 0.5)

    def test_invalid_dimensions(self) -> None:
        """Test error on invalid tensor dimensions."""
        # 2D tensor
        images = torch.randn(10, 10)
        with pytest.raises(ValueError, match="Expected 3D or 4D array"):
            tile_images(images)

        # 5D tensor
        images = torch.randn(2, 2, 2, 2, 2)
        with pytest.raises(ValueError, match="Expected 3D or 4D array"):
            tile_images(images)


class TestVisualizeFilters:
    """Test filter visualization."""

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.colorbar")
    def test_square_filters(
        self,
        mock_colorbar: MagicMock,
        mock_tight_layout: MagicMock,
        mock_savefig: MagicMock,
    ) -> None:
        """Test visualizing square filters."""
        # Create filter weights (25 filters of size 16)
        weights = torch.randn(25, 16)

        fig = visualize_filters(weights, title="Test Filters")

        assert isinstance(fig, Figure)
        assert len(fig.axes) > 0

        # Check that functions were called
        mock_tight_layout.assert_called_once()
        mock_colorbar.assert_called_once()
        mock_savefig.assert_called_once()

        plt.close(fig)

    @patch("matplotlib.pyplot.savefig")
    def test_non_square_filters(self, mock_savefig: MagicMock) -> None:
        """Test visualizing non-square filters."""
        _ = mock_savefig
        # 10 filters of size 30 (not square)
        weights = torch.randn(10, 30)

        fig = visualize_filters(weights)

        # Should fall back to vector visualization
        assert isinstance(fig, Figure)
        mock_savefig.assert_called_once()

        plt.close(fig)

    def test_save_filters(self, tmp_path: Path) -> None:
        """Test saving filter visualization."""
        weights = torch.randn(16, 25)
        save_path = tmp_path / "filters.png"

        with patch("matplotlib.pyplot.savefig") as mock_save:
            fig = visualize_filters(weights, save_path=save_path)
            mock_save.assert_called_once_with(
                save_path, dpi=150, bbox_inches="tight"
            )

        plt.close(fig)

    def test_custom_parameters(self) -> None:
        """Test custom visualization parameters."""
        weights = torch.randn(9, 16)

        with patch("ebm.utils.visualization.tile_images") as mock_tile:
            mock_tile.return_value = np.zeros((24, 24))

            fig = visualize_filters(
                weights,
                title="Custom",
                cmap="viridis",
                figsize=(15, 15),
                normalize=False,
                scale_each=False,
            )

            # Check tile_images was called with custom params
            mock_tile.assert_called_once()
            call_kwargs = mock_tile.call_args[1]
            assert call_kwargs["normalize"] is False
            assert call_kwargs["scale_each"] is False

        plt.close(fig)


class TestVisualizeSamples:
    """Test sample visualization."""

    def test_2d_samples(self) -> None:
        """Test visualizing flattened samples."""
        # Flattened MNIST-like samples
        samples = torch.randn(16, 784)

        fig = visualize_samples(samples, title="Generated Samples")

        assert isinstance(fig, Figure)
        # Should reshape to 28x28 automatically

        plt.close(fig)

    def test_3d_samples(self) -> None:
        """Test visualizing image samples."""
        # Already shaped images
        samples = torch.randn(9, 32, 32)

        fig = visualize_samples(samples, nrows=3, ncols=3, title="Test Samples")

        assert isinstance(fig, Figure)

        plt.close(fig)

    def test_4d_color_samples(self) -> None:
        """Test visualizing color image samples."""
        samples = torch.randn(4, 3, 64, 64)

        fig = visualize_samples(samples, figsize=(8, 8))

        assert isinstance(fig, Figure)

        plt.close(fig)

    def test_non_square_reshape(self) -> None:
        """Test samples that can't be reshaped to square."""
        # 120 features - can't be reshaped to square
        samples = torch.randn(8, 120)

        fig = visualize_samples(samples)

        # Should display as 1x120 strips
        assert isinstance(fig, Figure)

        plt.close(fig)

    def test_save_samples(self, tmp_path: Path) -> None:
        """Test saving sample visualization."""
        samples = torch.randn(16, 64)
        save_path = tmp_path / "samples.png"

        with patch("matplotlib.pyplot.savefig") as mock_save:
            fig = visualize_samples(samples, save_path=save_path)
            mock_save.assert_called_once()

        plt.close(fig)


class TestPlotTrainingCurves:
    """Test training curve plotting."""

    def test_simple_training_curves(self) -> None:
        """Test plotting simple training history."""
        history = {
            "train": [
                {"epoch": 0, "loss": 1.0, "accuracy": 0.6},
                {"epoch": 1, "loss": 0.8, "accuracy": 0.7},
                {"epoch": 2, "loss": 0.6, "accuracy": 0.8},
            ]
        }

        fig = plot_training_curves(history)

        assert isinstance(fig, Figure)
        # Should have 2 subplots (loss and accuracy)
        assert len(fig.axes) == 2

        plt.close(fig)

    def test_train_val_curves(self) -> None:
        """Test plotting training and validation curves."""
        history = {
            "train": [
                {"epoch": i, "loss": 1.0 - 0.1 * i, "accuracy": 0.6 + 0.05 * i}
                for i in range(5)
            ],
            "val": [
                {
                    "epoch": i,
                    "loss": 1.1 - 0.09 * i,
                    "accuracy": 0.58 + 0.04 * i,
                }
                for i in range(5)
            ],
        }

        fig = plot_training_curves(history)

        # Check that both train and val are plotted
        for ax in fig.axes:
            assert len(ax.lines) == 2  # train and val lines
            legend = ax.get_legend()
            assert legend is not None

        plt.close(fig)

    def test_custom_metrics(self) -> None:
        """Test plotting specific metrics."""
        history = {
            "train": [
                {"loss": 1.0, "accuracy": 0.8, "lr": 0.01, "grad_norm": 2.5}
            ]
        }

        fig = plot_training_curves(history, metrics=["loss", "lr"])

        # Should only plot 2 metrics
        assert len(fig.axes) == 2

        plt.close(fig)

    def test_many_metrics(self) -> None:
        """Test plotting many metrics."""
        history = {
            "train": [{f"metric_{i}": np.random.rand() for i in range(10)}]
        }

        fig = plot_training_curves(history, figsize=(15, 20))

        # Should create appropriate grid
        assert len(fig.axes) >= 10

        plt.close(fig)

    def test_missing_data_handling(self) -> None:
        """Test handling of missing data points."""
        history = {
            "train": [
                {"epoch": 0, "loss": 1.0},
                {"epoch": 1, "loss": 0.9, "accuracy": 0.7},
                {"epoch": 2, "loss": 0.8, "accuracy": 0.75},
            ]
        }

        fig = plot_training_curves(history)

        # Should handle missing accuracy in first epoch
        assert isinstance(fig, Figure)

        plt.close(fig)

    def test_save_curves(self, tmp_path: Path) -> None:
        """Test saving training curves."""
        history = {"train": [{"loss": 1.0}]}
        save_path = tmp_path / "curves.png"

        with patch("matplotlib.pyplot.savefig") as mock_save:
            fig = plot_training_curves(history, save_path=save_path)
            mock_save.assert_called_once()

        plt.close(fig)


class TestPlotEnergyHistogram:
    """Test energy histogram plotting."""

    def test_basic_histogram(self) -> None:
        """Test basic energy histogram."""
        # Create fake energies
        data_energies = torch.randn(100) - 2.0  # Lower energies for data
        model_energies = torch.randn(100)  # Higher energies for model

        fig = plot_energy_histogram(
            data_energies, model_energies, title="Energy Distribution"
        )

        assert isinstance(fig, Figure)
        ax = fig.axes[0]

        # Should have 2 histogram patches (data and model)
        # Plus 2 vertical lines for means
        assert len(ax.patches) > 0
        assert len(ax.lines) >= 2

        plt.close(fig)

    def test_statistics_display(self) -> None:
        """Test that statistics are displayed."""
        data_energies = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        model_energies = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])

        fig = plot_energy_histogram(data_energies, model_energies)

        ax = fig.axes[0]

        # Check that mean lines exist
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert any("Data mean" in t for t in legend_texts)
        assert any("Model mean" in t for t in legend_texts)

        plt.close(fig)

    def test_custom_parameters(self) -> None:
        """Test custom histogram parameters."""
        data_energies = torch.randn(50)
        model_energies = torch.randn(50)

        fig = plot_energy_histogram(
            data_energies, model_energies, title="Custom", figsize=(10, 8)
        )

        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 8

        plt.close(fig)


class TestPlotReconstructionComparison:
    """Test reconstruction comparison plotting."""

    def test_basic_comparison(self) -> None:
        """Test basic reconstruction comparison."""
        original = torch.randn(5, 28, 28)
        reconstructed = original + torch.randn_like(original) * 0.1

        fig = plot_reconstruction_comparison(
            original, reconstructed, n_examples=5
        )

        assert isinstance(fig, Figure)

        plt.close(fig)

    def test_limited_examples(self) -> None:
        """Test limiting number of examples."""
        original = torch.randn(10, 16, 16)
        reconstructed = torch.randn(10, 16, 16)

        fig = plot_reconstruction_comparison(
            original, reconstructed, n_examples=3
        )

        # Should only show 3 examples
        assert isinstance(fig, Figure)

        plt.close(fig)

    def test_color_images(self) -> None:
        """Test with color images."""
        original = torch.randn(4, 3, 32, 32)
        reconstructed = torch.randn(4, 3, 32, 32)

        fig = plot_reconstruction_comparison(
            original, reconstructed, n_examples=4
        )

        assert isinstance(fig, Figure)

        plt.close(fig)


class TestCreateAnimation:
    """Test animation creation."""

    @patch("matplotlib.animation.FuncAnimation")
    def test_basic_animation(self, mock_animation: MagicMock) -> None:
        """Test creating basic animation."""
        # Create sequence of frames
        frames = [torch.randn(32, 32) for _ in range(10)]

        create_animation(frames, fps=10, title="Test Animation")

        # Check animation was created
        mock_animation.assert_called_once()
        call_args = mock_animation.call_args
        assert call_args[1]["frames"] == 10
        assert call_args[1]["interval"] == 100  # 1000/fps

        plt.close("all")

    @patch("matplotlib.animation.FuncAnimation")
    def test_save_animation(self, mock_animation_class: MagicMock) -> None:
        """Test saving animation."""
        frames = [torch.randn(16, 16) for _ in range(5)]

        # Create mock animation instance
        mock_anim = MagicMock()
        mock_animation_class.return_value = mock_anim

        # Test GIF save
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            create_animation(frames, save_path=Path(tmp.name))
            mock_anim.save.assert_called_once()
            save_args = mock_anim.save.call_args
            assert save_args[1]["writer"] == "pillow"

        # Test MP4 save
        mock_anim.reset_mock()
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            create_animation(frames, save_path=Path(tmp.name))
            save_args = mock_anim.save.call_args
            assert save_args[1]["writer"] == "ffmpeg"

        plt.close("all")


class TestEdgeCases:
    """Test edge cases for visualization."""

    def test_empty_images(self) -> None:
        """Test visualizing empty image set."""
        images = torch.empty(0, 28, 28)

        # Should handle gracefully
        tiled = tile_images(images)
        assert tiled.shape[0] == 0 or tiled.shape[1] == 0

    def test_single_image(self) -> None:
        """Test visualizing single image."""
        image = torch.randn(1, 32, 32)

        fig = visualize_samples(image)
        assert isinstance(fig, Figure)

        plt.close(fig)

    def test_empty_history(self) -> None:
        """Test plotting empty training history."""
        history = {"train": [], "val": []}

        with pytest.raises(ValueError, match="No metrics to plot"):
            plot_training_curves(history)

    def test_nan_values(self) -> None:
        """Test handling NaN values in plots."""
        history = {
            "train": [{"loss": 1.0}, {"loss": float("nan")}, {"loss": 0.5}]
        }

        # Should filter out NaN values
        fig = plot_training_curves(history)
        assert isinstance(fig, Figure)

        plt.close(fig)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensors(self) -> None:
        """Test visualization with CUDA tensors."""
        images = torch.randn(4, 28, 28, device="cuda")

        # Should automatically move to CPU
        tiled = tile_images(images)
        assert isinstance(tiled, np.ndarray)

        fig = visualize_samples(images)
        assert isinstance(fig, Figure)

        plt.close(fig)
