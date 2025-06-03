"""Visualization utilities for energy-based models.

This module provides functions for visualizing model parameters,
samples, training dynamics, and other aspects of EBMs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.figure import Figure
from torch import Tensor

GRAY_NDIMS = 2
IMAGE_NDIM_3 = 3
IMAGE_NDIM_4 = 4
MARKER_THRESHOLD = 50

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def setup_style(style: str = "whitegrid") -> None:
    """Set up matplotlib style for consistent plots.

    Args:
        style: Style name ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
    """
    if HAS_SEABORN:
        sns.set_style(style)
        sns.set_context("paper", font_scale=1.2)
    else:
        plt.style.use(
            "seaborn-v0_8"
            if "seaborn-v0_8" in plt.style.available
            else "default"
        )


def _to_numpy(images: Tensor | np.ndarray) -> np.ndarray:
    """Convert images to numpy array."""
    if isinstance(images, Tensor):
        return images.detach().cpu().numpy()
    return images


def _normalize(images: np.ndarray, scale_each: bool) -> np.ndarray:
    # Make a copy to avoid modifying the input
    images = images.copy()
    if scale_each:
        for i in range(images.shape[0]):
            img_min, img_max = images[i].min(), images[i].max()
            if img_max > img_min:
                images[i] = (images[i] - img_min) / (img_max - img_min)
    else:
        img_min, img_max = images.min(), images.max()
        if img_max > img_min:
            images = (images - img_min) / (img_max - img_min)
    return images


def _process_images(
    images: Tensor | np.ndarray,
    normalize: bool,
    scale_each: bool,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Process images for tiling.

    Returns processed images array and original shape info.
    """
    images_np = _to_numpy(images)

    if images_np.ndim == IMAGE_NDIM_3:
        n, h, w = images_np.shape
        c = 1
        images_np = images_np[:, np.newaxis]
    elif images_np.ndim == IMAGE_NDIM_4:
        n, c, h, w = images_np.shape
    else:
        raise ValueError(f"Expected 3D or 4D array, got {images_np.ndim}D")

    # Handle empty images
    if n == 0:
        return np.array([]), (0, c, h, w)

    if normalize:
        images_np = _normalize(images_np, scale_each)

    return images_np, (n, c, h, w)


def _add_padding(
    images_np: np.ndarray,
    padding: int,
    pad_value: float,
    shape_info: tuple[int, ...],
) -> np.ndarray:
    """Add padding to images."""
    if padding <= 0:
        return images_np

    n, c, h, w = shape_info
    nrows_ncols = images_np.shape[0]

    # Pad each image
    padded_images = []
    for idx in range(nrows_ncols):
        img = images_np[idx]  # shape (c, h, w)
        # Add padding on all sides
        padded_img = np.pad(
            img,
            ((0, 0), (padding, padding), (padding, padding)),
            mode="constant",
            constant_values=pad_value,
        )
        padded_images.append(padded_img)

    return np.stack(padded_images)


def tile_images(
    images: Tensor | np.ndarray,
    nrows: int | None = None,
    ncols: int | None = None,
    padding: int = 0,
    pad_value: float = 0.0,
    scale_each: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """Tile images into a grid.

    Args:
        images: Input images
        nrows: Number of rows in grid
        ncols: Number of columns in grid
        padding: Padding around each image (applied to all sides)
        pad_value: Value to use for padding
        scale_each: Whether to normalize each image independently
        normalize: Whether to normalize images

    Returns
    -------
        Tiled image array
    """
    # Process images
    images_np, (n, c, h, w) = _process_images(images, normalize, scale_each)

    if n == 0:
        return images_np

    # Calculate grid dimensions
    if nrows is None and ncols is None:
        nrows = int(np.ceil(np.sqrt(n)))
        ncols = int(np.ceil(n / nrows))
    elif nrows is None:
        nrows = int(np.ceil(n / ncols))
    elif ncols is None:
        ncols = int(np.ceil(n / nrows))

    # Pad with extra images if needed
    n_pad = nrows * ncols - n
    if n_pad > 0:
        padding_shape = (n_pad, c, h, w)
        padding_images = np.full(
            padding_shape, pad_value, dtype=images_np.dtype
        )
        images_np = np.concatenate([images_np, padding_images], axis=0)

    # Add padding around each image
    if padding > 0:
        images_np = _add_padding(
            images_np, padding, pad_value, (nrows * ncols, c, h, w)
        )
        h += 2 * padding
        w += 2 * padding

    # Reshape to grid
    images_np = images_np.reshape(nrows, ncols, c, h, w)

    # Transpose to (nrows, h, ncols, w, c)
    images_np = images_np.transpose(0, 3, 1, 4, 2)
    # Reshape to final grid
    images_np = images_np.reshape(nrows * h, ncols * w, c)

    if c == 1:
        images_np = images_np.squeeze(-1)

    return images_np


def visualize_filters(
    weights: Tensor,
    title: str = "Filters",
    cmap: str = "RdBu_r",
    save_path: Path | None = None,
    figsize: tuple[int, int] | None = None,
    **kwargs: Any,
) -> Figure:
    """Visualize weight filters (e.g., RBM weights).

    Args:
        weights: Weight matrix of shape (num_filters, filter_size)
        title: Figure title
        cmap: Colormap
        save_path: Path to save figure
        figsize: Figure size
        **kwargs: Additional arguments for tile_images

    Returns
    -------
        Matplotlib figure
    """
    # Determine image shape
    num_filters, filter_size = weights.shape

    # Try to make square filters
    side = int(np.sqrt(filter_size))
    img_shape = (side, side) if side * side == filter_size else (1, filter_size)

    # Reshape weights
    weight_imgs = weights.reshape(num_filters, *img_shape)

    # Set defaults for tile_images if not specified in kwargs
    tile_kwargs = kwargs.copy()
    if "padding" not in tile_kwargs:
        tile_kwargs["padding"] = 2
    if "normalize" not in tile_kwargs:
        tile_kwargs["normalize"] = True
    if "scale_each" not in tile_kwargs:
        tile_kwargs["scale_each"] = True

    tiled = tile_images(weight_imgs, **tile_kwargs)

    # Create figure
    if figsize is None:
        figsize = (10, 10)
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(tiled, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.axis("off")

    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_samples(
    samples: Tensor,
    title: str = "Generated Samples",
    nrows: int | None = None,
    ncols: int | None = None,
    save_path: Path | None = None,
    figsize: tuple[int, int] | None = None,
    **kwargs: Any,
) -> Figure:
    """Visualize generated samples.

    Args:
        samples: Samples of shape (N, D) or (N, H, W) or (N, C, H, W)
        title: Figure title
        nrows: Number of rows
        ncols: Number of columns
        save_path: Path to save figure
        figsize: Figure size
        **kwargs: Additional arguments for tile_images

    Returns
    -------
        Matplotlib figure
    """
    # Handle flattened samples
    if samples.dim() == GRAY_NDIMS:
        n, d = samples.shape
        # Try to reshape to square images
        side = int(np.sqrt(d))
        if side * side == d:
            samples = samples.reshape(n, side, side)
        else:
            # Keep as vectors
            samples = samples.reshape(n, 1, d)

    # Set default padding if not specified
    if "padding" not in kwargs:
        kwargs["padding"] = 2

    # Tile samples
    tiled = tile_images(samples, nrows=nrows, ncols=ncols, **kwargs)

    # Create figure
    if figsize is None:
        figsize = (12, 12)
    fig, ax = plt.subplots(figsize=figsize)

    # Determine colormap
    cmap = "gray" if tiled.ndim == GRAY_NDIMS else None

    ax.imshow(tiled, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(  # noqa: C901
    history: dict[str, list[dict[str, float]]],
    metrics: list[str] | None = None,
    save_path: Path | None = None,
    figsize: tuple[int, int] | None = None,
) -> Figure:
    """Plot training curves.

    Args:
        history: Training history with 'train' and optionally 'val' keys
        metrics: Metrics to plot (plots all if None)
        save_path: Path to save figure
        figsize: Figure size

    Returns
    -------
        Matplotlib figure
    """
    # Extract metrics to plot
    if metrics is None:
        all_metrics = set()
        for phase_history in history.values():
            if phase_history:
                all_metrics.update(phase_history[0].keys())
        metrics = sorted([m for m in all_metrics if not m.startswith("_")])

    # Filter out non-numeric metrics
    metrics = [m for m in metrics if m not in ["epoch", "timestamp"]]

    if not metrics:
        raise ValueError("No metrics to plot")

    # Create subplots
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Plot each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        for phase, phase_history in history.items():
            if not phase_history:
                continue

            epochs = [h.get("epoch", i) for i, h in enumerate(phase_history)]
            values = [h.get(metric, np.nan) for h in phase_history]

            # Filter out NaN values
            valid_idx = ~np.isnan(values)
            epochs = np.array(epochs)[valid_idx]
            values = np.array(values)[valid_idx]

            if len(values) > 0:
                ax.plot(
                    epochs,
                    values,
                    label=phase,
                    marker="o" if len(values) < MARKER_THRESHOLD else None,
                )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_energy_histogram(
    data_energies: Tensor,
    model_energies: Tensor,
    title: str = "Energy Distribution",
    save_path: Path | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> Figure:
    """Plot histogram of energies for data vs model samples.

    Args:
        data_energies: Energies of data samples
        model_energies: Energies of model samples
        title: Figure title
        save_path: Path to save figure
        figsize: Figure size

    Returns
    -------
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to numpy
    data_e = data_energies.detach().cpu().numpy()
    model_e = model_energies.detach().cpu().numpy()

    # Plot histograms
    bins = np.linspace(
        min(data_e.min(), model_e.min()), max(data_e.max(), model_e.max()), 50
    )

    ax.hist(data_e, bins=bins, alpha=0.5, label="Data", density=True)
    ax.hist(model_e, bins=bins, alpha=0.5, label="Model", density=True)

    # Add statistics
    ax.axvline(
        data_e.mean(),
        color="blue",
        linestyle="--",
        label=f"Data mean: {data_e.mean():.1f}",
    )
    ax.axvline(
        model_e.mean(),
        color="orange",
        linestyle="--",
        label=f"Model mean: {model_e.mean():.1f}",
    )

    ax.set_xlabel("Energy")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_reconstruction_comparison(
    original: Tensor,
    reconstructed: Tensor,
    n_examples: int = 10,
    title: str = "Reconstruction Comparison",
    save_path: Path | None = None,
    figsize: tuple[int, int] | None = None,
) -> Figure:
    """Plot original vs reconstructed samples side by side.

    Args:
        original: Original samples
        reconstructed: Reconstructed samples
        n_examples: Number of examples to show
        title: Figure title
        save_path: Path to save figure
        figsize: Figure size

    Returns
    -------
        Matplotlib figure
    """
    n_examples = min(n_examples, original.shape[0])

    # Interleave original and reconstructed
    combined = torch.stack(
        [original[:n_examples], reconstructed[:n_examples]], dim=1
    ).reshape(2 * n_examples, *original.shape[1:])

    # Create tiled image with padding
    tiled = tile_images(combined, nrows=2, ncols=n_examples, padding=2)

    if figsize is None:
        figsize = (2 * n_examples, 4)

    fig, ax = plt.subplots(figsize=figsize)

    cmap = "gray" if tiled.ndim == GRAY_NDIMS else None
    ax.imshow(tiled, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.axis("off")

    # Add labels
    ax.text(0.25, -0.05, "Original", transform=ax.transAxes, ha="center")
    ax.text(0.75, -0.05, "Reconstructed", transform=ax.transAxes, ha="center")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_animation(
    frames: list[Tensor],
    fps: int = 10,
    save_path: Path | None = None,
    title: str = "Animation",
) -> animation.FuncAnimation:
    """Create animation from a sequence of frames.

    Args:
        frames: List of image tensors
        fps: Frames per second
        save_path: Path to save animation (as GIF or MP4)
        title: Animation title

    Returns
    -------
        Animation object
    """
    # Prepare frames
    frames_np = []
    for frm in frames:
        frm_np = frm.detach().cpu().numpy() if isinstance(frm, Tensor) else frm
        frames_np.append(frm_np)

    # Create figure
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.axis("off")

    # Determine colormap
    cmap = "gray" if frames_np[0].ndim == GRAY_NDIMS else None

    # Create animation
    im = ax.imshow(frames_np[0], cmap=cmap, aspect="auto")

    def update(i: int) -> list[plt.Artist]:
        im.set_array(frames_np[i])
        return [im]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames_np), interval=1000 / fps, blit=True
    )

    if save_path is not None:
        if save_path.suffix == ".gif":
            anim.save(save_path, writer="pillow", fps=fps)
        else:
            anim.save(save_path, writer="ffmpeg", fps=fps)

    plt.close(fig)

    return anim
