"""Utility functions for the EBM library."""

from .data import (
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
from .initialization import (
    Initializer,
    InitMethod,
    calculate_gain,
    get_fan_in_and_fan_out,
    init_from_data_statistics,
    initialize_module,
    kaiming_init,
    normal_init,
    uniform_init,
    xavier_init,
)
from .tensor import (
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
from .visualization import (
    create_animation,
    plot_energy_histogram,
    plot_reconstruction_comparison,
    plot_training_curves,
    setup_style,
    tile_images,
    visualize_filters,
    visualize_samples,
)

__all__ = [
    # Tensor utilities
    "ensure_tensor", "safe_log", "safe_sqrt",
    "log_sum_exp", "batch_outer_product", "batch_quadratic_form",
    "batch_mv", "shape_for_broadcast", "expand_dims_like",
    "masked_fill_inf", "create_causal_mask", "create_padding_mask",
    "split_tensor", "concat_tensors", "stack_tensors",
    "TensorStatistics",

    # Initialization
    "InitMethod", "Initializer",
    "get_fan_in_and_fan_out", "calculate_gain",
    "initialize_module",
    "uniform_init", "normal_init", "xavier_init", "kaiming_init",
    "init_from_data_statistics",

    # Data utilities
    "BinaryTransform", "AddNoise", "DequantizeTransform",
    "EnergyDataset", "SyntheticDataset",
    "get_mnist_datasets", "get_fashion_mnist_datasets",
    "create_data_loaders", "compute_data_statistics",

    # Visualization
    "setup_style", "tile_images",
    "visualize_filters", "visualize_samples",
    "plot_training_curves", "plot_energy_histogram",
    "plot_reconstruction_comparison", "create_animation",
]
