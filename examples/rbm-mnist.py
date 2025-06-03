"""Complete example of training an RBM on MNIST using the EBM library.

This script demonstrates:
- Loading and preprocessing data
- Creating and configuring models
- Setting up training with various samplers
- Using callbacks for monitoring
- Evaluating the trained model
- Visualizing results
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ebm
from ebm import (
    AISEstimator,
    BernoulliRBM,
    ContrastiveDivergence,
    GaussianBernoulliRBM,
    ModelEvaluator,
    OptimizerConfig,
    PersistentContrastiveDivergence,
    RBMConfig,
    Trainer,
    TrainingConfig,
    VisualizationCallback,
    WarmupCallback,
    create_data_loaders,
    get_mnist_datasets,
    plot_energy_histogram,
    plot_training_curves,
    visualize_filters,
    visualize_samples,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RBM on MNIST")

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="bernoulli",
        choices=["bernoulli", "centered", "gaussian"],
        help="RBM variant to use",
    )
    parser.add_argument(
        "--hidden", type=int, default=500, help="Number of hidden units"
    )
    parser.add_argument(
        "--weight-init",
        type=str,
        default="xavier_normal",
        help="Weight initialization method",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="SGD momentum"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0001, help="Weight decay"
    )

    # Sampler arguments
    parser.add_argument(
        "--sampler",
        type=str,
        default="pcd",
        choices=["cd", "pcd", "pt", "tap"],
        help="Sampling method",
    )
    parser.add_argument(
        "--k", type=int, default=1, help="Number of Gibbs steps"
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=100,
        help="Number of persistent chains",
    )
    parser.add_argument(
        "--num-temps",
        type=int,
        default=10,
        help="Number of temperatures for PT",
    )

    # Other arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda/cpu/auto)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Data directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output", help="Output directory"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="Logging level"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization during training",
    )
    parser.add_argument(
        "--estimate-logz",
        action="store_true",
        help="Estimate log partition function",
    )

    return parser.parse_args()


def create_model(args):
    """Create RBM model based on arguments."""
    # Model configuration
    if args.model == "gaussian":
        config = ebm.GaussianRBMConfig(
            visible_units=784,
            hidden_units=args.hidden,
            weight_init=args.weight_init,
            device=args.device,
            seed=args.seed,
            learn_sigma=True,
        )
        model = GaussianBernoulliRBM(config)
    else:
        config = RBMConfig(
            visible_units=784,
            hidden_units=args.hidden,
            weight_init=args.weight_init,
            device=args.device,
            seed=args.seed,
            centered=(args.model == "centered"),
        )
        if args.model == "centered":
            model = ebm.CenteredBernoulliRBM(config)
        else:
            model = BernoulliRBM(config)

    return model, config


def create_sampler(args, model):
    """Create gradient estimator based on arguments."""
    if args.sampler == "cd":
        sampler = ContrastiveDivergence(k=args.k)
    elif args.sampler == "pcd":
        sampler = PersistentContrastiveDivergence(
            k=args.k, num_chains=args.num_chains
        )
    elif args.sampler == "pt":
        sampler = ebm.PTGradientEstimator(num_temps=args.num_temps, k=args.k)
    elif args.sampler == "tap":
        sampler = ebm.TAPGradientEstimator(
            num_iter=20, damping=0.5, order="tap2"
        )
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    return sampler


def main() -> None:
    """Run the RBM training script."""
    args = parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    ebm.setup_logging(level=args.log_level, file=output_dir / "training.log")
    logger = ebm.logger
    logger.info("Starting RBM training", args=vars(args))

    # Set device
    ebm.set_device(args.device)
    device = ebm.get_device()
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading MNIST dataset...")
    train_dataset, val_dataset, test_dataset = get_mnist_datasets(
        data_dir=args.data_dir, binary=(args.model != "gaussian"), flatten=True
    )

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    logger.info("Creating model...")
    model, model_config = create_model(args)
    param_summary = model.parameter_summary()
    logger.info("Model created", **param_summary)

    # Initialize from data
    model.init_from_data(train_loader)

    # Create sampler
    logger.info(f"Creating {args.sampler} sampler...")
    gradient_estimator = create_sampler(args, model)

    # Training configuration
    optimizer_config = OptimizerConfig(
        name="sgd",
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        scheduler="cosine",
        scheduler_params={"eta_min": args.lr * 0.01},
    )

    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=optimizer_config,
        checkpoint_dir=output_dir / "checkpoints",
        checkpoint_every=10,
        eval_every=5,
        log_every=100,
        early_stopping=True,
        patience=10,
    )

    # Setup callbacks
    callbacks = []

    # Add warmup
    if args.lr > 0.001:
        callbacks.append(
            WarmupCallback(
                warmup_steps=len(train_loader), start_lr=1e-4, end_lr=args.lr
            )
        )

    # Add visualization
    if args.visualize:
        callbacks.append(
            VisualizationCallback(
                visualize_every=5,
                num_samples=100,
                save_dir=output_dir / "visualizations",
            )
        )

    # Create trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        gradient_estimator=gradient_estimator,
        callbacks=callbacks,
    )

    # Train model
    logger.info("Starting training...")
    history = trainer.fit(train_loader, val_loader)

    # Save final model
    final_checkpoint = output_dir / "final_model.pt"
    model.save_checkpoint(final_checkpoint)
    logger.info(f"Saved final model to {final_checkpoint}")

    # Plot training curves
    logger.info("Plotting training curves...")
    fig = plot_training_curves(history["history"])
    fig.savefig(output_dir / "training_curves.png")
    plt.close(fig)

    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = ModelEvaluator(model)

    # Reconstruction error
    test_batch = next(iter(test_loader))[:100].to(device)
    recon_errors = evaluator.reconstruction_error(test_batch, num_steps=10)
    logger.info(
        f"Reconstruction error: {recon_errors.mean():.4f} ± {recon_errors.std():.4f}"
    )

    # Energy gap
    energy_stats = evaluator.energy_gap(test_batch, num_model_samples=100)
    logger.info("Energy statistics", **energy_stats)

    # Generate samples
    logger.info("Generating samples...")
    if hasattr(model, "sample_fantasy_particles"):
        samples = model.sample_fantasy_particles(
            num_samples=100, num_steps=1000
        )

        # Visualize samples
        fig = visualize_samples(samples, title="Generated Samples")
        fig.savefig(output_dir / "generated_samples.png")
        plt.close(fig)

        # Plot energy histogram
        test_energies = model.free_energy(test_batch)
        sample_energies = model.free_energy(samples)
        fig = plot_energy_histogram(test_energies, sample_energies)
        fig.savefig(output_dir / "energy_histogram.png")
        plt.close(fig)

    # Visualize filters
    logger.info("Visualizing filters...")
    fig = visualize_filters(model.W, title="Learned Filters")
    fig.savefig(output_dir / "filters.png")
    plt.close(fig)

    # Estimate partition function
    if args.estimate_logz:
        logger.info("Estimating partition function...")
        ais_estimator = AISEstimator(model, num_temps=10000, num_chains=100)
        log_z, diagnostics = ais_estimator.estimate(return_diagnostics=True)
        logger.info(
            f"Log partition function: {log_z:.2f} ± {diagnostics['log_Z_std']:.2f}",
            ESS=diagnostics["effective_sample_size"],
        )

        # Save diagnostics
        import json

        with open(output_dir / "ais_diagnostics.json", "w") as f:
            json.dump(
                {
                    k: v
                    for k, v in diagnostics.items()
                    if not isinstance(v, np.ndarray)
                },
                f,
                indent=2,
            )

    logger.info("Training completed successfully!")

    # Print summary
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Model: {args.model} RBM with {args.hidden} hidden units")
    print(f"Sampler: {args.sampler} (k={args.k})")
    print(f"Final reconstruction error: {recon_errors.mean():.4f}")
    print(f"Energy gap: {energy_stats['energy_gap']:.2f}")
    if args.estimate_logz:
        print(
            f"Log partition function: {log_z:.2f} ± {diagnostics['log_Z_std']:.2f}"
        )
    print(f"Results saved to: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
