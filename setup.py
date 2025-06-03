"""Setup script for the EBM library."""

from pathlib import Path

from setuptools import find_packages, setup

# Read the README file
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")
else:
    long_description = "A modern PyTorch library for energy-based models."

# Read version from __init__.py
version_file = Path(__file__).parent / "ebm" / "__init__.py"
version = "0.1.0"  # Default version
if version_file.exists():
    with open(version_file) as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"\'')
                break

setup(
    name="ebm",
    version=version,
    author="EBM Contributors",
    author_email="",
    description="A modern PyTorch library for energy-based models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ebm",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.60.0",
        "pydantic>=2.0.0",
        "structlog>=21.0.0",
        "torchvision>=0.11.0",  # For datasets
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "mypy>=0.900",
            "pre-commit>=2.0",
        ],
        "viz": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "pillow>=8.0.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autodoc-typehints>=1.0",
            "nbsphinx>=0.8",
        ],
        "all": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "pillow>=8.0.0",
            "wandb>=0.12.0",
            "tensorboard>=2.0",
            "scikit-learn>=1.0",
            "pandas>=1.3.0",
            "jupyter>=1.0.0",
            "ipywidgets>=7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Could add CLI tools here
        ],
    },
    include_package_data=True,
    package_data={
        "ebm": ["py.typed"],  # For type hints
    },
    zip_safe=False,
)
