#!/bin/bash
set -euo pipefail

# setup-dev.sh - Prepare local development environment for the EBM project
# This script installs all Python dependencies using Poetry and pre-downloads
# example datasets so that development can continue without internet access.
# It is idempotent and safe to run multiple times.

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
POETRY_VERSION="1.8.2"

# Ensure required system packages are installed
sudo apt-get update
sudo apt-get install -y python3-venv git curl build-essential

# Create Python virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Upgrade pip and install Poetry if needed
pip install --upgrade pip
if ! command -v poetry >/dev/null 2>&1; then
    pip install "poetry==${POETRY_VERSION}"
fi

# Configure Poetry to use the in-project virtualenv
poetry config virtualenvs.create false --local || true

# Install project dependencies including development extras
poetry install --with dev,viz,extras

# Pre-download MNIST and Fashion-MNIST datasets for offline use
python - <<'PY'
from torchvision import datasets
from pathlib import Path
root = Path('data')
root.mkdir(exist_ok=True)
datasets.MNIST(root, train=True, download=True)
datasets.MNIST(root, train=False, download=True)
datasets.FashionMNIST(root, train=True, download=True)
datasets.FashionMNIST(root, train=False, download=True)
print('Datasets cached in', root.resolve())
PY

# Basic verification
poetry run python -c "import ebm, torch; print('EBM', ebm.__version__, 'Torch', torch.__version__)"

cat <<'MSG'

Development environment setup complete.
Activate the virtual environment using:
  source .venv/bin/activate

Run tests with:
  poetry run pytest

MSG

