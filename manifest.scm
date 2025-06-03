(specifications->manifest
 '(
   "git"
   "bash"
   "poetry"
   "python"
   "python-accelerate"
   "python-pytorch-cuda"
   "python-torchvision-cuda"
   "python-matplotlib"
   "python-numpy@1"
   "python-tqdm"
   "python-pydantic"
   "python-structlog"
   "python-typing-extensions"
   "python-rich"
   
   ;; viz group
   "python-seaborn"
   "python-plotly"
   "python-pillow"
   
   ;; dev dependencies
   "python-pytest"
   "python-pytest-cov"
   "python-pytest-xdist"
   "python-black"
   "python-isort"
   "python-flake8"
   "python-flake8-docstrings"
   "python-mypy"
   "python-ipython"
   "python-nbconvert"
   
   ;; docs dependencies
   "python-sphinx"
   "python-sphinx-rtd-theme"
   "python-sphinx-autodoc-typehints"
   "python-nbsphinx"
   "python-myst-parser"
   
   ;; extras dependencies
   "python-wandb"
   "python-tensorboard"
   "python-scikit-learn"
   "python-pandas"
   "python-ipywidgets"
   "python-pyyaml"
   
   ;; Additional useful packages from original manifest
   "python-einops"
   "python-memory-profiler"
   "python-scipy"
   "python-psutil"
   "python-pytest-benchmark"
   "python-mutmut"
   "python-ruff"
   "python-types-pyyaml"
   "python-lsp-server"
   "python-pylsp-mypy"
   "python-pylsp-ruff"))