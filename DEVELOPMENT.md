# Development Guide

This document outlines development workflows, testing strategies, and best practices for the EBM project.

## Getting Started

The project uses direnv for environment management and Guix for package management. Upon entering the project directory, direnv automatically:

- Loads the Guix environment from `manifest.scm`
- Sets up Python optimization flags
- Configures GPU settings
- Adds project `bin/` scripts to PATH
- Installs pre-commit hooks if missing

You'll see:
```
✓ Pre-commit hooks installed
```

These commands are available in the `bin/` directory and automatically added to your PATH.

## Quick Command Reference

### Testing Commands
- `t` - Run fast tests only (~seconds)
- `tf` - Run full test suite (including slow tests)
- `ts` - Run only statistical tests
- `tc` - Run fast tests with coverage report
- `tcf` - Run full test suite with coverage
- `test` - Smart test runner (detects changed files)
- `tgpu` - Run GPU tests (if GPU available)

### Quality Assurance
- `qa` - Quick quality check (lint + type check + fast tests)
- `qaf` - Full quality assurance (all checks + all tests)
- `rufffix` - Auto-fix code style issues

### Pre-commit Hooks
- `pc` - Run pre-commit on staged files
- `pca` - Run pre-commit on all files
- `pcf` - Run pre-commit with full test suite (manual stage)

### Development Tools
- `dev` - Project status dashboard (git, coverage, environment)
- `nb` - Launch Jupyter Lab/Notebook
- `profile` - Profile Python scripts for performance

### Utilities
- `clean` - Remove Python cache files
- `cov` - Open HTML coverage report in browser

## Development Workflows

### Daily Development

Start your development session:
```bash
cd ebm               # direnv loads environment
qa                   # Verify everything works
git checkout -b feature/my-feature
```

During development:
```bash
# Edit code...
t                    # Quick test after changes
t tests/test_specific.py::test_function  # Test specific function
rufffix              # Fix style issues
```

Before committing:
```bash
qa                   # Full quick check
git add .
pc                   # Pre-commit on staged files
git commit -m "feat: add new feature"
```

### Bug Fixing

Reproduce and fix bugs efficiently:
```bash
# Reproduce the bug
t tests/test_module.py -k test_failing -v

# Debug interactively
pytest tests/test_module.py::test_failing -v --pdb

# Fix the code...

# Verify fix
t tests/test_module.py
ts                   # Check statistical tests if relevant
tf                   # Full suite before pushing
```

### Feature Development

Develop new features systematically:
```bash
git checkout -b feature/new-sampler

# Write code incrementally
mypy ebm/samplers/new_sampler.py    # Type check as you go
t                                   # Fast feedback

# Add tests
# Write tests in tests/test_new_sampler.py
t tests/test_new_sampler.py -v

# Check coverage
tc tests/test_new_sampler.py --cov=ebm.samplers.new_sampler
cov                                 # View coverage gaps

# Final checks before PR
qa                                  # Quick quality check
pca                                 # Full pre-commit
```

### Pre-Release Checklist

Before tagging a release:
```bash
# 1. Full quality assurance
qaf                  # All tests + checks

# 2. Coverage analysis
tcf
cov                  # Ensure good coverage

# 3. Clean up
clean               # Remove cache files
rufffix            # Auto-fix any style issues

# 4. Final validation
pcf                 # Full pre-commit with all tests

# 5. Tag release
git tag -a v0.2.0 -m "Release 0.2.0"
git push origin v0.2.0
```

### Performance Testing

Monitor and improve performance:
```bash
# Benchmark current implementation
pytest -m performance --benchmark-only

# Profile slow tests
pytest --durations=10

# After optimization, compare
pytest -m performance --benchmark-compare

# Memory profiling
pytest -m "not slow" --memray
```

### Coverage Investigation

Improve test coverage:
```bash
# Quick coverage check
tc

# Detailed missing coverage
t --cov=ebm --cov-report=term-missing

# Module-specific coverage
t tests/test_module.py --cov=ebm.module --cov-report=html
cov

# Full coverage analysis
tcf
```

### Using Development Tools

**Project Dashboard (`dev`)**
```bash
dev  # Shows git status, coverage, test results, environment info
```

**Jupyter Notebooks (`nb`)**
```bash
nb          # Launch Jupyter Lab on port 8888
nb notebook # Launch classic notebook
nb lab 8889 # Launch on specific port
```

**Performance Profiling (`profile`)**
```bash
profile script.py       # Profile a script
profile -m module       # Profile a module
profile script.py args  # Profile with arguments
```

**Smart Testing (`test`)**
```bash
test                   # Auto-detects changed files and runs relevant tests
test tests/specific.py # Run specific test file
```

**GPU Testing (`tgpu`)**
```bash
tgpu              # Run GPU tests if available
tgpu -v           # Verbose GPU tests
```

## Test Organization

Tests are organized by speed and computational intensity:

```python
# Fast test (default)
def test_quick_validation():
    pass

# Slow computational test
@pytest.mark.slow
def test_complex_computation():
    pass

# Statistical test with many iterations
@pytest.mark.statistical
def test_markov_chain_convergence():
    pass
```

### Test Categories

1. **Fast Tests** (default)
   - Unit tests, quick validations
   - Run in seconds
   - Used for rapid development feedback

2. **Slow Tests** (`@pytest.mark.slow`)
   - Integration tests, complex computations
   - May take minutes
   - Run before commits

3. **Statistical Tests** (`@pytest.mark.statistical`)
   - Monte Carlo simulations, convergence tests
   - Can take many minutes
   - Run before releases

## CI/CD Pipeline

The GitHub Actions pipeline runs:

1. **Quick Check** (on all pushes)
   - Pre-commit hooks validation
   - Linting (ruff)
   - Type checking (mypy on ebm/ only)
   - Fast tests only
   - Runs on Python 3.11

2. **Full Test** (needs quick-check to pass)
   - All tests including slow/statistical
   - Coverage reporting with codecov
   - Matrix testing (Python 3.11 and 3.12)
   - Timeout after 30 minutes

### Debugging CI Failures

Reproduce CI environment locally:
```bash
# Fresh shell without local config
env -i bash --noprofile --norc

# Load project environment
source .envrc

# Run same commands as CI
qa                   # Should match CI behavior
```

## Code Quality Standards

### Style Guide
- Follow PEP 8 (enforced by ruff)
- Line length: 100 characters
- Use NumPy-style docstrings

### Type Hints
- All public functions must have type hints
- Use `mypy --strict` for validation
- Currently only checking `ebm/` directory (not tests)
- No `Any` types without justification

### Testing Requirements
- All new features need tests
- Maintain >90% coverage on fast tests
- Statistical tests for probabilistic components

## Project Structure

```
ebm/
├── bin/            # Development command scripts
├── ebm/            # Main package
│   ├── __init__.py
│   ├── rbm/        # RBM implementations
│   ├── samplers/   # MCMC samplers
│   └── utils/      # Utilities
├── tests/          # Test suite
├── docs/           # Documentation
├── .envrc          # direnv configuration
├── pyproject.toml  # Poetry/project config
└── DEVELOPMENT.md  # This file
```

## Troubleshooting

### Commands not found
```bash
direnv allow        # Reload environment
direnv reload       # Force reload
```

### Test failures
```bash
# Clean environment
clean
rm -rf .pytest_cache

# Verbose output
t -vv

# Debug mode
pytest --pdb tests/test_failing.py
```

### Type checking issues
```bash
# Clear mypy cache
rm -rf .mypy_cache

# Check specific file
mypy --strict ebm/module.py
```

## Advanced Workflows

### Memory Profiling
```bash
pip install memray
pytest --memray tests/
```

### Profiling Specific Functions
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# ... code to profile ...
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Debugging Statistical Tests
```bash
# Set random seed for reproducibility
PYTHONHASHSEED=0 pytest tests/test_statistical.py

# Run with specific seed
pytest tests/test_statistical.py --randomly-seed=42
```

---

*Remember: Use `qa` frequently during development for quick feedback, and `tf` before pushing for complete validation.*
