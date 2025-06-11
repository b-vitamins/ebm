# Development Guidelines for the EBM Project

This document defines standards for automated agents contributing to this
repository.

## Commit Message Standards
- Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
- Start the subject with a **type** (`feat`, `fix`, `chore`, `docs`, `test`, etc.).
- Keep the subject line under 72 characters.
- Provide a body when the change is not trivial.

## Commit Sequencing
- Group related changes into atomic commits.
- Run linting and tests before every commit.

## Pull Request Standards
- Title should use Conventional Commits style.
- Describe the purpose of the PR, testing instructions, and any context.

## Code Housekeeping
- Keep dependencies up to date.
- Remove dead code when encountered.
- Track performance or security issues via TODO comments referencing issues.

## Architecture and Design
- Document significant decisions in `docs/adr/` using the
  [ADR](https://adr.github.io/) format.
- Keep modules focused; avoid cross‑package imports where possible.

## Pre-commit Checks
- `ruff --fix` for linting and formatting.
- `mypy` for type checking.
- `pytest` for the test suite.
- `bandit -r ebm` for security checks.

## Version Management
- Follow [Semantic Versioning](https://semver.org/).
- Bump `pyproject.toml` version on release commits.

## CHANGELOG Maintenance
- Add an entry under `Unreleased` for every user facing change.
- Use the categories: Added, Changed, Deprecated, Removed, Fixed, Security.

## Testing Standards
- Organize tests under `tests/` by type (unit, integration, etc.).
- Aim for high coverage of critical utilities.

## Documentation Standards
- Update `README.md` when public APIs change.
- Keep docstrings in NumPy style.


