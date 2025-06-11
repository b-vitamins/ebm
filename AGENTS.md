# Contributor Guidelines

This document defines standards and conventions for all contributors to this repository, including automated agents.

## Commit Message Standards
- Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
- Format: `type: short description`
- Accepted types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- Keep the subject line under 72 characters
- Use imperative mood (e.g., "Add feature" not "Added feature")
- Provide a body when the change is not trivial, separated by a blank line

## Commit Sequencing
- Group related changes into atomic commits
- Each commit should leave the repository in a working state
- Include relevant tests in the same commit as the code change
- Do not mix unrelated changes in a single commit
- Run linting and tests before every commit

## Pull Request Standards
- Title should use Conventional Commits style
- Include the following sections in every PR description:
```
## Summary
- High level bullet list of changes

## Testing
- Commands run and their results
```
- Reference code lines using GitHub's line linking when summarizing
- Describe the purpose of the PR, testing instructions, and any context

## Pre-commit Checks
Run these commands before committing:
```bash
ruff .
mypy ebm
bandit -r ebm -ll
pytest -n auto
```
The commit should only proceed once these checks succeed.

## Code Standards
- Use Python 3.11+ features and typing whenever possible
- Maintain type hints and docstrings for new code
- Keep docstrings in NumPy style
- Prefer small, logically scoped commits
- Keep modules small and focused on a single responsibility
- Prefer pure functions for utilities and minimize side effects

## Architecture and Design
- Document significant decisions in `docs/adr/` using the [ADR](https://adr.github.io/) format
- Keep the architecture modular and avoid tight coupling between packages
- Favor composition over inheritance when structuring new modules
- Avoid cross-package imports where possible

## Code Housekeeping
- Keep dependencies in `pyproject.toml` up to date
- Remove dead code when encountered
- Track performance or security issues via TODO comments referencing issues

## Testing Standards
- Organize tests under `tests/` by type (unit, integration, etc.)
- Aim for high coverage of critical utilities
- New features should come with tests

## Documentation Standards
- Update `README.md` when public APIs change
- Keep documentation up to date with code changes

## Version Management
- Follow [Semantic Versioning](https://semver.org/)
- Bump version in `pyproject.toml`, `ebm/__init__.py`, and `setup.py` on release commits

## CHANGELOG Maintenance
- Maintain a `CHANGELOG.md` at the project root following [Keep a Changelog](https://keepachangelog.com/) style
- Add entries under `Unreleased` section for every user-facing change
- Use the categories: Added, Changed, Deprecated, Removed, Fixed, Security