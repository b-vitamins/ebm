# Contributor Guidelines

This file contains conventions and expectations for all future work in this
repository. These guidelines apply to the entire project tree.

## Housekeeping
- Keep dependencies in `pyproject.toml` up to date.
- Maintain type hints and docstrings for new code.
- Keep modules small and focused. New features should come with tests under
  `tests/` and documentation when appropriate.

## Development Practices
- Use Python 3.11+ features and typing whenever possible.
- Run the local checks listed below before every commit.
- Prefer small, logically scoped commits rather than large sweeping changes.
- Keep the architecture modular and avoid tight coupling between packages.

## Design and Architecture
- Favor composition over inheritance when structuring new modules.
- Keep components focused on a single responsibility.
- Prefer pure functions for utilities and minimise side effects.
- Document major decisions under `docs/architecture/` as needed.

## Pre-commit Checks
Run these commands before committing:

```bash
ruff .
mypy ebm
bandit -r ebm -ll
pytest -n auto
```

The commit should only proceed once these checks succeed.

## Commit Message Format
- Use the form `type: short description` for the subject line.
- Accepted `type` values: `feat`, `fix`, `docs`, `style`, `refactor`,
  `test`, `chore`.
- Keep the first line under 72 characters and use the imperative mood.
- Separate the subject from the body with a blank line when a body is
  provided.

## Commit Sequencing
- Do not mix unrelated changes in a single commit.
- Each commit should leave the repository in a working state.
- Include relevant tests in the same commit as the code change.

## Pull Request Messages
Include the following sections in every PR description:

```
## Summary
- High level bullet list of changes

## Testing
- Commands run and their results
```

Reference code lines using GitHub's line linking when summarising.

## CHANGELOG Maintenance
- Maintain a `CHANGELOG.md` at the project root following
  [Keep a Changelog](https://keepachangelog.com/) style.
- Add entries under an `Unreleased` section for every meaningful change.
- Bump the version in `pyproject.toml`, `ebm/__init__.py` and `setup.py` when
  releasing a new version.
- Follow [Semantic Versioning](https://semver.org/) for all releases.

