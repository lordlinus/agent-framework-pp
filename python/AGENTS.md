# Python Agent Framework - Development Guide

## Dev environment tips
- Use `cd python` to navigate to the Python workspace root before running any commands.
- The Python workspace should be rooted in the `./python` folder for VSCode.
- Use `uv run poe setup` to set up the development environment with virtual environment, dependencies, and pre-commit hooks.
- Run `uv run poe install` to install all dependencies including extras and dev dependencies.
- Use `uv run poe venv --python 3.12` to create or switch to a specific Python version.
- Run `uv sync --dev` to sync dependencies after pulling new changes.
- Use `source .venv/bin/activate` (Linux/macOS) to activate the virtual environment.
- Install Python 3.10, 3.11, 3.12, or 3.13 with `uv python install 3.10 3.11 3.12 3.13`.
- Check the `pyproject.toml` file in the python folder to confirm package dependencies and configuration.
- Use `uv run poe` to discover all available Poe tasks for development automation.

## Testing instructions
- Find the CI plan in the `.github/workflows` folder (see `python-merge-tests.yml` and `python-code-quality.yml`).
- Run `uv run poe test` to run all unit tests with coverage from the python directory.
- Run `uv run poe all-tests` to run all tests including integration tests.
- Use `uv run pytest tests/test_specific.py` to run a specific test file.
- To focus on one test, use: `uv run pytest tests/test_file.py::test_name` or `uv run pytest -k "test_name"`.
- Run `uv run pytest -v` for verbose test output.
- The commit should pass all tests before you merge.
- Run `uv run poe check` to run all code quality checks (same as CI runs).
- Fix any test or type errors until the whole suite is green.
- Add or update tests for the code you change, even if nobody asked.
- Target minimum 80% test coverage for all packages.

## PR instructions
- Title format: `[Python] <Title>` or `[<package_name>] <Title>` for specific packages.
- Always run `uv run poe fmt` to format code before committing.
- Always run `uv run poe lint` to check and fix linting issues before committing.
- Always run `uv run poe check` to run all code quality checks before committing.
- Always run `uv run poe test` to ensure tests pass before committing.
- Pre-commit hooks will automatically run checks on commit if installed with `uv run poe pre-commit-install`.
- Use Google-style docstrings for all public APIs.
- Follow the flat import structure (core from `agent_framework`, components from `agent_framework.<component>`).
- Ensure type hints are included and pass `uv run poe mypy` and `uv run poe pyright` checks.
