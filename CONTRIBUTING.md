# Contributing

Thank you for improving this research artifact. Contributions should preserve the study's
device-held-out evaluation boundary and its reproducibility guarantees.

## Development setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Before opening a pull request

Run the complete lightweight validation suite:

```bash
ruff format --check src tests scripts
ruff check src tests scripts
mypy src
bandit -q -r src scripts -ll
pytest
```

Please keep pull requests focused and include:

- the problem being addressed;
- the files and evaluation stages affected;
- tests or verification commands;
- any change to data inclusion, partitioning, preprocessing, metrics, or statistical targets.

Do not commit raw traffic, credentials, licensed papers, model binaries, row-level prediction
exports, local environments, or pre-release manuscript archives. If a change affects a scientific
claim or evaluation population, open an issue for discussion before implementing it.
