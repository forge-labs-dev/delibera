# Delibera

An agentic application framework.

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and requires Python 3.12 or later.

### Setup

```bash
# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Running Tests

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=delibera
```

### Code Quality

```bash
# Run linter
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Type checking
uv run mypy src

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

## Publishing

This project uses GitHub Actions for automated releases to PyPI using trusted publishing.

### Setup Trusted Publishing

Before creating a release, you need to configure trusted publishers on PyPI:

1. Go to https://pypi.org/manage/account/publishing/
2. Add a new pending publisher with:
   - PyPI Project Name: `delibera`
   - Owner: `forge-labs-dev`
   - Repository name: `delibera`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`

3. For TestPyPI, go to https://test.pypi.org/manage/account/publishing/ and repeat with environment name `testpypi`

### Creating a Release

```bash
# Tag a new version
git tag v0.1.0
git push origin v0.1.0
```

The GitHub Actions workflow will automatically:
1. Build the distribution packages
2. Publish to TestPyPI
3. Publish to PyPI (requires manual approval in GitHub)

## License

See [LICENSE](LICENSE) file for details.
