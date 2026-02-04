# Contributing to AtlasVLA

Thank you for your interest in contributing to AtlasVLA! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/AtlasVLA.git
   cd AtlasVLA
   ```
3. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** and test them

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```
   
   Please write clear commit messages following conventional commits format:
   - `feat: add new feature`
   - `fix: fix bug in training`
   - `docs: update README`
   - `refactor: restructure code`
   - `test: add tests for X`

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** on GitHub

## Code Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and small

## Testing

Before submitting a PR, please:
- Run the import tests: `python atlas/test_imports.py`
- Ensure your code doesn't break existing functionality
- Add tests for new features if applicable

## Pull Request Guidelines

- Provide a clear description of your changes
- Reference any related issues
- Ensure all tests pass
- Update documentation if needed
- Keep PRs focused on a single feature or fix

## Questions?

Feel free to open an issue for questions or discussions!
