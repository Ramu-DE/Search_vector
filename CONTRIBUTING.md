# Contributing to AI-Powered Search

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Search_vector.git
   cd Search_vector
   ```

3. **Set up development environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

5. **Make your changes** and test them

6. **Commit your changes**:
   ```bash
   git commit -m "Add: description of your changes"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** on GitHub

## 📋 How to Contribute

### Reporting Bugs

**Before submitting a bug report:**
- Check if the issue already exists in [Issues](https://github.com/Ramu-DE/Search_vector/issues)
- Ensure you're using the latest version
- Test with the minimal example to isolate the issue

**When creating a bug report, include:**
- Clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Environment details:
  - Python version (`python --version`)
  - OS and version
  - Dependency versions (`pip freeze`)
- Error messages and stack traces
- Code snippet to reproduce (if applicable)

**Template:**
```markdown
**Description:**
Brief description of the bug

**To Reproduce:**
1. Step one
2. Step two
3. Error occurs

**Expected behavior:**
What should happen

**Environment:**
- Python version: 3.11
- OS: Ubuntu 22.04
- Dependencies: (relevant package versions)

**Error message:**
```
Paste error here
```
```

### Suggesting Features

**Before suggesting a feature:**
- Check if it's already been suggested in [Issues](https://github.com/Ramu-DE/Search_vector/issues)
- Consider if it fits the project scope

**When suggesting a feature, include:**
- Clear, descriptive title
- Problem it solves
- Proposed solution
- Alternative solutions considered
- Example use case

### Contributing Code

**Good first issues:**
- Look for issues labeled `good first issue` or `help wanted`
- Documentation improvements
- Adding examples
- Fixing typos or clarifying comments

**Code guidelines:**

1. **Python Style:**
   - Follow [PEP 8](https://pep8.org/)
   - Use type hints for function signatures
   - Maximum line length: 100 characters
   - Use meaningful variable names

2. **Code Structure:**
   ```python
   def function_name(param1: str, param2: int = 5) -> bool:
       """
       Brief description of what the function does.
       
       Args:
           param1: Description of param1
           param2: Description of param2 (default: 5)
           
       Returns:
           Description of return value
           
       Example:
           >>> function_name("test", 10)
           True
       """
       # Implementation here
       return True
   ```

3. **Documentation:**
   - Add docstrings to all public functions and classes
   - Update relevant guides if behavior changes
   - Add comments for complex logic
   - Update README if adding new features

4. **Testing:**
   - Add tests for new features
   - Ensure existing tests pass
   - Run tests locally:
     ```bash
     python -m pytest tests/
     ```
   - Aim for >80% code coverage for new code

5. **Commit Messages:**
   - Use present tense ("Add feature" not "Added feature")
   - Use imperative mood ("Move cursor to..." not "Moves cursor to...")
   - Start with a type prefix:
     - `Add:` new feature
     - `Fix:` bug fix
     - `Update:` changes to existing feature
     - `Remove:` removal of feature/code
     - `Docs:` documentation only
     - `Test:` adding or updating tests
     - `Refactor:` code change that neither fixes nor adds
   
   **Examples:**
   ```
   Add: sparse encoding with term expansion
   Fix: cosine similarity calculation for zero vectors
   Update: hybrid search weights for e-commerce use case
   Docs: clarify distance metrics selection guide
   ```

6. **Pull Request Guidelines:**
   - One feature/fix per PR
   - Keep PRs focused and reasonably sized (<500 lines if possible)
   - Link related issues (#123)
   - Update documentation
   - Add/update tests
   - Ensure CI passes

### Contributing Documentation

Documentation improvements are always welcome!

**Types of documentation contributions:**
- Fixing typos or grammar
- Clarifying existing explanations
- Adding examples
- Creating tutorials
- Improving visualizations
- Translating documentation

**Documentation style:**
- Use clear, concise language
- Include code examples where appropriate
- Add visualizations for complex concepts
- Link to related sections
- Test all code examples

### Contributing Examples

**When adding examples:**
- Create in `examples/` directory
- Include a README explaining what it demonstrates
- Add requirements if dependencies needed
- Keep examples self-contained
- Add comments explaining key concepts

**Example structure:**
```python
"""
Example: Dense Vector Search with AWS Bedrock

Demonstrates:
- Initializing Bedrock embeddings
- Indexing documents
- Performing semantic search

Requirements:
- AWS credentials configured
- boto3 installed
"""

# Example code here
```

## 🧪 Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/Ramu-DE/Search_vector.git
cd Search_vector

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install pytest pytest-cov black ruff mypy
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_embeddings.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test
pytest tests/test_search.py::test_cosine_similarity
```

### Code Formatting

```bash
# Format code with black (optional)
black .

# Check code style with ruff (optional)
ruff check .
```

### Testing Before Submitting PR

**Checklist:**
- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] Docstrings added for new functions
- [ ] Examples tested and working
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] No sensitive data in commits (API keys, credentials)

## 🔒 Security

If you discover a security vulnerability:
- **DO NOT** open a public issue
- Email details privately to: [your-email]
- Include steps to reproduce
- Allow reasonable time for fix before public disclosure

## 📜 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 💬 Communication

- **Issues:** Bug reports and feature requests
- **Pull Requests:** Code contributions
- **Discussions:** Questions and general discussions (if enabled)

## 🎯 Project Scope

This project focuses on:
- Production-ready search implementations
- AWS integrations (Bedrock, OpenSearch)
- Educational documentation and visualizations
- Performance benchmarking

Out of scope:
- Non-AWS cloud providers (Azure, GCP) - unless community-contributed
- UI/frontend implementations - focus is on backend/core logic
- Non-Python implementations

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md (if we create it)
- Credited in release notes
- Mentioned in relevant documentation

## 📝 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## ❓ Questions?

- Open a [Discussion](https://github.com/Ramu-DE/Search_vector/discussions) (if enabled)
- Open an [Issue](https://github.com/Ramu-DE/Search_vector/issues) with `question` label
- Check existing documentation

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!

---

**Happy Contributing!** 🚀
