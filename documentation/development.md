# InCA System Development Guide

## Development Environment Setup

### Prerequisites
1. Python 3.8+
2. Git
3. Virtual Environment
4. Code Editor (VS Code recommended)
5. API Keys for LLM services

### Initial Setup
```bash
# Clone repository
git clone <repository-url>
cd inca_agent

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
.\venv\Scripts\activate
# Unix/MacOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Setup environment
cp .env.example .env
# Edit .env with your API keys
```

### Development Tools
```bash
# Install development tools
pip install black isort flake8 mypy pytest pytest-cov

# Format code
black .
isort .

# Lint code
flake8 .
mypy .

# Run tests
pytest
pytest --cov=src tests/
```

## Project Structure

```
inca_agent/
├── src/
│   ├── __init__.py
│   ├── inca_agent.py          # Main agent class
│   ├── llm_integration/       # LLM handlers
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── gpt4_handler.py
│   │   └── ...
│   └── utils/                 # Utility modules
│       ├── __init__.py
│       ├── active_learning.py
│       ├── drift_detector.py
│       └── performance_optimizer.py
├── tests/
│   ├── unit_tests/
│   │   ├── test_inca_agent.py
│   │   └── ...
│   └── integration_tests/
│       └── test_end_to_end.py
├── documentation/
│   ├── API.md
│   ├── configuration.md
│   └── development.md
├── demonstration/
│   └── notebooks/
│       ├── 01_system_setup.ipynb
│       └── ...
├── .env.example
├── requirements.txt
├── setup.py
└── README.md
```

## Development Guidelines

### Code Style
1. Follow PEP 8 guidelines
2. Use type hints
3. Write docstrings (Google style)
4. Keep functions focused and small
5. Use meaningful variable names

Example:
```python
from typing import List, Dict, Optional

def process_text(
    text: str,
    options: Optional[Dict[str, any]] = None
) -> List[float]:
    """
    Process input text and return embeddings.

    Args:
        text: Input text to process
        options: Optional processing parameters

    Returns:
        List of embedding values

    Raises:
        ValueError: If text is empty
    """
    if not text:
        raise ValueError("Input text cannot be empty")
    
    options = options or {}
    # Process text...
    return embeddings
```

### Testing

#### Unit Tests
```python
# test_inca_agent.py
import pytest
from src.inca_agent import InCAAgent

def test_classification():
    """Test basic classification functionality."""
    agent = InCAAgent(...)
    result = agent.classify("test text")
    assert isinstance(result, dict)
    assert all(0 <= score <= 1 for score in result.values())

def test_invalid_input():
    """Test handling of invalid input."""
    agent = InCAAgent(...)
    with pytest.raises(ValueError):
        agent.classify("")
```

#### Integration Tests
```python
# test_end_to_end.py
def test_complete_workflow():
    """Test complete classification workflow."""
    agent = InCAAgent(...)
    
    # Add class
    agent.add_class("test", "Test class", ["example"])
    
    # Classify text
    result = agent.classify("test text")
    
    # Check drift
    drift = agent.check_drift("test")
    
    assert result["test"] > 0.5
    assert drift is not None
```

### Error Handling

1. Use specific exception types
2. Provide helpful error messages
3. Log errors appropriately
4. Handle API errors gracefully

Example:
```python
class InCAError(Exception):
    """Base exception for InCA system."""
    pass

class ClassificationError(InCAError):
    """Error during text classification."""
    pass

def classify_text(text: str) -> Dict[str, float]:
    """Classify text with error handling."""
    try:
        # Attempt classification
        return perform_classification(text)
    except APIError as e:
        logger.error(f"API error during classification: {e}")
        raise ClassificationError(f"Classification failed: {e}")
    except Exception as e:
        logger.exception("Unexpected error during classification")
        raise ClassificationError(f"Unexpected error: {e}")
```

### Logging

1. Use structured logging
2. Include relevant context
3. Use appropriate log levels
4. Configure logging properly

Example:
```python
import logging

logger = logging.getLogger(__name__)

def process_request(text: str, options: Dict):
    """Process request with logging."""
    logger.info("Processing request", extra={
        "text_length": len(text),
        "options": options
    })
    
    try:
        result = perform_processing(text, options)
        logger.debug("Processing successful", extra={
            "result_size": len(result)
        })
        return result
    except Exception as e:
        logger.error("Processing failed", extra={
            "error": str(e),
            "text_length": len(text)
        })
        raise
```

### Performance Optimization

1. Use caching effectively
2. Implement batch processing
3. Optimize memory usage
4. Monitor performance metrics

Example:
```python
from functools import lru_cache
from typing import List

class EmbeddingProcessor:
    """Process and cache embeddings."""
    
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
    
    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> List[float]:
        """Get cached embedding for text."""
        return compute_embedding(text)
    
    def batch_process(self, texts: List[str]) -> List[List[float]]:
        """Process texts in batches."""
        return [
            self.get_embedding(text)
            for text in texts
        ]
```

## Contributing

### Pull Request Process

1. Create feature branch
2. Write tests
3. Update documentation
4. Run linters and formatters
5. Submit pull request

Example workflow:
```bash
# Create branch
git checkout -b feature/new-feature

# Make changes
# Write tests
# Update docs

# Format and lint
black .
isort .
flake8 .
mypy .

# Run tests
pytest

# Commit and push
git add .
git commit -m "Add new feature"
git push origin feature/new-feature
```

### Code Review Guidelines

1. Check code style
2. Verify test coverage
3. Review documentation
4. Test functionality
5. Consider performance
6. Look for security issues

## Deployment

### Preparation
1. Update version numbers
2. Run full test suite
3. Update documentation
4. Create release notes

### Release Process
1. Tag release
2. Build distribution
3. Update production
4. Monitor deployment

Example:
```bash
# Update version
# Edit setup.py

# Create tag
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Build
python setup.py sdist bdist_wheel

# Deploy
# Follow deployment procedures
```

## Support

For development support:
1. Check documentation
2. Review issue tracker
3. Contact development team
4. Join developer forums
