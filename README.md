# InCA: In-context Continual Learning System

An advanced continual learning system that combines multiple LLM backends with an External Continual Learner (ECL) for improved classification and adaptation capabilities.

The development of this Github Repository was inspired by the "In-context Continual Learning Assisted by an External Continual Learner" paper. To read the full paper, visit [https://arxiv.org/pdf/2412.15563v1](https://arxiv.org/pdf/2412.15563v1)

## Features

- **Multi-LLM Integration**: Support for GPT-4, Claude, Mistral, Groq, and Gemini
- **External Continual Learning**: Adaptive learning through ECL component
- **Active Learning**: Smart sample selection for efficient model improvement
- **Drift Detection**: Automated detection of concept drift in data streams
- **Performance Optimization**: Advanced caching and throughput optimization
- **Visualization Tools**: Comprehensive tools for performance analysis and monitoring

## Installation

```bash
# Clone the repository
git clone https://github.com/ai-in-pm/InCA-In-context-Continual-Learning-System.git
cd InCA-In-context-Continual-Learning-System

# Install dependencies
pip install -r requirements.txt

# Run installation verification
python verify_installation.py
```

## Quick Start

```python
from src.inca_agent import InCAAgent
from src.llm_integration.gpt4_handler import GPT4Handler

# Initialize LLM handlers
llm_handlers = {
    'gpt4': GPT4Handler()
}

# Create InCA agent
agent = InCAAgent(llm_handlers)

# Add classes
agent.add_class(
    class_name="technical",
    description="Technical content",
    examples=["API documentation", "System architecture"]
)

# Classify text
result = agent.classify("Optimize database indexing")
print(result)
```

## System Architecture

The InCA system consists of several key components:

1. **LLM Integration Layer**
   - Multiple LLM backend support
   - Unified interface for all models
   - Automatic fallback handling

2. **External Continual Learner (ECL)**
   - Gaussian modeling for class distributions
   - Adaptive threshold management
   - Continuous model refinement

3. **Active Learning System**
   - Uncertainty-based sample selection
   - Diversity-aware sampling
   - Batch optimization

4. **Performance Optimization**
   - Smart caching mechanisms
   - Request batching
   - Memory management
   - Response time optimization

## Documentation

- [API Documentation](documentation/API.md)
- [Configuration Guide](documentation/configuration.md)
- [Development Guide](documentation/development.md)
- [Troubleshooting Guide](documentation/troubleshooting.md)

## Examples and Demonstrations

The `demonstration` directory contains:

- Example notebooks showing various use cases
- Visualization scripts for performance analysis
- Sample configurations and setups
- Integration examples with different LLMs

## Performance Monitoring

The system includes comprehensive monitoring tools:

- Response time tracking
- Memory usage analysis
- Cache hit rate monitoring
- Throughput optimization
- Drift detection metrics

## Contributing

We welcome contributions! Please see our [Contributing Guide](documentation/development.md) for details on:

- Code style guidelines
- Pull request process
- Development setup
- Testing requirements

## Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit_tests
pytest tests/integration_tests
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use InCA in your research, please cite:

```bibtex
@article{zhang2023context,
  title={In-context Continual Learning Assisted by an External Continual Learner},
  author={Zhang, Yiyang and Liang, Weizhe and Deng, Chenghao and Jiang, Nan and Xie, Yixuan and Pu, Yewen and Xie, Pengtao},
  journal={arXiv preprint arXiv:2412.15563},
  year={2023}
}
```

## Acknowledgments

- Thanks to the authors of the original paper for their groundbreaking research
- Special thanks to all contributors
- Built with support from the open-source community
