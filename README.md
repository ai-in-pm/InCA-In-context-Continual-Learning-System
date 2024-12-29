# InCA (In-context Continual Learning) System

An advanced continual learning system that combines multiple LLM backends with an External Continual Learner (ECL) for improved classification and adaptation capabilities.

The development of this Github Repository was inspired by the "In-context Continual Learning Assisted by an External Continual Learner" paper. The read the full paper, visit https://arxiv.org/pdf/2412.15563v1

## Features

- Multi-LLM Integration (GPT-4, Claude, Mistral, Groq, Gemini)
- Gaussian-based External Continual Learner
- Dynamic class distribution modeling
- Mahalanobis distance-based classification
- Ensemble learning capabilities

## Installation

### Automatic Installation (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd inca_agent
```

2. Run the installation script:
```bash
python install.py
```

3. Activate the virtual environment:
- Windows:
  ```bash
  .\venv\Scripts\activate
  ```
- Unix/MacOS:
  ```bash
  source venv/bin/activate
  ```

4. Edit the `.env` file with your API keys:
```bash
# Use your favorite text editor
notepad .env  # Windows
nano .env     # Unix/MacOS
```

5. Verify the installation:
```bash
python verify_installation.py
```

### Manual Installation

If you prefer to install manually:

1. Clone the repository:
```bash
git clone <repository-url>
cd inca_agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Unix/MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

4. Copy the environment template:
```bash
cp .env.example .env
```

5. Edit `.env` with your API keys:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
MISTRAL_API_KEY=your_mistral_key
GROQ_API_KEY=your_groq_key
```

6. Verify the installation:
```bash
python verify_installation.py
```

### Troubleshooting

If you encounter any issues:

1. Check Python version (requires 3.8 or higher):
```bash
python --version
```

2. Verify all dependencies are installed:
```bash
pip list
```

3. Ensure all API keys are set in `.env`

4. Run the verification script for detailed diagnostics:
```bash
python verify_installation.py
```

For more detailed troubleshooting, see the [Troubleshooting Guide](documentation/troubleshooting.md).

## Usage

### Quick Start

1. Start Python and import required modules:
```python
from src.inca_agent import InCAAgent
from src.llm_integration.gpt4_handler import GPT4Handler
from src.utils.active_learning import ActiveLearner
from src.utils.drift_detector import DriftDetector
from src.utils.performance_optimizer import PerformanceOptimizer
```

2. Initialize the InCA system:
```python
# Initialize LLM handlers
llm_handlers = {
    'gpt4': GPT4Handler()  # Uses OPENAI_API_KEY from .env
}

# Create InCA agent
agent = InCAAgent(llm_handlers)

# Optional: Initialize additional components
active_learner = ActiveLearner()
drift_detector = DriftDetector()
optimizer = PerformanceOptimizer()
```

3. Add classes to the system:
```python
# Add classes with examples
agent.add_class(
    class_name="technical",
    description="Technical content related to software and systems",
    examples=[
        "API documentation for REST endpoints",
        "Database optimization techniques",
        "System architecture design patterns"
    ]
)

agent.add_class(
    class_name="business",
    description="Business-related content",
    examples=[
        "Quarterly financial report analysis",
        "Market strategy overview",
        "Customer acquisition metrics"
    ]
)
```

4. Classify text:
```python
# Single LLM classification
result = agent.classify("Optimize database indexing for better performance")
print("Classification results:", result)

# Ensemble classification (if multiple LLMs configured)
results = {}
for llm_name in llm_handlers:
    results[llm_name] = agent.classify(text, llm_name=llm_name)
print("Ensemble results:", results)
```

### Advanced Usage

#### Active Learning
```python
# Get unlabeled candidates
candidates = [
    {
        'text': 'New technical document to classify',
        'confidences': agent.classify('New technical document to classify'),
        'embedding': agent.get_embeddings('New technical document to classify')
    }
]

# Select most informative samples
selected = active_learner.select_samples(candidates, n_samples=1)
print("Selected for labeling:", selected[0]['text'])
print("Uncertainty score:", selected[0]['uncertainty'])
```

#### Drift Detection
```python
# Update reference data
reference_texts = [
    "Technical documentation example",
    "System architecture review",
    "Database optimization guide"
]
reference_embeddings = [
    agent.get_embeddings(text) for text in reference_texts
]
drift_detector.update_reference_window('technical', reference_embeddings)

# Check for drift
current_texts = [
    "Code review feedback",
    "Performance optimization needed",
    "Technical debt assessment"
]
current_embeddings = [
    agent.get_embeddings(text) for text in current_texts
]
drift_metrics = drift_detector.detect_drift('technical', current_embeddings)

if drift_metrics:
    print(f"Drift detected! Score: {drift_metrics.drift_score}")
    print(f"Severity: {drift_metrics.severity}")
    
    # Get recommendations
    recommendations = drift_detector.get_drift_recommendations(drift_metrics)
    for rec in recommendations:
        print(f"- {rec}")
```

#### Performance Optimization
```python
# Enable caching and batch processing
optimizer = PerformanceOptimizer()

# Process multiple texts efficiently
texts = [
    "First document to classify",
    "Second document to classify",
    "Third document to classify"
]
results = optimizer.batch_process(texts, agent.classify)

# Monitor performance
metrics = optimizer.get_optimization_metrics()
print(f"Cache Hit Rate: {metrics.cache_hit_rate:.2f}")
print(f"Average Response Time: {metrics.avg_response_time:.2f}s")
print(f"Memory Usage: {metrics.memory_usage_mb:.2f}MB")
```

### Running from Command Line

1. Basic Classification:
```bash
python -m src.cli.classify "Text to classify"
```

2. Batch Classification:
```bash
python -m src.cli.batch_classify input.txt output.csv
```

3. System Monitoring:
```bash
python -m src.cli.monitor --interval 60
```

4. Interactive Mode:
```bash
python -m src.cli.interactive
```

### Using Jupyter Notebooks

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to demonstration notebooks:
```bash
cd demonstration/notebooks
```

3. Open and run notebooks:
- `01_system_setup.ipynb`: Basic setup and configuration
- `02_advanced_classification.ipynb`: Advanced classification techniques
- `03_performance_analysis.ipynb`: Performance monitoring and optimization
- `04_advanced_features.ipynb`: Active learning and drift detection

### Configuration

The system can be configured through environment variables in `.env`:

```bash
# Core Settings
PRIMARY_LLM=gpt4
ENABLE_ENSEMBLE=true
BATCH_SIZE=32

# Feature Flags
ENABLE_ACTIVE_LEARNING=true
ENABLE_DRIFT_DETECTION=true
ENABLE_PERFORMANCE_MONITORING=true

# Performance Settings
CACHE_SIZE=1000
MAX_WORKERS=4
ENABLE_GPU=false
```

For detailed configuration options, see the [Configuration Guide](documentation/configuration.md).

### Best Practices

1. **Data Quality**:
   - Provide diverse examples for each class
   - Keep class distributions balanced
   - Regularly update training examples

2. **Performance**:
   - Use batch processing for multiple texts
   - Enable caching for repeated operations
   - Monitor system metrics

3. **Maintenance**:
   - Regularly check for concept drift
   - Update class distributions as needed
   - Monitor API usage and quotas

For more detailed usage examples and best practices, see the [API Documentation](documentation/API.md).

## Directory Structure

```
inca_agent/
├── src/
│   ├── llm_integration/
│   ├── ecl/
│   ├── prompt_management/
│   └── utils/
├── tests/
├── demonstration/
├── documentation/
└── config/
```

## Testing

Run tests using pytest:
```bash
pytest tests/
```

## Performance Metrics

- Response time: <2 seconds
- Memory usage: <8GB
- Classification accuracy: >90%
- Scalability: Supports 1000+ classes

## Contributing

We welcome contributions to the InCA system! Here's how you can help:

1. **Fork the Repository**
   - Click the Fork button in the top right of this page
   - Clone your fork locally

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Write clear, commented code
   - Follow our coding standards
   - Add tests for new features
   - Update documentation

4. **Test Your Changes**
   ```bash
   # Run all tests
   pytest
   
   # Check code style
   black .
   isort .
   flake8 .
   ```

5. **Submit a Pull Request**
   - Push your changes to your fork
   - Submit a pull request from your branch to our main branch
   - Describe your changes in detail

### Development Guidelines

1. **Code Style**
   - Follow PEP 8 guidelines
   - Use type hints
   - Write docstrings (Google style)
   - Keep functions focused and small

2. **Testing**
   - Write unit tests for new features
   - Maintain test coverage
   - Test edge cases
   - Add integration tests when needed

3. **Documentation**
   - Update relevant documentation
   - Add docstrings to new code
   - Include usage examples
   - Update API documentation

4. **Commit Messages**
   - Write clear, descriptive commit messages
   - Reference issues and pull requests
   - Use present tense ("Add feature" not "Added feature")

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
