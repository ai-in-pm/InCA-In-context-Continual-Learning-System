# InCA System API Documentation

## Core Components

### InCAAgent

The main agent class that orchestrates the InCA system.

```python
class InCAAgent:
    def __init__(self, llm_handlers: Dict[str, LLMBase], primary_llm: str = "gpt4"):
        """
        Initialize the InCA agent.

        Args:
            llm_handlers: Dictionary mapping LLM names to their handler instances
            primary_llm: Name of the primary LLM to use for embeddings and classification
        """

    def add_class(self, class_name: str, description: str, examples: List[str]) -> None:
        """
        Add a new class to the system.

        Args:
            class_name: Unique identifier for the class
            description: Human-readable description of the class
            examples: List of example texts belonging to this class
        """

    def classify(self, text: str, llm_name: Optional[str] = None, top_k: int = 3) -> Dict[str, float]:
        """
        Classify input text using specified LLM or ensemble.

        Args:
            text: Input text to classify
            llm_name: Optional specific LLM to use (uses primary_llm if None)
            top_k: Number of top classes to return

        Returns:
            Dictionary mapping class names to confidence scores
        """

    def get_mahalanobis_distance(self, text: str, class_name: str) -> float:
        """
        Calculate Mahalanobis distance between text and class distribution.

        Args:
            text: Input text
            class_name: Target class name

        Returns:
            Mahalanobis distance (float)
        """
```

### LLM Integration

Base class and implementations for LLM handlers.

```python
class LLMBase:
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """
        Generate text response from LLM.

        Args:
            prompt: Input prompt text

        Returns:
            Generated response text
        """

    @abstractmethod
    def get_embeddings(self, text: str) -> List[float]:
        """
        Get vector embeddings for input text.

        Args:
            text: Input text

        Returns:
            List of embedding values
        """

    @abstractmethod
    def classify_text(self, text: str, classes: List[str]) -> Dict[str, float]:
        """
        Classify text into given classes.

        Args:
            text: Input text
            classes: List of possible class names

        Returns:
            Dictionary mapping class names to confidence scores
        """
```

### External Continual Learner

Gaussian modeling for class distributions.

```python
class GaussianModeler:
    def update_distribution(self, class_name: str, embeddings: List[float]):
        """
        Update Gaussian distribution for a class.

        Args:
            class_name: Name of the class to update
            embeddings: New embedding vector to incorporate
        """

    def compute_mahalanobis_distance(self, embeddings: List[float], class_name: str) -> float:
        """
        Compute Mahalanobis distance between embeddings and class distribution.

        Args:
            embeddings: Input embedding vector
            class_name: Target class name

        Returns:
            Mahalanobis distance (float)
        """

    def get_class_probability(self, embeddings: List[float], class_name: str) -> float:
        """
        Get probability of embeddings belonging to a class.

        Args:
            embeddings: Input embedding vector
            class_name: Target class name

        Returns:
            Probability score between 0 and 1
        """
```

### Active Learning

The active learning system helps select the most informative samples for labeling.

```python
class ActiveLearner:
    def select_samples(self, candidates: List[Dict], n_samples: int = 10) -> List[Dict]:
        """
        Select most informative samples for labeling.

        Args:
            candidates: List of candidate samples with text and confidences
            n_samples: Number of samples to select

        Returns:
            List of selected samples with uncertainty and diversity scores
        """

    def compute_uncertainty(self, confidences: Dict[str, float]) -> float:
        """
        Compute uncertainty score for a sample.

        Args:
            confidences: Dictionary mapping class names to confidence scores

        Returns:
            Uncertainty score between 0 and 1
        """

    def compute_diversity_score(self, embedding: List[float], selected_embeddings: List[List[float]]) -> float:
        """
        Compute diversity score for a sample.

        Args:
            embedding: Embedding vector of the candidate sample
            selected_embeddings: List of embeddings from already selected samples

        Returns:
            Diversity score between 0 and 1
        """
```

### Drift Detection

Monitor and detect concept drift in the classification system.

```python
class DriftDetector:
    def update_reference_window(self, class_name: str, embeddings: List[List[float]]):
        """
        Update reference window for a class.

        Args:
            class_name: Name of the class to update
            embeddings: List of embedding vectors for the reference window
        """

    def detect_drift(self, class_name: str, current_embeddings: List[List[float]]) -> Optional[DriftMetrics]:
        """
        Detect if drift has occurred for a specific class.

        Args:
            class_name: Name of the class to check
            current_embeddings: Current embedding vectors to compare against reference

        Returns:
            DriftMetrics if drift detected, None otherwise
        """

    def get_drift_statistics(self, time_window: Optional[timedelta] = None) -> Dict:
        """
        Get drift statistics over a time window.

        Args:
            time_window: Optional time window to analyze

        Returns:
            Dictionary containing drift statistics
        """

    def get_drift_recommendations(self, metrics: DriftMetrics) -> List[str]:
        """
        Get recommendations based on drift metrics.

        Args:
            metrics: Drift metrics from detection

        Returns:
            List of recommended actions
        """
```

### Performance Optimization

Optimize system performance through caching and parallel processing.

```python
class PerformanceOptimizer:
    def cached_embedding(self, text: str) -> List[float]:
        """
        Get cached embeddings for text.

        Args:
            text: Input text

        Returns:
            Cached embedding vector if available, else compute new
        """

    def batch_process(self, texts: List[str], processor_func: Any) -> List[Any]:
        """
        Process texts in batches for better throughput.

        Args:
            texts: List of input texts
            processor_func: Function to apply to each text

        Returns:
            List of processed results
        """

    def optimize_embedding_computation(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Optimize embedding vectors for memory efficiency.

        Args:
            embeddings: Input embedding vectors

        Returns:
            Optimized embedding vectors
        """

    def get_optimization_metrics(self) -> OptimizationMetrics:
        """
        Get current optimization metrics.

        Returns:
            OptimizationMetrics containing cache hit rate, response time, etc.
        """
```

## Usage Examples

### Basic Classification

```python
# Initialize agent
llm_handlers = {
    'gpt4': GPT4Handler(api_key)
}
agent = InCAAgent(llm_handlers)

# Add classes
agent.add_class(
    class_name="technical",
    description="Technical content",
    examples=["API documentation", "Code review feedback"]
)

# Classify text
results = agent.classify("The API requires authentication")
```

### Ensemble Classification

```python
# Initialize with multiple LLMs
llm_handlers = {
    'gpt4': GPT4Handler(api_key),
    'claude': ClaudeHandler(api_key),
    'mistral': MistralHandler(api_key)
}
agent = InCAAgent(llm_handlers)

# Get ensemble classification
results = {}
for llm_name in llm_handlers:
    results[llm_name] = agent.classify(text, llm_name=llm_name)
```

### Distribution Analysis

```python
# Get Mahalanobis distance
distance = agent.get_mahalanobis_distance(text, class_name)

# Get class information
info = agent.get_class_info(class_name)
```

### Active Learning

```python
# Initialize components
active_learner = ActiveLearner()
agent = InCAAgent(llm_handlers)

# Get unlabeled candidates
candidates = [
    {
        'text': 'Example text',
        'confidences': agent.classify('Example text'),
        'embedding': agent.get_embeddings('Example text')
    }
    # ... more candidates
]

# Select informative samples
selected = active_learner.select_samples(candidates, n_samples=10)

# Review selected samples
for sample in selected:
    print(f"Text: {sample['text']}")
    print(f"Uncertainty: {sample['uncertainty']:.2f}")
    print(f"Diversity Score: {sample['diversity_score']:.2f}")
```

### Drift Detection

```python
# Initialize drift detector
drift_detector = DriftDetector()

# Update reference data
reference_embeddings = [
    agent.get_embeddings(text) for text in reference_texts
]
drift_detector.update_reference_window('technical', reference_embeddings)

# Check for drift
current_embeddings = [
    agent.get_embeddings(text) for text in current_texts
]
drift_metrics = drift_detector.detect_drift('technical', current_embeddings)

if drift_metrics:
    print(f"Drift detected! Score: {drift_metrics.drift_score:.2f}")
    print(f"Severity: {drift_metrics.severity}")
    
    # Get recommendations
    recommendations = drift_detector.get_drift_recommendations(drift_metrics)
    for rec in recommendations:
        print(f"- {rec}")
```

### Performance Optimization

```python
# Initialize optimizer
optimizer = PerformanceOptimizer()

# Process batch of texts
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = optimizer.batch_process(texts, agent.get_embeddings)

# Get optimization metrics
metrics = optimizer.get_optimization_metrics()
print(f"Cache Hit Rate: {metrics.cache_hit_rate:.2f}")
print(f"Average Response Time: {metrics.avg_response_time:.2f}s")
print(f"Memory Usage: {metrics.memory_usage_mb:.2f}MB")
print(f"Throughput: {metrics.throughput:.2f} requests/second")
```

## Error Handling

The system includes comprehensive error handling:

- Invalid API keys or API errors
- Missing or invalid class names
- Embedding generation failures
- Classification errors

All errors are raised with descriptive messages and proper exception types.

## Performance Considerations

- Response time: <2 seconds for classification
- Memory usage: <8GB
- Scalability: Supports 1000+ classes
- Embedding dimension: 1536 (GPT-4 standard)
- Cache size: Configurable, default 1000 entries
- Batch size: Configurable, default 32
- Thread pool: Configurable, default 4 workers

## Best Practices

1. **Class Management**
   - Use descriptive class names
   - Provide diverse examples per class
   - Maintain balanced class distributions
   - Regularly update class distributions with new examples

2. **Classification**
   - Use ensemble classification for critical decisions
   - Implement confidence thresholds
   - Monitor and log classification metrics
   - Use active learning for sample selection

3. **Drift Management**
   - Monitor class distributions regularly
   - Set appropriate drift thresholds
   - Follow drift detection recommendations
   - Update models when significant drift detected

4. **Performance Optimization**
   - Use appropriate batch sizes
   - Enable caching for repeated operations
   - Monitor system metrics
   - Scale resources based on usage

5. **System Maintenance**
   - Regularly update class distributions
   - Monitor API usage and quotas
   - Implement proper error handling
   - Backup system state periodically

6. **Security**
   - Secure API key storage
   - Implement rate limiting
   - Monitor and log system access
   - Regular security audits

## Troubleshooting

### Common Issues

1. **Classification Issues**
   - Low confidence scores
   - Inconsistent results
   - Slow response times

2. **Drift Detection Issues**
   - False positives
   - Missed drift events
   - Incorrect severity assessment

3. **Performance Issues**
   - High memory usage
   - Slow response times
   - Cache misses
   - API rate limits

### Solutions

1. **Classification**
   - Verify API keys and quotas
   - Check input text quality
   - Review class examples
   - Adjust confidence thresholds

2. **Drift Detection**
   - Update reference windows
   - Adjust drift thresholds
   - Verify data quality
   - Check distribution metrics

3. **Performance**
   - Optimize batch sizes
   - Increase cache size
   - Scale thread pool
   - Monitor system resources

## Support

For additional support:
1. Check the troubleshooting guide
2. Review system logs
3. Contact the development team
4. Join our community forums

## System Requirements

### Hardware Requirements
- CPU: 4+ cores recommended
- RAM: 16GB+ recommended
- Disk Space: 10GB+ free space
- GPU: Optional, but recommended for large-scale processing

### Software Requirements
- Python 3.8+
- Required packages (see requirements.txt)
- Internet connection for API access

### API Requirements
- OpenAI API key
- Anthropic API key (optional)
- Google API key (optional)
- Mistral API key (optional)
- Groq API key (optional)
