"""
Example usage of the InCA system.
"""
from src.inca_agent import InCAAgent
from src.llm_integration.gpt4_handler import GPT4Handler
from src.utils.active_learning import ActiveLearner
from src.utils.drift_detector import DriftDetector
from src.utils.performance_optimizer import PerformanceOptimizer

def basic_classification_example():
    """Basic classification example."""
    # Initialize LLM handlers
    llm_handlers = {
        'gpt4': GPT4Handler()
    }
    
    # Create InCA agent
    agent = InCAAgent(llm_handlers)
    
    # Add classes
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
    
    # Classify text
    text = "Optimize database indexing for better performance"
    result = agent.classify(text)
    print(f"Classification results for: {text}")
    print(result)

def ensemble_classification_example():
    """Ensemble classification example."""
    # Initialize multiple LLM handlers
    llm_handlers = {
        'gpt4': GPT4Handler(),
        'anthropic': AnthropicHandler(),
        'mistral': MistralHandler()
    }
    
    # Create InCA agent
    agent = InCAAgent(llm_handlers)
    
    # Classify using ensemble
    text = "Implement caching strategy for API responses"
    results = {}
    for llm_name in llm_handlers:
        results[llm_name] = agent.classify(text, llm_name=llm_name)
    
    print(f"Ensemble classification results for: {text}")
    for llm_name, result in results.items():
        print(f"{llm_name}: {result}")

def active_learning_example():
    """Active learning example."""
    # Initialize components
    agent = InCAAgent({'gpt4': GPT4Handler()})
    active_learner = ActiveLearner()
    
    # Get unlabeled candidates
    candidates = [
        {
            'text': 'New technical document to classify',
            'confidences': agent.classify('New technical document to classify'),
            'embedding': agent.get_embeddings('New technical document to classify')
        },
        {
            'text': 'Another document for review',
            'confidences': agent.classify('Another document for review'),
            'embedding': agent.get_embeddings('Another document for review')
        }
    ]
    
    # Select most informative samples
    selected = active_learner.select_samples(candidates, n_samples=1)
    print("\nActive Learning Results:")
    print(f"Selected text: {selected[0]['text']}")
    print(f"Uncertainty score: {selected[0]['uncertainty']:.2f}")
    print(f"Diversity score: {selected[0]['diversity_score']:.2f}")

def drift_detection_example():
    """Drift detection example."""
    # Initialize components
    agent = InCAAgent({'gpt4': GPT4Handler()})
    drift_detector = DriftDetector()
    
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
    
    print("\nDrift Detection Results:")
    if drift_metrics:
        print(f"Drift detected! Score: {drift_metrics.drift_score:.2f}")
        print(f"Severity: {drift_metrics.severity}")
        recommendations = drift_detector.get_drift_recommendations(drift_metrics)
        print("Recommendations:")
        for rec in recommendations:
            print(f"- {rec}")
    else:
        print("No significant drift detected")

def performance_optimization_example():
    """Performance optimization example."""
    # Initialize components
    agent = InCAAgent({'gpt4': GPT4Handler()})
    optimizer = PerformanceOptimizer()
    
    # Process multiple texts efficiently
    texts = [
        "First document to classify",
        "Second document to classify",
        "Third document to classify"
    ]
    
    # Batch processing
    results = optimizer.batch_process(texts, agent.classify)
    
    # Monitor performance
    metrics = optimizer.get_optimization_metrics()
    print("\nPerformance Metrics:")
    print(f"Cache Hit Rate: {metrics.cache_hit_rate:.2f}")
    print(f"Average Response Time: {metrics.avg_response_time:.2f}s")
    print(f"Memory Usage: {metrics.memory_usage_mb:.2f}MB")
    print(f"Throughput: {metrics.throughput:.2f} requests/second")

if __name__ == "__main__":
    print("Running InCA System Examples...")
    
    print("\n1. Basic Classification Example")
    basic_classification_example()
    
    print("\n2. Ensemble Classification Example")
    ensemble_classification_example()
    
    print("\n3. Active Learning Example")
    active_learning_example()
    
    print("\n4. Drift Detection Example")
    drift_detection_example()
    
    print("\n5. Performance Optimization Example")
    performance_optimization_example()
