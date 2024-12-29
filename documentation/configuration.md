# InCA System Configuration Guide

## Environment Variables

### API Keys
```bash
# Required
OPENAI_API_KEY=your_openai_key

# Optional (but recommended for ensemble classification)
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
MISTRAL_API_KEY=your_mistral_key
GROQ_API_KEY=your_groq_key
```

### Performance Settings
```bash
# Memory and Cache
MAX_MEMORY_MB=8192              # Maximum memory usage in MB
CACHE_SIZE=1000                 # Number of embeddings to cache
ENABLE_MEMORY_OPTIMIZATION=true # Enable memory optimization features

# Processing
BATCH_SIZE=32                   # Batch size for parallel processing
MAX_WORKERS=4                   # Number of worker threads
ENABLE_GPU=false               # Enable GPU acceleration if available

# Response Time
MAX_RESPONSE_TIME_MS=2000      # Maximum allowed response time
TIMEOUT_MS=5000                # API call timeout
```

### Feature Flags
```bash
# Core Features
ENABLE_ENSEMBLE_CLASSIFICATION=true  # Use multiple LLMs for classification
ENABLE_ACTIVE_LEARNING=true         # Enable active learning system
ENABLE_DRIFT_DETECTION=true         # Enable drift detection
ENABLE_PERFORMANCE_MONITORING=true   # Enable performance monitoring

# Advanced Features
ENABLE_AUTO_SCALING=false           # Automatically scale resources
ENABLE_DISTRIBUTED_PROCESSING=false  # Enable distributed processing
ENABLE_ADVANCED_CACHING=true        # Enable advanced caching strategies
```

### Monitoring Settings
```bash
# Logging
LOG_LEVEL=INFO                      # Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_FILE=logs/inca.log             # Log file location
ENABLE_DETAILED_LOGGING=true        # Enable detailed logging

# Monitoring
ENABLE_METRICS_COLLECTION=true      # Collect performance metrics
METRICS_EXPORT_INTERVAL=300         # Export metrics every 5 minutes
ENABLE_HEALTH_CHECKS=true           # Enable system health checks
```

### Classification Settings
```bash
# Thresholds
MIN_CONFIDENCE_THRESHOLD=0.6        # Minimum confidence for classification
ENSEMBLE_AGREEMENT_THRESHOLD=0.7    # Required agreement for ensemble
MAX_CLASSES_PER_REQUEST=5          # Maximum classes to return

# Active Learning
UNCERTAINTY_WEIGHT=0.7              # Weight for uncertainty in sample selection
DIVERSITY_WEIGHT=0.3                # Weight for diversity in sample selection
MIN_SAMPLES_PER_CLASS=50           # Minimum samples per class
```

### Drift Detection Settings
```bash
# Windows and Thresholds
REFERENCE_WINDOW_SIZE=1000          # Size of reference window
DRIFT_THRESHOLD=0.3                 # Threshold for drift detection
MIN_SAMPLES_FOR_DRIFT=50           # Minimum samples for drift detection

# Severity Levels
DRIFT_SEVERITY_LOW=0.3             # Low severity threshold
DRIFT_SEVERITY_MEDIUM=0.5          # Medium severity threshold
DRIFT_SEVERITY_HIGH=0.7            # High severity threshold
```

## Configuration Examples

### High Performance Setup
```bash
# Memory and Processing
MAX_MEMORY_MB=16384
CACHE_SIZE=5000
BATCH_SIZE=64
MAX_WORKERS=8
ENABLE_GPU=true

# Features
ENABLE_ADVANCED_CACHING=true
ENABLE_AUTO_SCALING=true
ENABLE_DISTRIBUTED_PROCESSING=true

# Monitoring
ENABLE_DETAILED_LOGGING=true
ENABLE_METRICS_COLLECTION=true
LOG_LEVEL=DEBUG
```

### Low Resource Setup
```bash
# Memory and Processing
MAX_MEMORY_MB=4096
CACHE_SIZE=500
BATCH_SIZE=16
MAX_WORKERS=2
ENABLE_GPU=false

# Features
ENABLE_ADVANCED_CACHING=false
ENABLE_AUTO_SCALING=false
ENABLE_DISTRIBUTED_PROCESSING=false

# Monitoring
ENABLE_DETAILED_LOGGING=false
ENABLE_METRICS_COLLECTION=true
LOG_LEVEL=WARNING
```

### Production Setup
```bash
# Memory and Processing
MAX_MEMORY_MB=8192
CACHE_SIZE=2000
BATCH_SIZE=32
MAX_WORKERS=4
ENABLE_GPU=true

# Features
ENABLE_ENSEMBLE_CLASSIFICATION=true
ENABLE_DRIFT_DETECTION=true
ENABLE_PERFORMANCE_MONITORING=true

# Security and Monitoring
ENABLE_DETAILED_LOGGING=true
ENABLE_METRICS_COLLECTION=true
ENABLE_HEALTH_CHECKS=true
LOG_LEVEL=INFO
```

## Configuration Best Practices

### Memory Management
1. Set `MAX_MEMORY_MB` based on available system memory
2. Adjust `CACHE_SIZE` based on typical workload
3. Enable memory optimization for large datasets
4. Monitor memory usage and adjust as needed

### Performance Tuning
1. Find optimal `BATCH_SIZE` for your use case
2. Adjust `MAX_WORKERS` based on CPU cores
3. Enable GPU acceleration if available
4. Monitor response times and throughput

### Feature Configuration
1. Enable ensemble classification for critical applications
2. Use active learning for continuous improvement
3. Enable drift detection for production systems
4. Configure appropriate thresholds for your use case

### Monitoring Setup
1. Use detailed logging during development
2. Enable metrics collection in production
3. Set up health checks for critical systems
4. Configure appropriate log levels

### Security Considerations
1. Secure storage of API keys
2. Implement rate limiting
3. Enable detailed logging for auditing
4. Regular security reviews

## Troubleshooting

### Common Configuration Issues

1. **Memory Issues**
   - Symptom: Out of memory errors
   - Solution: Reduce `CACHE_SIZE` or increase `MAX_MEMORY_MB`
   - Check: Monitor memory usage metrics

2. **Performance Issues**
   - Symptom: Slow response times
   - Solution: Adjust `BATCH_SIZE` and `MAX_WORKERS`
   - Check: Review performance monitoring logs

3. **Classification Issues**
   - Symptom: Low confidence scores
   - Solution: Adjust confidence thresholds
   - Check: Review classification metrics

4. **Drift Detection Issues**
   - Symptom: False positives/negatives
   - Solution: Adjust drift thresholds
   - Check: Review drift detection logs

## Support

For configuration support:
1. Check the troubleshooting guide
2. Review system logs
3. Contact system administrators
4. Join our community forums
