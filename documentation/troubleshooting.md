# Troubleshooting Guide

## Common Installation Issues

### Python Version Issues

**Problem**: Incorrect Python version
```
❌ Python version X.Y is not supported
```

**Solution**:
1. Check your Python version:
   ```bash
   python --version
   ```
2. Install Python 3.8 or higher from [python.org](https://python.org)
3. Ensure the correct Python version is in your PATH

### Virtual Environment Issues

**Problem**: Virtual environment not activating
```
'venv' is not recognized as a command...
```

**Solution**:
1. Ensure Python's venv module is installed:
   ```bash
   python -m pip install --upgrade virtualenv
   ```
2. Create a new virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate properly:
   - Windows: `.\venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`

### Dependency Installation Issues

**Problem**: Package installation fails
```
ERROR: Could not install packages due to an OSError
```

**Solution**:
1. Upgrade pip:
   ```bash
   python -m pip install --upgrade pip
   ```
2. Install packages individually:
   ```bash
   pip install numpy
   pip install scipy
   # etc.
   ```
3. Check for system dependencies (Linux):
   ```bash
   sudo apt-get update
   sudo apt-get install python3-dev
   ```

### API Key Issues

**Problem**: API keys not being recognized
```
❌ OPENAI_API_KEY is not set in .env file
```

**Solution**:
1. Check .env file exists:
   ```bash
   ls .env  # Unix/MacOS
   dir .env # Windows
   ```
2. Verify .env format:
   ```
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-...
   ```
3. No quotes around keys
4. No spaces around '='
5. Reload environment:
   ```bash
   deactivate
   source venv/bin/activate  # Unix/MacOS
   .\venv\Scripts\activate   # Windows
   ```

## Runtime Issues

### Memory Issues

**Problem**: Out of memory errors
```
MemoryError: Unable to allocate array
```

**Solution**:
1. Check available system memory
2. Reduce batch size in `.env`:
   ```
   BATCH_SIZE=16  # Reduce from default 32
   ```
3. Enable memory optimization:
   ```
   ENABLE_MEMORY_OPTIMIZATION=true
   ```

### Performance Issues

**Problem**: Slow response times
```
❌ Classification time exceeds threshold
```

**Solution**:
1. Enable caching:
   ```
   CACHE_ENABLED=true
   ```
2. Adjust batch size:
   ```
   BATCH_SIZE=64  # Increase for better throughput
   ```
3. Check network connection
4. Monitor API rate limits

### Import Errors

**Problem**: Module not found errors
```
ModuleNotFoundError: No module named 'xyz'
```

**Solution**:
1. Verify package installation:
   ```bash
   pip list | grep xyz
   ```
2. Install missing package:
   ```bash
   pip install xyz
   ```
3. Check Python path:
   ```python
   import sys
   print(sys.path)
   ```

## Advanced Issues

### GPU Support

**Problem**: GPU not being utilized
```
WARNING: CUDA not available
```

**Solution**:
1. Install CUDA toolkit
2. Install GPU version of PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Verify GPU detection:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

### Distributed Setup

**Problem**: Redis connection issues
```
Could not connect to Redis
```

**Solution**:
1. Install Redis:
   - Windows: Use WSL or Docker
   - Unix/MacOS: `brew install redis`
2. Start Redis server:
   ```bash
   redis-server
   ```
3. Test connection:
   ```bash
   redis-cli ping
   ```

## System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2GB free disk space
- Internet connection

### Recommended Requirements
- Python 3.10+
- 16GB RAM
- 10GB free disk space
- High-speed internet connection
- GPU (optional)

## Error Code Reference

### Installation Errors (1xxx)

| Code | Description | Solution |
|------|-------------|----------|
| 1001 | Python version not supported | Install Python 3.8+ |
| 1002 | Virtual environment creation failed | Check Python installation |
| 1003 | Package installation failed | Check internet connection, upgrade pip |
| 1004 | API key validation failed | Verify API keys in .env |

### Runtime Errors (2xxx)

| Code | Description | Solution |
|------|-------------|----------|
| 2001 | Classification failed | Check input text, API status |
| 2002 | Memory error | Reduce batch size, enable optimization |
| 2003 | API rate limit exceeded | Implement rate limiting, check quotas |
| 2004 | Cache initialization failed | Check disk space, permissions |

### Configuration Errors (3xxx)

| Code | Description | Solution |
|------|-------------|----------|
| 3001 | Invalid configuration | Check .env format |
| 3002 | Missing required setting | Add required setting to .env |
| 3003 | Invalid value type | Correct value type in .env |
| 3004 | Conflicting settings | Review configuration guide |

## Performance Optimization

### Memory Optimization
```bash
# .env settings
MAX_MEMORY_MB=8192
ENABLE_MEMORY_OPTIMIZATION=true
CACHE_SIZE=1000
```

### CPU Optimization
```bash
# .env settings
BATCH_SIZE=32
MAX_WORKERS=4
ENABLE_PARALLEL_PROCESSING=true
```

### GPU Optimization
```bash
# .env settings
ENABLE_GPU=true
GPU_MEMORY_FRACTION=0.8
CUDA_VISIBLE_DEVICES=0
```

## Security Issues

### API Key Security

**Problem**: Exposed API keys
```
WARNING: API key found in code/logs
```

**Solution**:
1. Move keys to .env
2. Add *.env to .gitignore
3. Rotate exposed keys
4. Use environment variables

### Rate Limiting

**Problem**: Excessive API usage
```
ERROR: Rate limit exceeded
```

**Solution**:
1. Implement rate limiting:
   ```python
   RATE_LIMIT_PER_MINUTE=60
   ```
2. Use token bucket algorithm
3. Monitor usage metrics

## Logging and Monitoring

### Enable Detailed Logging
```bash
# .env settings
LOG_LEVEL=DEBUG
ENABLE_DETAILED_LOGGING=true
LOG_FILE=logs/inca.log
```

### Monitor System Health
```bash
# Run health check
python -m src.monitoring.health_check

# View metrics
python -m src.monitoring.view_metrics
```

## Common Error Messages

### API Errors

```
ERROR: OpenAI API error: Rate limit exceeded
```
- Wait and retry
- Implement exponential backoff
- Check API quotas

```
ERROR: Invalid API key
```
- Verify key in .env
- Check key format
- Ensure key is active

### System Errors

```
ERROR: CUDA out of memory
```
- Reduce batch size
- Clear GPU memory
- Monitor GPU usage

```
ERROR: Process killed (OOM)
```
- Reduce memory usage
- Enable memory optimization
- Increase system memory

## Debugging Tools

### Memory Profiling
```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Your code here
    pass
```

### Performance Profiling
```python
import cProfile

cProfile.run('your_function()')
```

### API Debugging
```bash
# Enable API debug mode
DEBUG_API_CALLS=true
LOG_API_RESPONSES=true
```

## Getting Help

If you continue to experience issues:

1. Run the verification script with verbose logging:
   ```bash
   python verify_installation.py --verbose
   ```

2. Check the logs:
   ```bash
   cat logs/inca.log  # Unix/MacOS
   type logs\inca.log # Windows
   ```

3. Create an issue on GitHub with:
   - Full error message
   - Python version
   - Operating system
   - Verification script output
   - Relevant logs

4. Join our Discord community for real-time support

## Best Practices

### Installation
1. Use virtual environments
2. Install from requirements.txt
3. Verify installation
4. Keep dependencies updated

### Configuration
1. Use .env for settings
2. Follow security guidelines
3. Optimize for your hardware
4. Monitor performance

### Maintenance
1. Regular updates
2. Log rotation
3. Cache cleanup
4. Performance monitoring

## Additional Resources

1. Documentation
   - API Reference
   - Configuration Guide
   - Development Guide

2. External Resources
   - Python Documentation
   - LLM API Documentation
   - GPU Computing Guide

3. Community
   - GitHub Discussions
   - Stack Overflow
   - Discord Channel
