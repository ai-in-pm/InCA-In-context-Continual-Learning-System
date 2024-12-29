"""
Visualization script for performance monitoring.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_response_times(response_times: pd.Series, title: str = "Response Times"):
    """Plot response time distribution and trends."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Response time distribution
    sns.histplot(data=response_times, ax=ax1)
    ax1.set_title("Response Time Distribution")
    ax1.set_xlabel("Response Time (ms)")
    ax1.set_ylabel("Count")
    
    # Response time trend
    ax2.plot(response_times.index, response_times.values)
    ax2.set_title("Response Time Trend")
    ax2.set_xlabel("Request")
    ax2.set_ylabel("Response Time (ms)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_memory_usage(memory_data: pd.DataFrame, title: str = "Memory Usage Over Time"):
    """Plot memory usage metrics."""
    plt.figure(figsize=(12, 6))
    plt.plot(memory_data.index, memory_data['used_mb'], label='Used Memory')
    plt.fill_between(memory_data.index, 0, memory_data['used_mb'], alpha=0.3)
    plt.axhline(y=memory_data['total_mb'].iloc[0], color='r', linestyle='--', 
                label='Total Memory')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Memory (MB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_cache_metrics(cache_data: pd.DataFrame, title: str = "Cache Performance"):
    """Plot cache hit rate and size."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Cache hit rate
    ax1.plot(cache_data.index, cache_data['hit_rate'], label='Hit Rate')
    ax1.set_title("Cache Hit Rate Over Time")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Hit Rate")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Cache size
    ax2.plot(cache_data.index, cache_data['size'], label='Cache Size')
    ax2.set_title("Cache Size Over Time")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Size (entries)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_throughput(throughput_data: pd.DataFrame, title: str = "System Throughput"):
    """Plot system throughput metrics."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Requests per second
    ax1.plot(throughput_data.index, throughput_data['requests_per_second'], 
             label='Requests/Second')
    ax1.set_title("Requests per Second Over Time")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Requests/Second")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Success rate
    ax2.plot(throughput_data.index, throughput_data['success_rate'], 
             label='Success Rate')
    ax2.set_title("Success Rate Over Time")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Success Rate")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_resource_usage(resource_data: pd.DataFrame, title: str = "Resource Usage"):
    """Plot CPU, memory, and disk usage."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # CPU Usage
    ax1.plot(resource_data.index, resource_data['cpu_percent'], label='CPU Usage')
    ax1.set_title("CPU Usage Over Time")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("CPU %")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Memory Usage
    ax2.plot(resource_data.index, resource_data['memory_percent'], 
             label='Memory Usage')
    ax2.set_title("Memory Usage Over Time")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Memory %")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Disk Usage
    ax3.plot(resource_data.index, resource_data['disk_percent'], 
             label='Disk Usage')
    ax3.set_title("Disk Usage Over Time")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Disk %")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Example response times
    n_requests = 1000
    response_times = pd.Series(
        np.random.gamma(2, 100, n_requests),
        index=range(n_requests)
    )
    plot_response_times(response_times)
    
    # Example memory usage
    dates = pd.date_range(start='2024-01-01', periods=30, freq='H')
    memory_data = pd.DataFrame({
        'used_mb': np.random.uniform(2000, 6000, 30),
        'total_mb': [8192] * 30
    }, index=dates)
    plot_memory_usage(memory_data)
    
    # Example cache metrics
    cache_data = pd.DataFrame({
        'hit_rate': np.random.uniform(0.6, 0.9, 30),
        'size': np.random.uniform(800, 1000, 30)
    }, index=dates)
    plot_cache_metrics(cache_data)
    
    # Example throughput data
    throughput_data = pd.DataFrame({
        'requests_per_second': np.random.uniform(10, 30, 30),
        'success_rate': np.random.uniform(0.95, 1.0, 30)
    }, index=dates)
    plot_throughput(throughput_data)
    
    # Example resource usage
    resource_data = pd.DataFrame({
        'cpu_percent': np.random.uniform(20, 80, 30),
        'memory_percent': np.random.uniform(40, 70, 30),
        'disk_percent': np.random.uniform(50, 60, 30)
    }, index=dates)
    plot_resource_usage(resource_data)
