from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass
import time
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class OptimizationMetrics:
    cache_hit_rate: float
    avg_response_time: float
    memory_usage_mb: float
    throughput: float  # requests per second

class PerformanceOptimizer:
    """Optimize system performance through caching and parallel processing."""
    
    def __init__(self, cache_size: int = 1000,
                 max_workers: int = 4,
                 batch_size: int = 32):
        self.cache_size = cache_size
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.metrics = {}
        self.lock = threading.Lock()
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize performance metrics."""
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'total_time': 0.0,
            'peak_memory': 0.0
        }
    
    @lru_cache(maxsize=1000)
    def cached_embedding(self, text: str) -> List[float]:
        """Cache embeddings for frequently accessed texts."""
        with self.lock:
            self.metrics['cache_hits'] += 1
        return []  # Placeholder for actual embedding computation
    
    def batch_process(self, texts: List[str],
                     processor_func: Any) -> List[Any]:
        """Process texts in batches for better throughput."""
        results = []
        
        # Split into batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = self._process_batch(batch, processor_func)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, batch: List[str],
                      processor_func: Any) -> List[Any]:
        """Process a single batch using thread pool."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(processor_func, text) 
                      for text in batch]
            return [f.result() for f in as_completed(futures)]
    
    def optimize_embedding_computation(self, 
                                    embeddings: List[List[float]]) -> List[List[float]]:
        """Optimize embedding vectors for memory efficiency."""
        # Convert to numpy for efficient operations
        embedding_array = np.array(embeddings)
        
        # Perform dimensionality reduction if needed
        if embedding_array.shape[1] > 100:  # Arbitrary threshold
            from sklearn.decomposition import PCA
            pca = PCA(n_components=100)
            embedding_array = pca.fit_transform(embedding_array)
        
        # Quantize values to reduce memory
        embedding_array = np.round(embedding_array, decimals=5)
        
        return embedding_array.tolist()
    
    def get_optimization_metrics(self) -> OptimizationMetrics:
        """Get current optimization metrics."""
        total_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total_requests == 0:
            return OptimizationMetrics(
                cache_hit_rate=0.0,
                avg_response_time=0.0,
                memory_usage_mb=0.0,
                throughput=0.0
            )
        
        return OptimizationMetrics(
            cache_hit_rate=self.metrics['cache_hits'] / total_requests,
            avg_response_time=self.metrics['total_time'] / total_requests,
            memory_usage_mb=self.metrics['peak_memory'],
            throughput=total_requests / max(1, self.metrics['total_time'])
        )
    
    def optimize_class_storage(self, 
                             class_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize storage of class data."""
        optimized_data = {}
        
        for class_name, data in class_data.items():
            # Optimize embeddings
            if 'embeddings' in data:
                data['embeddings'] = self.optimize_embedding_computation(
                    data['embeddings']
                )
            
            # Compress text data if needed
            if 'examples' in data and len(data['examples']) > 100:
                # Keep only representative examples
                data['examples'] = data['examples'][:100]
            
            optimized_data[class_name] = data
        
        return optimized_data
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get recommendations for performance optimization."""
        metrics = self.get_optimization_metrics()
        recommendations = []
        
        # Cache-related recommendations
        if metrics.cache_hit_rate < 0.5:
            recommendations.append(
                "Consider increasing cache size or reviewing cache strategy"
            )
        
        # Response time recommendations
        if metrics.avg_response_time > 1.0:  # 1 second threshold
            recommendations.extend([
                "Consider increasing batch size for better throughput",
                "Review thread pool size for optimal concurrency"
            ])
        
        # Memory usage recommendations
        if metrics.memory_usage_mb > 1024:  # 1GB threshold
            recommendations.extend([
                "Consider implementing embedding compression",
                "Review class storage optimization"
            ])
        
        return recommendations
    
    def clear_metrics(self):
        """Reset performance metrics."""
        with self.lock:
            self._initialize_metrics()
