import time
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PerformanceMetric:
    timestamp: datetime
    duration_ms: float
    memory_mb: float
    success: bool
    error_message: Optional[str] = None

class PerformanceMonitor:
    """Monitor and track system performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[PerformanceMetric]] = {
            'classification': [],
            'embedding': [],
            'distribution_update': []
        }
        self.thresholds = {
            'classification_time_ms': 2000,  # 2 seconds
            'embedding_time_ms': 1000,       # 1 second
            'memory_mb': 8192                # 8GB
        }
    
    def record_metric(self, operation: str, duration_ms: float, 
                     memory_mb: float, success: bool, 
                     error_message: Optional[str] = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            memory_mb=memory_mb,
            success=success,
            error_message=error_message
        )
        self.metrics[operation].append(metric)
    
    def get_statistics(self, operation: str, 
                      time_window_minutes: Optional[int] = None) -> Dict:
        """Get performance statistics for an operation."""
        metrics = self.metrics[operation]
        
        if time_window_minutes:
            cutoff = datetime.now().timestamp() - (time_window_minutes * 60)
            metrics = [m for m in metrics 
                      if m.timestamp.timestamp() > cutoff]
        
        if not metrics:
            return {
                'count': 0,
                'success_rate': 0.0,
                'avg_duration_ms': 0.0,
                'p95_duration_ms': 0.0,
                'avg_memory_mb': 0.0
            }
        
        durations = [m.duration_ms for m in metrics]
        memories = [m.memory_mb for m in metrics]
        successes = [m.success for m in metrics]
        
        return {
            'count': len(metrics),
            'success_rate': sum(successes) / len(successes),
            'avg_duration_ms': np.mean(durations),
            'p95_duration_ms': np.percentile(durations, 95),
            'avg_memory_mb': np.mean(memories)
        }
    
    def check_thresholds(self, operation: str) -> List[str]:
        """Check if any performance thresholds are exceeded."""
        stats = self.get_statistics(operation, time_window_minutes=5)
        warnings = []
        
        if operation == 'classification':
            if stats['p95_duration_ms'] > self.thresholds['classification_time_ms']:
                warnings.append(
                    f"Classification time (P95) exceeds threshold: "
                    f"{stats['p95_duration_ms']:.2f}ms > "
                    f"{self.thresholds['classification_time_ms']}ms"
                )
        
        if stats['avg_memory_mb'] > self.thresholds['memory_mb']:
            warnings.append(
                f"Memory usage exceeds threshold: "
                f"{stats['avg_memory_mb']:.2f}MB > "
                f"{self.thresholds['memory_mb']}MB"
            )
        
        return warnings
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        report = {}
        for operation in self.metrics:
            stats = self.get_statistics(operation)
            warnings = self.check_thresholds(operation)
            report[operation] = {
                'statistics': stats,
                'warnings': warnings,
                'status': 'healthy' if not warnings else 'warning'
            }
        return report
