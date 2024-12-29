from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.stats import ks_2samp
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class DriftMetrics:
    timestamp: datetime
    drift_score: float
    p_value: float
    affected_classes: List[str]
    severity: str  # 'low', 'medium', 'high'

class DriftDetector:
    """Detect and monitor concept drift in the classification system."""
    
    def __init__(self, window_size: int = 1000,
                 drift_threshold: float = 0.3,
                 min_samples: int = 50):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples
        self.reference_embeddings: Dict[str, List[List[float]]] = {}
        self.reference_stats: Dict[str, Dict] = {}
        self.drift_history: List[DriftMetrics] = []
    
    def update_reference_window(self, class_name: str,
                              embeddings: List[List[float]]):
        """Update reference window for a class."""
        if len(embeddings) < self.min_samples:
            return
        
        self.reference_embeddings[class_name] = embeddings[-self.window_size:]
        
        # Update reference statistics
        embedding_array = np.array(embeddings[-self.window_size:])
        self.reference_stats[class_name] = {
            'mean': np.mean(embedding_array, axis=0),
            'std': np.std(embedding_array, axis=0),
            'covariance': np.cov(embedding_array.T)
        }
    
    def compute_distribution_distance(self,
                                   reference: List[List[float]],
                                   current: List[List[float]]) -> Tuple[float, float]:
        """
        Compute distribution distance using Kolmogorov-Smirnov test.
        Returns drift score and p-value.
        """
        if len(reference) < self.min_samples or len(current) < self.min_samples:
            return 0.0, 1.0
        
        # Project data to 1D for KS test
        ref_proj = np.array(reference).mean(axis=1)
        curr_proj = np.array(current).mean(axis=1)
        
        # Perform KS test
        ks_stat, p_value = ks_2samp(ref_proj, curr_proj)
        return float(ks_stat), float(p_value)
    
    def detect_drift(self, class_name: str,
                    current_embeddings: List[List[float]]) -> Optional[DriftMetrics]:
        """
        Detect if drift has occurred for a specific class.
        Returns drift metrics if drift is detected, None otherwise.
        """
        if class_name not in self.reference_embeddings:
            return None
        
        if len(current_embeddings) < self.min_samples:
            return None
        
        # Compute drift score
        drift_score, p_value = self.compute_distribution_distance(
            self.reference_embeddings[class_name],
            current_embeddings
        )
        
        # Determine drift severity
        severity = 'low'
        if drift_score > self.drift_threshold * 1.5:
            severity = 'high'
        elif drift_score > self.drift_threshold:
            severity = 'medium'
        
        # Create drift metrics
        metrics = DriftMetrics(
            timestamp=datetime.now(),
            drift_score=drift_score,
            p_value=p_value,
            affected_classes=[class_name],
            severity=severity
        )
        
        # Store in history if drift detected
        if drift_score > self.drift_threshold:
            self.drift_history.append(metrics)
            return metrics
        
        return None
    
    def get_drift_statistics(self, 
                           time_window: Optional[timedelta] = None) -> Dict:
        """Get drift statistics over a time window."""
        if not self.drift_history:
            return {
                'total_drifts': 0,
                'avg_drift_score': 0.0,
                'severity_distribution': {'low': 0, 'medium': 0, 'high': 0},
                'affected_classes': set()
            }
        
        # Filter by time window if specified
        if time_window:
            cutoff_time = datetime.now() - time_window
            relevant_drifts = [d for d in self.drift_history 
                             if d.timestamp > cutoff_time]
        else:
            relevant_drifts = self.drift_history
        
        # Compute statistics
        severity_counts = {'low': 0, 'medium': 0, 'high': 0}
        affected_classes = set()
        total_drift_score = 0.0
        
        for drift in relevant_drifts:
            severity_counts[drift.severity] += 1
            affected_classes.update(drift.affected_classes)
            total_drift_score += drift.drift_score
        
        return {
            'total_drifts': len(relevant_drifts),
            'avg_drift_score': total_drift_score / len(relevant_drifts),
            'severity_distribution': severity_counts,
            'affected_classes': affected_classes
        }
    
    def get_drift_recommendations(self, metrics: DriftMetrics) -> List[str]:
        """Get recommendations based on drift metrics."""
        recommendations = []
        
        if metrics.severity == 'high':
            recommendations.extend([
                "Immediately collect new labeled samples for affected classes",
                "Consider retraining the model with recent data",
                "Alert monitoring team for investigation"
            ])
        elif metrics.severity == 'medium':
            recommendations.extend([
                "Schedule collection of new samples",
                "Monitor affected classes more frequently",
                "Review recent classification performance"
            ])
        else:  # low
            recommendations.extend([
                "Continue regular monitoring",
                "Flag for review in next maintenance cycle"
            ])
        
        return recommendations
