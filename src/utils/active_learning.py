from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.stats import entropy
from sklearn.cluster import KMeans
from dataclasses import dataclass

@dataclass
class SampleCandidate:
    text: str
    uncertainty: float
    diversity_score: float
    combined_score: float
    predicted_class: Optional[str] = None

class ActiveLearner:
    """Active learning component for selecting informative samples."""
    
    def __init__(self, uncertainty_weight: float = 0.7, 
                 diversity_weight: float = 0.3):
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.selected_samples: List[Tuple[str, str]] = []  # (text, class)
    
    def compute_uncertainty(self, confidences: Dict[str, float]) -> float:
        """Compute uncertainty score using entropy."""
        probs = list(confidences.values())
        return float(entropy(probs))
    
    def compute_diversity(self, embedding: List[float], 
                         existing_embeddings: List[List[float]]) -> float:
        """Compute diversity score based on distance to existing samples."""
        if not existing_embeddings:
            return 1.0
        
        distances = []
        for exist_embed in existing_embeddings:
            dist = np.linalg.norm(np.array(embedding) - np.array(exist_embed))
            distances.append(dist)
        
        # Normalize by maximum distance
        max_dist = max(distances) if distances else 1.0
        return float(min(distances) / max_dist if distances else 1.0)
    
    def select_samples(self, candidates: List[Dict],
                      n_samples: int = 5) -> List[SampleCandidate]:
        """
        Select most informative samples for labeling.
        
        Args:
            candidates: List of dicts with 'text', 'confidences', 'embedding'
            n_samples: Number of samples to select
            
        Returns:
            List of selected samples with their scores
        """
        sample_candidates = []
        existing_embeddings = [s['embedding'] for s in self.selected_samples]
        
        for candidate in candidates:
            # Compute uncertainty score
            uncertainty = self.compute_uncertainty(candidate['confidences'])
            
            # Compute diversity score
            diversity = self.compute_diversity(candidate['embedding'],
                                            existing_embeddings)
            
            # Compute combined score
            combined_score = (self.uncertainty_weight * uncertainty +
                            self.diversity_weight * diversity)
            
            # Get predicted class
            predicted_class = max(candidate['confidences'].items(),
                                key=lambda x: x[1])[0]
            
            sample_candidates.append(
                SampleCandidate(
                    text=candidate['text'],
                    uncertainty=uncertainty,
                    diversity_score=diversity,
                    combined_score=combined_score,
                    predicted_class=predicted_class
                )
            )
        
        # Sort by combined score and select top N
        sample_candidates.sort(key=lambda x: x.combined_score, reverse=True)
        return sample_candidates[:n_samples]
    
    def cluster_samples(self, embeddings: List[List[float]], 
                       n_clusters: int = 5) -> List[int]:
        """Cluster samples to ensure diverse selection."""
        if len(embeddings) < n_clusters:
            return list(range(len(embeddings)))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(embeddings)
    
    def update_selected_samples(self, text: str, assigned_class: str,
                              embedding: List[float]):
        """Update the list of selected samples."""
        self.selected_samples.append({
            'text': text,
            'class': assigned_class,
            'embedding': embedding
        })
    
    def get_sample_statistics(self) -> Dict:
        """Get statistics about selected samples."""
        class_counts = {}
        for sample in self.selected_samples:
            class_name = sample['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'total_samples': len(self.selected_samples),
            'class_distribution': class_counts,
            'unique_classes': len(class_counts)
        }
