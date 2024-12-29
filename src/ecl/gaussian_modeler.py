import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import multivariate_normal

class GaussianModeler:
    """Manages Gaussian distributions for class modeling."""
    
    def __init__(self):
        self.class_distributions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    
    def update_distribution(self, class_name: str, embeddings: List[float]):
        """Update the Gaussian distribution for a class."""
        embedding_array = np.array(embeddings)
        
        if class_name not in self.class_distributions:
            # Initialize new distribution
            mean = embedding_array
            cov = np.eye(len(embedding_array)) * 0.1  # Initial covariance
            self.class_distributions[class_name] = (mean, cov)
        else:
            # Update existing distribution
            old_mean, old_cov = self.class_distributions[class_name]
            n_samples = len(self.class_distributions)  # Approximate sample count
            
            # Update mean
            new_mean = ((n_samples * old_mean) + embedding_array) / (n_samples + 1)
            
            # Update covariance
            diff = embedding_array - old_mean
            new_cov = (n_samples * old_cov + np.outer(diff, diff)) / (n_samples + 1)
            
            self.class_distributions[class_name] = (new_mean, new_cov)
    
    def compute_mahalanobis_distance(self, embeddings: List[float], class_name: str) -> float:
        """Compute Mahalanobis distance between embeddings and class distribution."""
        if class_name not in self.class_distributions:
            raise ValueError(f"No distribution found for class {class_name}")
            
        mean, cov = self.class_distributions[class_name]
        embedding_array = np.array(embeddings)
        
        try:
            inv_cov = np.linalg.inv(cov)
            diff = embedding_array - mean
            distance = np.sqrt(diff.dot(inv_cov).dot(diff))
            return float(distance)
        except np.linalg.LinAlgError:
            # Fallback to Euclidean distance if covariance is singular
            return float(np.linalg.norm(embedding_array - mean))
    
    def get_class_probability(self, embeddings: List[float], class_name: str) -> float:
        """Get probability of embeddings belonging to a class."""
        if class_name not in self.class_distributions:
            return 0.0
            
        mean, cov = self.class_distributions[class_name]
        try:
            prob = multivariate_normal.pdf(embeddings, mean=mean, cov=cov)
            return float(prob)
        except:
            # Fallback to simpler probability calculation
            distance = self.compute_mahalanobis_distance(embeddings, class_name)
            return float(np.exp(-distance))
