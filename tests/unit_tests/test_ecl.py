import pytest
import numpy as np
from src.ecl.gaussian_modeler import GaussianModeler

@pytest.fixture
def gaussian_modeler():
    return GaussianModeler()

@pytest.fixture
def sample_embeddings():
    return [0.1, 0.2, 0.3]

class TestGaussianModeler:
    def test_update_distribution_new_class(self, gaussian_modeler, sample_embeddings):
        class_name = "test_class"
        gaussian_modeler.update_distribution(class_name, sample_embeddings)
        
        assert class_name in gaussian_modeler.class_distributions
        mean, cov = gaussian_modeler.class_distributions[class_name]
        
        assert mean.shape == (len(sample_embeddings),)
        assert cov.shape == (len(sample_embeddings), len(sample_embeddings))
        assert np.allclose(mean, np.array(sample_embeddings))
    
    def test_update_distribution_existing_class(self, gaussian_modeler, sample_embeddings):
        class_name = "test_class"
        # First update
        gaussian_modeler.update_distribution(class_name, sample_embeddings)
        initial_mean = gaussian_modeler.class_distributions[class_name][0].copy()
        
        # Second update with different embeddings
        new_embeddings = [0.4, 0.5, 0.6]
        gaussian_modeler.update_distribution(class_name, new_embeddings)
        
        updated_mean = gaussian_modeler.class_distributions[class_name][0]
        assert not np.allclose(initial_mean, updated_mean)
    
    def test_compute_mahalanobis_distance(self, gaussian_modeler, sample_embeddings):
        class_name = "test_class"
        gaussian_modeler.update_distribution(class_name, sample_embeddings)
        
        # Test distance to self should be small
        distance = gaussian_modeler.compute_mahalanobis_distance(sample_embeddings, class_name)
        assert distance >= 0
        assert distance < 1e-10  # Should be very close to 0
        
        # Test distance to different point should be larger
        different_embeddings = [1.0, 1.0, 1.0]
        distance = gaussian_modeler.compute_mahalanobis_distance(different_embeddings, class_name)
        assert distance > 0
    
    def test_get_class_probability(self, gaussian_modeler, sample_embeddings):
        class_name = "test_class"
        gaussian_modeler.update_distribution(class_name, sample_embeddings)
        
        # Probability for same point should be high
        prob = gaussian_modeler.get_class_probability(sample_embeddings, class_name)
        assert 0 <= prob <= 1
        
        # Probability for distant point should be lower
        different_embeddings = [1.0, 1.0, 1.0]
        different_prob = gaussian_modeler.get_class_probability(different_embeddings, class_name)
        assert different_prob < prob
    
    def test_invalid_class_name(self, gaussian_modeler, sample_embeddings):
        with pytest.raises(ValueError):
            gaussian_modeler.compute_mahalanobis_distance(sample_embeddings, "nonexistent_class")
