from typing import Dict, List, Any, Optional
import numpy as np
from .llm_integration.llm_base import LLMBase
from .ecl.gaussian_modeler import GaussianModeler

class InCAAgent:
    """Main InCA (In-context Continual Learning) Agent."""
    
    def __init__(self, llm_handlers: Dict[str, LLMBase], primary_llm: str = "gpt4"):
        """
        Initialize InCA Agent.
        
        Args:
            llm_handlers: Dictionary mapping LLM names to their handlers
            primary_llm: Name of the primary LLM to use
        """
        self.llm_handlers = llm_handlers
        self.primary_llm = primary_llm
        self.gaussian_modeler = GaussianModeler()
        self.class_metadata: Dict[str, Dict] = {}
    
    def add_class(self, class_name: str, description: str, examples: List[str]) -> None:
        """
        Add a new class to the system.
        
        Args:
            class_name: Name of the class
            description: Description of the class
            examples: List of example texts for the class
        """
        # Get embeddings for all examples using primary LLM
        embeddings_list = []
        for example in examples:
            embeddings = self.llm_handlers[self.primary_llm].get_embeddings(example)
            embeddings_list.append(embeddings)
        
        # Update Gaussian distribution for the class
        for embeddings in embeddings_list:
            self.gaussian_modeler.update_distribution(class_name, embeddings)
        
        # Store class metadata
        self.class_metadata[class_name] = {
            "description": description,
            "example_count": len(examples)
        }
    
    def classify(self, text: str, top_k: int = 3) -> Dict[str, float]:
        """
        Classify input text using ensemble of LLMs and ECL.
        
        Args:
            text: Input text to classify
            top_k: Number of top classes to return
            
        Returns:
            Dictionary mapping class names to confidence scores
        """
        # Get embeddings for input text
        embeddings = self.llm_handlers[self.primary_llm].get_embeddings(text)
        
        # Get probabilities from Gaussian model
        probabilities = {}
        for class_name in self.class_metadata:
            prob = self.gaussian_modeler.get_class_probability(embeddings, class_name)
            probabilities[class_name] = prob
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: v/total_prob for k, v in probabilities.items()}
        
        # Get top-k classes
        sorted_classes = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_classes[:top_k])
    
    def get_class_info(self, class_name: str) -> Optional[Dict]:
        """Get information about a specific class."""
        return self.class_metadata.get(class_name)
    
    def get_mahalanobis_distance(self, text: str, class_name: str) -> float:
        """Get Mahalanobis distance between text and class distribution."""
        embeddings = self.llm_handlers[self.primary_llm].get_embeddings(text)
        return self.gaussian_modeler.compute_mahalanobis_distance(embeddings, class_name)
