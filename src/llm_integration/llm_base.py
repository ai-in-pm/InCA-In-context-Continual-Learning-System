from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMBase(ABC):
    """Base class for all LLM implementations."""
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def get_embeddings(self, text: str) -> List[float]:
        """Get vector embeddings for the input text."""
        pass
    
    @abstractmethod
    def classify_text(self, text: str, classes: List[str]) -> Dict[str, float]:
        """Classify text into given classes with confidence scores."""
        pass
