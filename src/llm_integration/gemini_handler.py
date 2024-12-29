from typing import List, Dict
import google.generativeai as genai
from .llm_base import LLMBase

class GeminiHandler(LLMBase):
    def __init__(self, api_key: str):
        """Initialize Gemini handler with API key."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
    def generate_response(self, prompt: str) -> str:
        """Generate a response using Gemini."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Error generating Gemini response: {str(e)}")
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using Gemini's embedding model."""
        try:
            # Note: Using PaLM's text embedding model as Gemini's embedding
            # capabilities might be limited
            model = 'models/embedding-001'
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def classify_text(self, text: str, classes: List[str]) -> Dict[str, float]:
        """Classify text using Gemini."""
        prompt = f"""Classify the following text into the given classes. 
        Return confidence scores for each class as a decimal between 0 and 1.
        Text: {text}
        Classes: {', '.join(classes)}
        Format your response as a JSON object with class names as keys and confidence scores as values."""
        
        try:
            response = self.generate_response(prompt)
            # Parse response into confidence scores
            # This is a simplified implementation
            scores = {}
            for class_name in classes:
                scores[class_name] = 0.0
            return scores
        except Exception as e:
            raise Exception(f"Error in classification: {str(e)}")
