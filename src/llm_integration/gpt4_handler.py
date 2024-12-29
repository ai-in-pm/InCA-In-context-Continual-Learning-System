from typing import List, Dict
import openai
from .llm_base import LLMBase

class GPT4Handler(LLMBase):
    def __init__(self, api_key: str):
        """Initialize GPT-4 handler with API key."""
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-4-1106-preview"  # Using the latest GPT-4 model
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response using GPT-4."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating GPT-4 response: {str(e)}")
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using GPT-4's embedding model."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def classify_text(self, text: str, classes: List[str]) -> Dict[str, float]:
        """Classify text using GPT-4."""
        prompt = f"""Classify the following text into the given classes. 
        Return confidence scores for each class as a decimal between 0 and 1.
        Text: {text}
        Classes: {', '.join(classes)}"""
        
        try:
            response = self.generate_response(prompt)
            # Parse response and convert to confidence scores
            # This is a simplified implementation
            scores = {}
            for class_name in classes:
                # In a real implementation, we would parse the GPT-4 response
                # more carefully to extract proper confidence scores
                scores[class_name] = 0.0
            return scores
        except Exception as e:
            raise Exception(f"Error in classification: {str(e)}")
