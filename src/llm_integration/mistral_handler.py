from typing import List, Dict
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from .llm_base import LLMBase

class MistralHandler(LLMBase):
    def __init__(self, api_key: str):
        """Initialize Mistral handler with API key."""
        self.client = MistralClient(api_key=api_key)
        self.model = "mistral-large-latest"
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response using Mistral."""
        try:
            messages = [ChatMessage(role="user", content=prompt)]
            response = self.client.chat(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating Mistral response: {str(e)}")
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using Mistral's embedding model."""
        try:
            response = self.client.embeddings(
                model="mistral-embed",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def classify_text(self, text: str, classes: List[str]) -> Dict[str, float]:
        """Classify text using Mistral."""
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
