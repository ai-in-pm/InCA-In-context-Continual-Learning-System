import pytest
from unittest.mock import Mock, patch
import numpy as np
from src.llm_integration.gpt4_handler import GPT4Handler
from src.llm_integration.claude_handler import ClaudeHandler
from src.llm_integration.mistral_handler import MistralHandler
from src.llm_integration.groq_handler import GroqHandler
from src.llm_integration.gemini_handler import GeminiHandler

@pytest.fixture
def mock_api_key():
    return "test_api_key"

class TestGPT4Handler:
    @patch('openai.OpenAI')
    def test_generate_response(self, mock_openai, mock_api_key):
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        handler = GPT4Handler(mock_api_key)
        response = handler.generate_response("Test prompt")
        
        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch('openai.OpenAI')
    def test_get_embeddings(self, mock_openai, mock_api_key):
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        
        handler = GPT4Handler(mock_api_key)
        embeddings = handler.get_embeddings("Test text")
        
        assert len(embeddings) == 1536
        mock_client.embeddings.create.assert_called_once()

class TestClaudeHandler:
    @patch('anthropic.Anthropic')
    def test_generate_response(self, mock_anthropic, mock_api_key):
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_client.messages.create.return_value = mock_response
        
        handler = ClaudeHandler(mock_api_key)
        response = handler.generate_response("Test prompt")
        
        assert response == "Test response"
        mock_client.messages.create.assert_called_once()

class TestMistralHandler:
    @patch('mistralai.client.MistralClient')
    def test_generate_response(self, mock_mistral, mock_api_key):
        mock_client = Mock()
        mock_mistral.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_client.chat.return_value = mock_response
        
        handler = MistralHandler(mock_api_key)
        response = handler.generate_response("Test prompt")
        
        assert response == "Test response"
        mock_client.chat.assert_called_once()

class TestGroqHandler:
    @patch('groq.Groq')
    def test_generate_response(self, mock_groq, mock_api_key):
        mock_client = Mock()
        mock_groq.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        handler = GroqHandler(mock_api_key)
        response = handler.generate_response("Test prompt")
        
        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once()

class TestGeminiHandler:
    @patch('google.generativeai.GenerativeModel')
    def test_generate_response(self, mock_genai, mock_api_key):
        mock_model = Mock()
        mock_genai.return_value = mock_model
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_model.generate_content.return_value = mock_response
        
        handler = GeminiHandler(mock_api_key)
        handler.model = mock_model
        response = handler.generate_response("Test prompt")
        
        assert response == "Test response"
        mock_model.generate_content.assert_called_once()
