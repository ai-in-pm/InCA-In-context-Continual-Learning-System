import pytest
from unittest.mock import Mock, patch
from src.inca_agent import InCAAgent
from src.llm_integration.gpt4_handler import GPT4Handler

@pytest.fixture
def mock_llm_handler():
    handler = Mock(spec=GPT4Handler)
    handler.get_embeddings.return_value = [0.1] * 1536
    handler.generate_response.return_value = "Test response"
    return handler

@pytest.fixture
def inca_agent(mock_llm_handler):
    llm_handlers = {"gpt4": mock_llm_handler}
    return InCAAgent(llm_handlers)

class TestInCAAgentIntegration:
    def test_add_and_classify(self, inca_agent):
        # Add a test class
        inca_agent.add_class(
            class_name="positive",
            description="Positive sentiment",
            examples=["Great!", "Excellent!", "Amazing!"]
        )
        
        # Test classification
        result = inca_agent.classify("This is great!")
        assert isinstance(result, dict)
        assert "positive" in result
        assert 0 <= result["positive"] <= 1
    
    def test_multiple_classes(self, inca_agent):
        # Add multiple classes
        classes = {
            "positive": ["Great!", "Excellent!"],
            "negative": ["Bad!", "Terrible!"],
            "neutral": ["Okay.", "Fine."]
        }
        
        for class_name, examples in classes.items():
            inca_agent.add_class(
                class_name=class_name,
                description=f"{class_name} sentiment",
                examples=examples
            )
        
        # Test classification returns scores for all classes
        result = inca_agent.classify("This is great!")
        assert len(result) == 3  # Should return top 3 classes by default
        assert all(0 <= score <= 1 for score in result.values())
    
    def test_mahalanobis_distance(self, inca_agent):
        # Add a test class
        inca_agent.add_class(
            class_name="test_class",
            description="Test class",
            examples=["Test example"]
        )
        
        # Get distance
        distance = inca_agent.get_mahalanobis_distance(
            text="Test example",
            class_name="test_class"
        )
        
        assert isinstance(distance, float)
        assert distance >= 0
    
    def test_class_info(self, inca_agent):
        class_name = "test_class"
        description = "Test class"
        examples = ["Test example 1", "Test example 2"]
        
        # Add class
        inca_agent.add_class(
            class_name=class_name,
            description=description,
            examples=examples
        )
        
        # Get class info
        info = inca_agent.get_class_info(class_name)
        assert info is not None
        assert info["description"] == description
        assert info["example_count"] == len(examples)
