import pytest
from unittest.mock import patch, MagicMock

from workflows.pipelines.text_classifier import (
    classify_text_rule_based, 
    text_classification_flow,
    CLASSIFICATION_RULES
)

# --- Tests for classify_text_rule_based ---

@pytest.mark.parametrize("text, expected_category", [
    ("This document is about python and REST api development.", "Technical"),
    ("The new market strategy aims to increase revenue.", "Business"),
    ("Please review the legal contract for compliance issues.", "Legal"),
    ("This is a general update about the project status.", "General"), 
    ("PYTHON and API are core to our software.", "Technical"), # Case-insensitivity
    ("Machine learning algorithms are key to data science.", "Technical"),
    ("The product launch boosted customer acquisition and finance.", "Business"),
    # More keywords for technical
    ("Software engineering using python, discussing an api and database.", "Technical"), 
    # Only general keywords
    ("This is some information in a report.", "General"),
    ("", "General"), # Empty text
    ("    ", "General"), # Whitespace text
])
def test_classify_text_rule_based_various_inputs(text, expected_category):
    """Test rule-based classification with various inputs."""
    category = classify_text_rule_based(text, CLASSIFICATION_RULES)
    assert category == expected_category

def test_classify_text_rule_based_priority():
    """Test that a more specific category is chosen over general if keywords match."""
    # Has 'python' (Technical) and 'report' (General)
    text_with_conflict = "This python report is important."
    category = classify_text_rule_based(text_with_conflict, CLASSIFICATION_RULES)
    assert category == "Technical"

# --- Tests for text_classification_flow ---

def test_text_classification_flow_success():
    """Test successful execution of the text classification flow."""
    sample_text = "This is a technical document about Python APIs."
    expected_cat = "Technical"
    
    with patch("workflows.pipelines.text_classifier.classify_text_rule_based", return_value=expected_cat) as mock_classify_task:
        result = text_classification_flow(sample_text)
        
    mock_classify_task.assert_called_once_with(sample_text, CLASSIFICATION_RULES)
    assert result == {
        "status": "success",
        "message": "Text classified successfully.",
        "category": expected_cat
    }

@pytest.mark.parametrize("empty_text", ["", "   "])
def test_text_classification_flow_empty_input(empty_text):
    """Test flow with empty or whitespace-only input text."""
    with patch("workflows.pipelines.text_classifier.classify_text_rule_based") as mock_classify_task:
        result = text_classification_flow(empty_text)
    
    mock_classify_task.assert_not_called()
    assert result == {
        "status": "error",
        "message": "Input text is empty.",
        "category": "N/A"
    }

def test_text_classification_flow_task_error():
    """Test flow when the underlying classification task raises an error."""
    sample_text = "Some text that will cause an error."
    task_exception = Exception("Rule processing failed")
    
    with patch("workflows.pipelines.text_classifier.classify_text_rule_based", side_effect=task_exception) as mock_classify_task:
        result = text_classification_flow(sample_text)
        
    mock_classify_task.assert_called_once_with(sample_text, CLASSIFICATION_RULES)
    assert result == {
        "status": "error",
        "message": f"Flow failed: {str(task_exception)}",
        "category": "Error"
    } 