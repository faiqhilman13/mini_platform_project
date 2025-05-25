"""
Text Classification Pipeline Logic
"""
import logging
from typing import Dict, Any, List

from prefect import task, flow

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Simple Rule-Based Classifier ---
# Define categories and associated keywords
CLASSIFICATION_RULES = {
    "Technical": ["python", "api", "database", "software", "code", "algorithm", "machine learning", "data science"],
    "Business": ["market", "strategy", "revenue", "product", "customer", "finance", "investment"],
    "Legal": ["contract", "agreement", "compliance", "regulation", "law", "court"],
    "General": ["news", "update", "information", "document", "report"] # Fallback category
}

@task
def classify_text_rule_based(text: str, rules: Dict[str, List[str]]) -> str:
    """
    Classify text into a category based on keyword matching.

    Args:
        text (str): The text to classify.
        rules (Dict[str, List[str]]): A dictionary where keys are category names
                                     and values are lists of keywords.

    Returns:
        str: The detected category name. Defaults to 'General' if no specific keywords are found.
    """
    logger.info(f"Classifying text (length: {len(text)}) using rule-based approach.")
    lower_text = text.lower()
    
    category_scores: Dict[str, int] = {category: 0 for category in rules}

    for category, keywords in rules.items():
        if category == "General": # Skip general for scoring, it's a fallback
            continue
        for keyword in keywords:
            if keyword.lower() in lower_text:
                category_scores[category] += 1
                
    # Determine the category with the highest score
    # Filter out categories with zero score before finding max, unless all are zero
    scored_categories = {cat: score for cat, score in category_scores.items() if score > 0 and cat != "General"}

    if not scored_categories:
        logger.info("No specific category keywords found. Defaulting to 'General'.")
        return "General"

    # Find the category with the maximum score
    best_category = max(scored_categories, key=scored_categories.get)
    logger.info(f"Text classified as: {best_category} with score: {scored_categories[best_category]}")
    return best_category

@flow(name="Text Classification Flow")
def text_classification_flow(text_content: str) -> Dict[str, Any]:
    """
    Prefect flow to classify a given text.

    Args:
        text_content (str): The text content to be classified.

    Returns:
        Dict[str, Any]: A dictionary containing the classification result.
                          Includes 'status', 'message', and 'category'.
    """
    logger.info(f"Starting text classification flow for text (length: {len(text_content)}).")
    
    if not text_content or not text_content.strip():
        logger.warning("Input text is empty or whitespace only.")
        return {
            "status": "error",
            "message": "Input text is empty.",
            "category": "N/A"
        }
        
    try:
        # For now, we directly use the rule-based classifier task
        # In the future, this flow could decide between different classification tasks (rule-based, model-based)
        assigned_category = classify_text_rule_based(text_content, CLASSIFICATION_RULES)
        
        return {
            "status": "success",
            "message": "Text classified successfully.",
            "category": assigned_category
        }
    except Exception as e:
        logger.error(f"Error in text classification flow: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Flow failed: {str(e)}",
            "category": "Error"
        }

if __name__ == "__main__":
    # Example Usage
    sample_texts = [
        "This document discusses software development and Python APIs.",
        "The company's new market strategy focuses on customer acquisition.",
        "Please review the attached legal agreement for compliance.",
        "Here is an update on the project.",
        "Exploring advanced data science techniques for a new model.",
        "Financial report for Q3 is now available."
    ]

    for i, text in enumerate(sample_texts):
        print(f"\n--- Example {i+1} ---")
        print(f"Text: {text}")
        result = text_classification_flow(text)
        print(f"Classification Result: {result}")

    print("\n--- Example with empty text ---")
    empty_text_result = text_classification_flow("  ")
    print(f"Classification Result (Empty Text): {empty_text_result}") 