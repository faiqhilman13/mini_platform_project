"""
Functions for initializing and loading embedding and cross-encoder models.
"""
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from .rag_config import EMBEDDING_MODEL_NAME, CROSS_ENCODER_MODEL_NAME

logger = logging.getLogger(__name__)

def initialize_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize and return the embedding model.
    
    Returns:
        HuggingFaceEmbeddings: The initialized embedding model.
    """
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        return embedding_model
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        raise

def initialize_cross_encoder() -> CrossEncoder:
    """
    Initialize and return the cross-encoder model for re-ranking.
    
    Returns:
        CrossEncoder: The initialized cross-encoder model.
    """
    logger.info(f"Initializing cross-encoder model: {CROSS_ENCODER_MODEL_NAME}")
    try:
        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
        return cross_encoder
    except Exception as e:
        logger.error(f"Failed to initialize cross-encoder model: {e}")
        raise 