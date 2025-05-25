import pytest
from unittest.mock import patch, MagicMock

# Assuming rag_config contains the model names
from workflows.pipelines.rag_config import EMBEDDING_MODEL_NAME, CROSS_ENCODER_MODEL_NAME
from workflows.pipelines.model_loader import initialize_embedding_model, initialize_cross_encoder

# --- Tests for initialize_embedding_model ---

@patch("workflows.pipelines.model_loader.HuggingFaceEmbeddings")
def test_initialize_embedding_model_success(mock_hugging_face_embeddings_constructor):
    """Test successful initialization of the embedding model."""
    mock_embedding_instance = MagicMock()
    mock_hugging_face_embeddings_constructor.return_value = mock_embedding_instance
    
    embedding_model = initialize_embedding_model()
    
    mock_hugging_face_embeddings_constructor.assert_called_once_with(model_name=EMBEDDING_MODEL_NAME)
    assert embedding_model == mock_embedding_instance

@patch("workflows.pipelines.model_loader.HuggingFaceEmbeddings")
def test_initialize_embedding_model_failure(mock_hugging_face_embeddings_constructor):
    """Test failure during embedding model initialization."""
    mock_hugging_face_embeddings_constructor.side_effect = Exception("Embedding init error")
    
    with pytest.raises(Exception) as exc_info:
        initialize_embedding_model()
    assert "Embedding init error" in str(exc_info.value)

# --- Tests for initialize_cross_encoder ---

@patch("workflows.pipelines.model_loader.CrossEncoder")
def test_initialize_cross_encoder_success(mock_cross_encoder_constructor):
    """Test successful initialization of the cross-encoder model."""
    mock_cross_encoder_instance = MagicMock()
    mock_cross_encoder_constructor.return_value = mock_cross_encoder_instance
    
    cross_encoder_model = initialize_cross_encoder()
    
    mock_cross_encoder_constructor.assert_called_once_with(CROSS_ENCODER_MODEL_NAME)
    assert cross_encoder_model == mock_cross_encoder_instance

@patch("workflows.pipelines.model_loader.CrossEncoder")
def test_initialize_cross_encoder_failure(mock_cross_encoder_constructor):
    """Test failure during cross-encoder model initialization."""
    mock_cross_encoder_constructor.side_effect = Exception("CrossEncoder init error")
    
    with pytest.raises(Exception) as exc_info:
        initialize_cross_encoder()
    assert "CrossEncoder init error" in str(exc_info.value) 