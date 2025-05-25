import pytest
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings # For type hint
from langchain_community.vectorstores import FAISS

from workflows.pipelines.vector_store_manager import load_or_create_vectorstore
from workflows.pipelines.rag_config import VECTORSTORE_DIR

# Mock embedding model instance for tests
@pytest.fixture
def mock_embedding_model() -> MagicMock:
    return MagicMock(spec=HuggingFaceEmbeddings)

@pytest.fixture
def mock_document_chunks() -> list[Document]:
    return [Document(page_content="Chunk 1"), Document(page_content="Chunk 2")]

# --- Tests for load_or_create_vectorstore ---

def test_load_existing_vectorstore_no_new_docs(mock_embedding_model: MagicMock):
    """Test loading an existing vector store without adding new documents."""
    mock_vs_instance = MagicMock(spec=FAISS)
    
    with patch("os.path.exists", return_value=True) as mock_exists, \
         patch("os.path.isfile", return_value=True) as mock_isfile, \
         patch("workflows.pipelines.vector_store_manager.FAISS.load_local", return_value=mock_vs_instance) as mock_load_local:
        
        vectorstore = load_or_create_vectorstore(mock_embedding_model, document_chunks=None)
        
    mock_exists.assert_called_once_with(os.path.join(str(VECTORSTORE_DIR), "index.faiss"))
    mock_isfile.assert_called_once_with(os.path.join(str(VECTORSTORE_DIR), "index.faiss"))
    mock_load_local.assert_called_once_with(
        str(VECTORSTORE_DIR), 
        mock_embedding_model, 
        allow_dangerous_deserialization=True
    )
    assert vectorstore == mock_vs_instance
    mock_vs_instance.add_documents.assert_not_called()
    mock_vs_instance.save_local.assert_not_called() # Not called if no new docs

def test_load_existing_vectorstore_with_new_docs(mock_embedding_model: MagicMock, mock_document_chunks: list[Document]):
    """Test loading an existing vector store and adding new documents."""
    mock_vs_instance = MagicMock(spec=FAISS)
    
    with patch("os.path.exists", return_value=True) as mock_exists, \
         patch("os.path.isfile", return_value=True) as mock_isfile, \
         patch("workflows.pipelines.vector_store_manager.FAISS.load_local", return_value=mock_vs_instance) as mock_load_local:
        
        vectorstore = load_or_create_vectorstore(mock_embedding_model, mock_document_chunks)
        
    mock_load_local.assert_called_once()
    assert vectorstore == mock_vs_instance
    mock_vs_instance.add_documents.assert_called_once_with(mock_document_chunks)
    mock_vs_instance.save_local.assert_called_once_with(str(VECTORSTORE_DIR))

def test_create_new_vectorstore_no_existing(mock_embedding_model: MagicMock, mock_document_chunks: list[Document]):
    """Test creating a new vector store when none exists."""
    mock_vs_instance = MagicMock(spec=FAISS)
    
    with patch("os.path.exists", return_value=False) as mock_exists, \
         patch("workflows.pipelines.vector_store_manager.FAISS.from_documents", return_value=mock_vs_instance) as mock_from_docs:
        
        vectorstore = load_or_create_vectorstore(mock_embedding_model, mock_document_chunks)
        
    mock_exists.assert_called_once_with(os.path.join(str(VECTORSTORE_DIR), "index.faiss"))
    mock_from_docs.assert_called_once_with(mock_document_chunks, mock_embedding_model)
    assert vectorstore == mock_vs_instance
    mock_vs_instance.save_local.assert_called_once_with(str(VECTORSTORE_DIR))

def test_load_fails_creates_new_with_docs(mock_embedding_model: MagicMock, mock_document_chunks: list[Document]):
    """Test creating a new store if loading fails and documents are provided."""
    mock_new_vs_instance = MagicMock(spec=FAISS)
    
    with patch("os.path.exists", return_value=True) as mock_exists, \
         patch("os.path.isfile", return_value=True) as mock_isfile, \
         patch("workflows.pipelines.vector_store_manager.FAISS.load_local", side_effect=Exception("Load failed")) as mock_load_local, \
         patch("workflows.pipelines.vector_store_manager.FAISS.from_documents", return_value=mock_new_vs_instance) as mock_from_docs:
        
        vectorstore = load_or_create_vectorstore(mock_embedding_model, mock_document_chunks)
        
    mock_load_local.assert_called_once()
    mock_from_docs.assert_called_once_with(mock_document_chunks, mock_embedding_model)
    assert vectorstore == mock_new_vs_instance
    mock_new_vs_instance.save_local.assert_called_once_with(str(VECTORSTORE_DIR))

def test_load_fails_no_docs_raises_error(mock_embedding_model: MagicMock):
    """Test that loading failure re-raises error if no new documents are provided."""
    load_exception = Exception("Persistent Load Error")
    with patch("os.path.exists", return_value=True) as mock_exists, \
         patch("os.path.isfile", return_value=True) as mock_isfile, \
         patch("workflows.pipelines.vector_store_manager.FAISS.load_local", side_effect=load_exception) as mock_load_local:
        
        with pytest.raises(Exception) as exc_info:
            load_or_create_vectorstore(mock_embedding_model, document_chunks=None)
        
    assert exc_info.value == load_exception # Should be the original exception

def test_create_new_no_existing_no_docs_raises_error(mock_embedding_model: MagicMock):
    """Test ValueError if no existing store and no documents to create one."""
    with patch("os.path.exists", return_value=False) as mock_exists:
        with pytest.raises(ValueError) as exc_info:
            load_or_create_vectorstore(mock_embedding_model, document_chunks=None)
            
    assert "Cannot create vector store without documents" in str(exc_info.value) 