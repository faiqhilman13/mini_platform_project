"""
Manages the FAISS vector store: loading, creating, and updating.
"""
import logging
import os
import shutil
from typing import List, Optional, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from .rag_config import VECTORSTORE_DIR

logger = logging.getLogger(__name__)

def load_or_create_vectorstore(embedding_model: HuggingFaceEmbeddings, 
                              document_chunks: Optional[List[Document]] = None,
                              vectorstore_path: Optional[str] = None) -> FAISS:
    """
    Load existing vector store or create a new one with the given documents.
    
    Args:
        embedding_model (HuggingFaceEmbeddings): The embedding model.
        document_chunks (List[Document], optional): Document chunks to add. If None, only loads existing store.
        vectorstore_path (str, optional): Custom path to the vector store directory. If None, uses default VECTORSTORE_DIR.
    
    Returns:
        FAISS: The FAISS vector store.
    """
    if vectorstore_path is None:
        vectorstore_path = str(VECTORSTORE_DIR)
    
    index_path = os.path.join(vectorstore_path, "index.faiss")
    
    if os.path.exists(index_path) and os.path.isfile(index_path):
        logger.info(f"Loading existing vector store from {vectorstore_path}")
        try:
            vectorstore = FAISS.load_local(
                vectorstore_path, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
            logger.info("Vector store loaded successfully")
            
            # Add new documents if provided
            if document_chunks:
                logger.info(f"Adding {len(document_chunks)} chunks to existing vector store")
                vectorstore.add_documents(document_chunks)
                vectorstore.save_local(vectorstore_path)
                logger.info("Updated vector store saved successfully")
                
            return vectorstore
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            # If loading fails and we have documents, create new
            if document_chunks:
                logger.info("Creating new vector store as loading failed")
            else:
                raise
    
    # Create new vector store if we have documents
    if document_chunks:
        logger.info(f"Creating new vector store with {len(document_chunks)} chunks")
        vectorstore = FAISS.from_documents(document_chunks, embedding_model)
        
        # Ensure directory exists
        os.makedirs(vectorstore_path, exist_ok=True)
        
        vectorstore.save_local(vectorstore_path)
        logger.info(f"Vector store saved to {vectorstore_path}")
        return vectorstore
    else:
        logger.error("No existing vector store and no documents provided")
        raise ValueError("Cannot create vector store without documents or an existing one to load.")

def check_vectorstore_exists(doc_id: str) -> bool:
    """
    Check if a vector store exists for the given document ID.
    
    Args:
        doc_id (str): The document ID to check.
        
    Returns:
        bool: True if the vector store exists, False otherwise.
    """
    vectorstore_path = os.path.join(str(VECTORSTORE_DIR), str(doc_id))
    index_path = os.path.join(vectorstore_path, "index.faiss")
    
    return os.path.exists(index_path) and os.path.isfile(index_path)

def delete_vectorstore(doc_id: str) -> bool:
    """
    Delete the vector store for the given document ID.
    
    Args:
        doc_id (str): The document ID whose vector store should be deleted.
        
    Returns:
        bool: True if successfully deleted, False if it didn't exist.
    """
    vectorstore_path = os.path.join(str(VECTORSTORE_DIR), str(doc_id))
    
    if os.path.exists(vectorstore_path):
        try:
            shutil.rmtree(vectorstore_path)
            logger.info(f"Vector store for document {doc_id} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting vector store for document {doc_id}: {e}")
            raise
    else:
        logger.warning(f"Vector store for document {doc_id} does not exist")
        return False

def list_all_vectorstores() -> List[Dict[str, Any]]:
    """
    List all available vector stores in the system.
    
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing information about each vector store.
                             Each dictionary contains 'doc_id', 'path', and 'size_bytes'.
    """
    if not os.path.exists(str(VECTORSTORE_DIR)):
        logger.warning(f"Vector store directory {str(VECTORSTORE_DIR)} does not exist")
        return []
    
    vectorstores = []
    
    try:
        # List all subdirectories in the vector store directory
        # Each subdirectory is a document ID
        for item in os.listdir(str(VECTORSTORE_DIR)):
            item_path = os.path.join(str(VECTORSTORE_DIR), item)
            
            # Check if it's a directory and contains an index.faiss file
            if os.path.isdir(item_path) and os.path.isfile(os.path.join(item_path, "index.faiss")):
                # Calculate the total size of the vector store
                size_bytes = 0
                for root, _, files in os.walk(item_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        size_bytes += os.path.getsize(file_path)
                
                vectorstores.append({
                    "doc_id": item,
                    "path": item_path,
                    "size_bytes": size_bytes,
                    "size_mb": round(size_bytes / (1024 * 1024), 2)  # Convert to MB with 2 decimal places
                })
        
        return vectorstores
    
    except Exception as e:
        logger.error(f"Error listing vector stores: {e}")
        raise 