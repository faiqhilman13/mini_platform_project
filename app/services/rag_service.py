import logging
import os
import uuid
from typing import Dict, Any, Optional, List

from fastapi import HTTPException
from sqlmodel import Session, select

from app.models.pipeline_models import PipelineRun, PipelineType, PipelineRunStatus
from workflows.pipelines.model_loader import initialize_embedding_model, initialize_cross_encoder
from workflows.pipelines.vector_store_manager import (
    load_or_create_vectorstore, 
    check_vectorstore_exists as vs_check_exists, 
    delete_vectorstore as vs_delete,
    list_all_vectorstores as vs_list_all
)
from workflows.pipelines.rag_core import retrieve_context, generate_answer
from workflows.pipelines.rag_config import VECTORSTORE_DIR

logger = logging.getLogger(__name__)

def get_rag_answer(
    db: Session,
    pipeline_run_id: str,
    question: str
) -> Dict[str, Any]:
    """
    Generate an answer to a question using the RAG approach with a previously processed document.

    Args:
        db (Session): Database session
        pipeline_run_id (str): UUID of the RAG pipeline run that processed the document
        question (str): The user's question

    Returns:
        Dict[str, Any]: Response containing answer and source information

    Raises:
        HTTPException: If the pipeline run does not exist or is not a completed RAG pipeline
    """
    try:
        # Convert string to UUID
        run_uuid = uuid.UUID(pipeline_run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pipeline run ID format. Must be a valid UUID.")

    # Find the pipeline run
    statement = select(PipelineRun).where(
        PipelineRun.run_uuid == run_uuid,
        PipelineRun.pipeline_type == PipelineType.RAG_CHATBOT
    )
    pipeline_run = db.exec(statement).one_or_none()

    if not pipeline_run:
        raise HTTPException(status_code=404, detail=f"RAG pipeline run with ID {pipeline_run_id} not found")

    if pipeline_run.status != PipelineRunStatus.COMPLETED:
        raise HTTPException(
            status_code=400, 
            detail=f"RAG pipeline run is not completed. Current status: {pipeline_run.status}"
        )

    # Check if we have result data with the document ID
    if not pipeline_run.result or "doc_id" not in pipeline_run.result:
        raise HTTPException(
            status_code=500, 
            detail="Invalid RAG pipeline result data. Missing document ID."
        )

    doc_id = pipeline_run.result.get("doc_id")
    logger.info(f"Generating answer for question: '{question}' using document {doc_id}")
    
    # Use document ID to find the right vector store
    # In RAG document processing, we store each document's vector store in its own subdirectory
    vector_store_path = os.path.join(str(VECTORSTORE_DIR), str(doc_id))
    
    if not os.path.exists(vector_store_path) or not os.path.exists(os.path.join(vector_store_path, "index.faiss")):
        logger.error(f"Vector store not found at path: {vector_store_path}")
        raise HTTPException(
            status_code=500,
            detail="Vector store not found for the specified document."
        )

    try:
        # Initialize models
        embedding_model = initialize_embedding_model()
        cross_encoder = initialize_cross_encoder()
        
        # Load vector store for the specific document
        vectorstore = load_or_create_vectorstore(
            embedding_model=embedding_model,
            vectorstore_path=vector_store_path
        )
        
        # Retrieve context for the question
        retrieved_docs = retrieve_context(vectorstore, question, cross_encoder=cross_encoder)
        
        # Generate answer using the retrieved context
        answer_result = generate_answer(question, retrieved_docs)
        
        # Return the result
        return {
            "status": answer_result["status"],
            "answer": answer_result["answer"],
            "sources": answer_result["sources"]
        }
    
    except Exception as e:
        logger.exception(f"Error processing RAG question: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

def check_vectorstore_exists(doc_id: str) -> bool:
    """
    Check if a vector store exists for the specified document ID.
    
    Args:
        doc_id (str): The document ID to check
        
    Returns:
        bool: True if the vector store exists, False otherwise
    """
    try:
        return vs_check_exists(doc_id)
    except Exception as e:
        logger.exception(f"Error checking vector store existence: {e}")
        raise

def delete_vectorstore(doc_id: str) -> bool:
    """
    Delete the vector store for the specified document ID.
    
    Args:
        doc_id (str): The document ID whose vector store should be deleted
        
    Returns:
        bool: True if successfully deleted, False if it didn't exist
        
    Raises:
        Exception: If there was an error during deletion
    """
    try:
        return vs_delete(doc_id)
    except Exception as e:
        logger.exception(f"Error deleting vector store: {e}")
        raise

def list_all_vectorstores() -> List[Dict[str, Any]]:
    """
    List all available vector stores in the system.
    
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing information about each vector store.
    
    Raises:
        Exception: If there was an error listing vector stores
    """
    try:
        return vs_list_all()
    except Exception as e:
        logger.exception(f"Error listing vector stores: {e}")
        raise 