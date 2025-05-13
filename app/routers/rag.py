# CURSOR: This file should only handle route wiring, not business logic.
# All logic must be called from services/ or utils/

import logging
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlmodel import Session
from pydantic import BaseModel
from typing import Dict, Any, List

from app.core.config import settings
from app.db.session import get_session
from app.services import rag_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix=f"{settings.API_V1_STR}/rag",
    tags=["RAG"],
)

class RagQuestionRequest(BaseModel):
    pipeline_run_id: str
    question: str

class RagAnswerResponse(BaseModel):
    status: str
    answer: str
    sources: List[Dict[str, Any]]

class VectorStoreStatusResponse(BaseModel):
    exists: bool

class VectorStoreInfo(BaseModel):
    doc_id: str
    path: str
    size_bytes: int
    size_mb: float

class VectorStoreListResponse(BaseModel):
    vectorstores: List[VectorStoreInfo]
    count: int

@router.post("/ask", response_model=RagAnswerResponse)
def ask_rag_question(
    request_body: RagQuestionRequest,
    db: Session = Depends(get_session)
) -> RagAnswerResponse:
    """
    Ask a question to a RAG chatbot using a processed document.
    
    This endpoint:
    1. Takes a question and pipeline run ID
    2. Retrieves the vector store associated with the document
    3. Uses RAG to find relevant context
    4. Generates an answer based on that context
    
    The pipeline_run_id must refer to a completed RAG_CHATBOT pipeline run.
    """
    logger.info(f"Received RAG question request for pipeline run: {request_body.pipeline_run_id}")
    
    try:
        response = rag_service.get_rag_answer(
            db=db,
            pipeline_run_id=request_body.pipeline_run_id,
            question=request_body.question
        )
        
        logger.info(f"Generated RAG answer for pipeline run: {request_body.pipeline_run_id}")
        return RagAnswerResponse(**response)
    
    except HTTPException as http_exc:
        # Re-raise HTTPExceptions
        logger.error(f"HTTPException from service: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
        
    except Exception as e:
        logger.exception(f"Unexpected error generating RAG answer: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.get("/vectorstore/{doc_id}/status", response_model=VectorStoreStatusResponse)
def check_vectorstore_status(
    doc_id: str,
    db: Session = Depends(get_session)
) -> VectorStoreStatusResponse:
    """
    Check if a vector store exists for a specific document ID.
    
    Args:
        doc_id: The document ID to check
        
    Returns:
        VectorStoreStatusResponse with exists=True if the vector store exists, False otherwise
    """
    logger.info(f"Checking vector store status for document ID: {doc_id}")
    
    try:
        exists = rag_service.check_vectorstore_exists(doc_id)
        return VectorStoreStatusResponse(exists=exists)
    
    except Exception as e:
        logger.exception(f"Error checking vector store status: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking vector store status: {str(e)}")

@router.delete("/vectorstore/{doc_id}", status_code=204)
def delete_vectorstore(
    doc_id: str,
    db: Session = Depends(get_session)
) -> None:
    """
    Delete the vector store for a specific document ID.
    
    Args:
        doc_id: The document ID whose vector store should be deleted
        
    Returns:
        204 No Content on successful deletion
    """
    logger.info(f"Deleting vector store for document ID: {doc_id}")
    
    try:
        success = rag_service.delete_vectorstore(doc_id)
        if not success:
            logger.warning(f"Vector store for document {doc_id} did not exist")
            
    except Exception as e:
        logger.exception(f"Error deleting vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting vector store: {str(e)}")

@router.get("/vectorstores", response_model=VectorStoreListResponse)
def list_all_vectorstores(
    db: Session = Depends(get_session)
) -> VectorStoreListResponse:
    """
    List all available vector stores in the system.
    
    Returns:
        VectorStoreListResponse containing a list of all vector stores and their count
    """
    logger.info("Listing all vector stores")
    
    try:
        vectorstores = rag_service.list_all_vectorstores()
        return VectorStoreListResponse(
            vectorstores=vectorstores,
            count=len(vectorstores)
        )
    
    except Exception as e:
        logger.exception(f"Error listing vector stores: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing vector stores: {str(e)}") 