"""
RAG (Retrieval Augmented Generation) Chatbot Pipeline Logic
Main file containing Prefect flows and tasks orchestration.
"""

import logging
import os
import uuid
from typing import Dict, Any, Optional

from prefect import task, flow
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv() 

# Initialize and ensure data directories from rag_config before other imports that might use them
from .rag_config import DOCUMENTS_DIR, ensure_data_directories_exist, VECTORSTORE_DIR
ensure_data_directories_exist() # Call this early

from .model_loader import initialize_embedding_model, initialize_cross_encoder
from .rag_utils import extract_text_from_pdf, create_document_chunks
from .vector_store_manager import load_or_create_vectorstore
from .rag_core import retrieve_context, generate_answer

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@task
def process_pdf_for_rag(file_path: str, doc_id: Optional[str] = None, title: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a PDF file for RAG: extract, chunk, embed, and store.
    This is a Prefect task that orchestrates calls to other utility functions.
    """
    logger.info(f"Processing PDF for RAG: {file_path}")
    
    try:
        # Generate doc_id and title if not provided
        if not doc_id:
            doc_id = str(uuid.uuid4())
        
        filename = os.path.basename(file_path)
        if not title:
            title = filename
        
        # Basic metadata
        metadata = {
            "source": filename,
            "path": file_path,
            "title": title,
            "doc_id": doc_id
        }
        
        # Extract text from PDF (from rag_utils)
        extracted_pages = extract_text_from_pdf(file_path)
        
        if not extracted_pages:
            logger.warning(f"No text extracted from {file_path}")
            return {
                "status": "error",
                "message": "No text extracted from PDF",
                "doc_id": doc_id,
                "chunks": 0
            }
        
        # Create document chunks (from rag_utils)
        chunks = create_document_chunks(extracted_pages, metadata)
        
        # Initialize embedding model (from model_loader)
        embedding_model = initialize_embedding_model()
        
        # Construct the specific vector store path for this document
        doc_vectorstore_path = os.path.join(str(VECTORSTORE_DIR), doc_id)
        
        # Load or create vector store (from vector_store_manager)
        # This function now uses document_chunks to add to the store if they exist
        _ = load_or_create_vectorstore(
            embedding_model=embedding_model, 
            document_chunks=chunks,
            vectorstore_path=doc_vectorstore_path # Pass the specific path
        )
        
        return {
            "status": "success",
            "message": f"PDF processed successfully. Created {len(chunks)} chunks.",
            "doc_id": doc_id,
            "chunks": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return {
            "status": "error",
            "message": f"Failed to process PDF: {str(e)}",
            "doc_id": doc_id if doc_id else "unknown",
            "chunks": 0
        }

@flow(name="PDF Processing for RAG Flow")
def process_document_rag_flow(pdf_path: str, doc_id: Optional[str] = None, title: Optional[str] = None) -> Dict[str, Any]:
    """
    Prefect flow to process a document for RAG.
    """
    logger.info(f"Starting document processing flow for RAG: {pdf_path}")
    
    try:
        # The main processing logic is encapsulated in the process_pdf_for_rag task
        result = process_pdf_for_rag(pdf_path, doc_id, title)
        return result
    except Exception as e:
        logger.error(f"Error in PDF processing flow: {e}")
        return {
            "status": "error",
            "message": f"Flow failed: {str(e)}",
            "doc_id": doc_id if doc_id else "unknown",
            "chunks": 0
        }

@flow(name="RAG Query Flow")
def query_rag_flow(question: str) -> Dict[str, Any]:
    """
    Prefect flow to query the RAG system.
    """
    logger.info(f"Starting RAG query flow for question: {question}")
    
    try:
        # Initialize embedding model (from model_loader)
        embedding_model = initialize_embedding_model()
        
        # Initialize cross-encoder (from model_loader)
        try:
            cross_encoder = initialize_cross_encoder()
        except Exception as e:
            logger.warning(f"Cross-encoder initialization failed, proceeding without re-ranking: {e}")
            cross_encoder = None
        
        # Load vector store (from vector_store_manager)
        # When querying, we don't pass document_chunks, so it only loads existing.
        try:
            vectorstore = load_or_create_vectorstore(embedding_model)
        except ValueError as ve:
            logger.error(f"Failed to load vector store: {ve}")
            return {
                "status": "error",
                "message": "Vector store not initialized or empty. Please process documents first.",
                "answer": "I don't have any documents to work with. Please upload and process documents first.",
                "sources": []
            }
        
        # Retrieve context (from rag_core)
        retrieved_context = retrieve_context(vectorstore, question, cross_encoder)
        
        # Generate answer (from rag_core)
        answer_result = generate_answer(question, retrieved_context)
        
        return answer_result
    except Exception as e:
        logger.error(f"Error in RAG query flow: {e}")
        return {
            "status": "error",
            "message": f"Flow failed: {str(e)}",
            "answer": "Sorry, something went wrong while processing your question.",
            "sources": []
        }

if __name__ == "__main__":
    import sys
    
    # If run as a script, check if we have a command line argument
    if len(sys.argv) > 1:
        action = sys.argv[1]
        
        if action == "process" and len(sys.argv) > 2:
            pdf_path_arg = sys.argv[2]
            demo_result = process_document_rag_flow(pdf_path_arg)
            print(f"Processing result: {demo_result}")
        
        elif action == "query" and len(sys.argv) > 2:
            question_arg = sys.argv[2]
            demo_result = query_rag_flow(question_arg)
            print(f"Query answer: {demo_result.get('answer', 'N/A')}")
            print("\nSources:")
            for i, src in enumerate(demo_result.get('sources', [])):
                print(f"  {i+1}. {src.get('title', 'N/A')} (page {src.get('page', 'N/A')})")
        
        else:
            print("Usage: python -m workflows.pipelines.rag_chatbot process <pdf_path>")
            print("       python -m workflows.pipelines.rag_chatbot query <question>")
    
    else:
        # No args, run a demo
        print("Running RAG chatbot demo...")
        
        # Check if we have an example document in the data directory
        example_pdf = DOCUMENTS_DIR / "example.pdf"
        if not example_pdf.exists():
            print("Creating example PDF for demo...")
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            example_pdf.parent.mkdir(exist_ok=True)
            c = canvas.Canvas(str(example_pdf), pagesize=letter)
            c.drawString(100, 750, "RAG Chatbot Example Document")
            c.drawString(100, 700, "This is a test document for demo.")
            c.drawString(100, 650, "Prefect is a workflow tool.")
            c.drawString(100, 600, "FAISS is for similarity search.")
            c.save()
            print(f"Created example PDF at {example_pdf}")
        
        print("\nProcessing example document for demo...")
        process_result = process_document_rag_flow(str(example_pdf), title="RAG Demo Document")
        print(f"Processing result: {process_result}")
        
        print("\nQuerying the RAG system (demo)...")
        demo_questions = [
            "What is Prefect?",
            "Tell me about FAISS."
        ]
        
        for q_text in demo_questions:
            print(f"\nQuestion: {q_text}")
            query_result = query_rag_flow(q_text)
            print(f"Answer: {query_result.get('answer', 'N/A')}")
            print("Sources:")
            for i, src_item in enumerate(query_result.get('sources', [])):
                print(f"  {i+1}. {src_item.get('title', 'N/A')} (page {src_item.get('page', 'N/A')})") 