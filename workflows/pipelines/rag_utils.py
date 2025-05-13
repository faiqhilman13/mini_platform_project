"""
Utility functions for the RAG pipeline, such as PDF text extraction and document chunking.
"""
import logging
import os
from typing import List, Tuple, Dict, Any

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .rag_config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> List[Tuple[str, int]]:
    """
    Extract text from a PDF file, page by page.
    
    Args:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        List[Tuple[str, int]]: List of (text, page_number) tuples.
    
    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        Exception: If extraction fails.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found at {pdf_path}")
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    logger.info(f"Extracting text from PDF: {pdf_path}")
    
    try:
        extracted_pages = []
        with open(pdf_path, "rb") as file:
            pdf = PdfReader(file)
            logger.info(f"PDF has {len(pdf.pages)} pages")
            
            for i, page in enumerate(pdf.pages):
                content = page.extract_text()
                if content:
                    extracted_pages.append((content, i + 1))
                else:
                    logger.warning(f"No text extracted from page {i+1}")
        
        return extracted_pages
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise

def create_document_chunks(extracted_pages: List[Tuple[str, int]], 
                          metadata: Dict[str, Any]) -> List[Document]:
    """
    Create document chunks from extracted text.
    
    Args:
        extracted_pages (List[Tuple[str, int]]): List of (text, page_number) tuples.
        metadata (Dict[str, Any]): Basic metadata for the document.
    
    Returns:
        List[Document]: List of document chunks.
    """
    logger.info(f"Creating document chunks from {len(extracted_pages)} pages")
    
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    for text, page_num in extracted_pages:
        # Create page-specific metadata
        page_metadata = metadata.copy()
        page_metadata["page"] = page_num
        
        # Create chunks for this page
        page_chunks = text_splitter.create_documents([text], [page_metadata])
        all_chunks.extend(page_chunks)
    
    logger.info(f"Created {len(all_chunks)} document chunks")
    return all_chunks 