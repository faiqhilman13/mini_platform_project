"""
RAG (Retrieval Augmented Generation) Chatbot Pipeline Logic
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import uuid
from pathlib import Path
import json

from prefect import task, flow
from langchain.schema import Document, StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
INITIAL_RETRIEVAL_K = 20
FINAL_RETRIEVAL_K = 5
LLM_MODEL_NAME = "gpt-3.5-turbo" # Or specify another model

# Paths
BASE_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = DATA_DIR / "vector_store"
DOCUMENTS_DIR = DATA_DIR / "documents"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)
DOCUMENTS_DIR.mkdir(exist_ok=True)

@task
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

@task
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

@task
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

@task
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

@task
def load_or_create_vectorstore(embedding_model: HuggingFaceEmbeddings, 
                              document_chunks: Optional[List[Document]] = None) -> FAISS:
    """
    Load existing vector store or create a new one with the given documents.
    
    Args:
        embedding_model (HuggingFaceEmbeddings): The embedding model.
        document_chunks (List[Document], optional): Document chunks to add. If None, only loads existing store.
    
    Returns:
        FAISS: The FAISS vector store.
    """
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
        vectorstore.save_local(vectorstore_path)
        logger.info(f"Vector store saved to {vectorstore_path}")
        return vectorstore
    else:
        logger.error("No existing vector store and no documents provided")
        raise ValueError("Cannot create vector store without documents")

@task
def retrieve_context(vectorstore: FAISS, question: str, 
                    cross_encoder: Optional[CrossEncoder] = None,
                    top_k: int = FINAL_RETRIEVAL_K) -> List[Document]:
    """
    Retrieve relevant context for a question.
    
    Args:
        vectorstore (FAISS): The vector store.
        question (str): The question to retrieve context for.
        cross_encoder (CrossEncoder, optional): Cross-encoder for re-ranking.
        top_k (int, optional): Number of results to return. Defaults to FINAL_RETRIEVAL_K.
    
    Returns:
        List[Document]: List of relevant documents.
    """
    logger.info(f"Retrieving context for question: {question}")
    
    # Initial retrieval
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": INITIAL_RETRIEVAL_K if cross_encoder else top_k}
        )
        initial_docs = retriever.get_relevant_documents(question)
        logger.info(f"Initial retrieval returned {len(initial_docs)} documents")
        
        if not initial_docs:
            logger.warning("No relevant documents found")
            return []
        
        # Re-rank if cross-encoder is available
        if cross_encoder and len(initial_docs) > top_k:
            logger.info("Re-ranking documents using cross-encoder")
            pairs = [(question, doc.page_content) for doc in initial_docs]
            scores = cross_encoder.predict(pairs)
            
            # Combine documents with their scores
            docs_with_scores = list(zip(initial_docs, scores))
            
            # Sort documents by score in descending order
            docs_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select the top N documents after re-ranking
            reranked_docs = [doc for doc, score in docs_with_scores[:top_k]]
            logger.info(f"Returning {len(reranked_docs)} re-ranked documents")
            return reranked_docs
        else:
            # Limit to top_k if not using re-ranking or if initial retrieval returned fewer docs
            return initial_docs[:top_k]
            
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return []

@task
def process_pdf_for_rag(file_path: str, doc_id: Optional[str] = None, title: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a PDF file for RAG.
    
    Args:
        file_path (str): Path to the PDF file.
        doc_id (str, optional): Document ID. If None, a UUID will be generated.
        title (str, optional): Document title. If None, the filename will be used.
    
    Returns:
        Dict[str, Any]: Processing result.
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
        
        # Extract text from PDF
        extracted_pages = extract_text_from_pdf(file_path)
        
        if not extracted_pages:
            logger.warning(f"No text extracted from {file_path}")
            return {
                "status": "error",
                "message": "No text extracted from PDF",
                "doc_id": doc_id,
                "chunks": 0
            }
        
        # Create document chunks
        chunks = create_document_chunks(extracted_pages, metadata)
        
        # Initialize embedding model
        embedding_model = initialize_embedding_model()
        
        # Load or create vector store
        vectorstore = load_or_create_vectorstore(embedding_model, chunks)
        
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

@task
def generate_answer(question: str, retrieved_context: List[Document]) -> Dict[str, Any]:
    """
    Generate an answer for a question using retrieved context and an LLM.
    
    Args:
        question (str): The question to answer.
        retrieved_context (List[Document]): Retrieved context documents.
    
    Returns:
        Dict[str, Any]: Answer result including the generated answer and sources.
    """
    logger.info(f"Generating answer for question: {question}")
    
    # Extract source info first
    sources = []
    for doc in retrieved_context:
        source = {
            "source": doc.metadata.get("source", "Unknown"),
            "title": doc.metadata.get("title", "Unknown"),
            "page": doc.metadata.get("page", 0)
        }
        sources.append(source)

    if not retrieved_context:
        logger.warning("No relevant context found to generate answer.")
        return {
            "status": "success", # Still success, but with a specific message
            "message": "No relevant context found",
            "answer": "I couldn't find any relevant information in the provided documents to answer your question.",
            "sources": []
        }

    # Format context for the LLM prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_context])
    
    # Define the prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a helpful assistant. Answer the user's question based ONLY on the following context. 
If the context doesn't contain the answer, say you don't know. 
Be concise and mention the source document and page number if possible, but DO NOT make up information.

Context:
{context}"""),
            ("user", "Question: {question}")
        ]
    )

    try:
        # Initialize the LLM
        # Assumes OPENAI_API_KEY is set in the environment
        llm = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0.1) 

        # Create the chain
        chain = prompt_template | llm | StrOutputParser()

        # Generate the answer
        logger.info("Invoking LLM to generate answer...")
        generated_answer = chain.invoke({
            "context": context_text,
            "question": question
        })
        logger.info("LLM generation complete.")

        return {
            "status": "success",
            "message": "Answer generated successfully",
            "answer": generated_answer,
            "sources": sources # Return the sources identified earlier
        }
    except Exception as e:
        logger.error(f"Error generating answer with LLM: {e}")
        # Fallback or error state
        return {
            "status": "error",
            "message": f"Failed to generate answer using LLM: {str(e)}",
            "answer": "Sorry, I encountered an error while trying to generate an answer.",
            "sources": sources # Still return sources if context was retrieved
        }

@flow(name="PDF Processing for RAG")
def process_document_rag_flow(pdf_path: str, doc_id: Optional[str] = None, title: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a document for RAG as a Prefect flow.
    
    Args:
        pdf_path (str): Path to the PDF file.
        doc_id (str, optional): Document ID. If None, a UUID will be generated.
        title (str, optional): Document title. If None, the filename will be used.
    
    Returns:
        Dict[str, Any]: Processing result.
    """
    logger.info(f"Starting document processing flow for RAG: {pdf_path}")
    
    try:
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
    Query the RAG system as a Prefect flow.
    
    Args:
        question (str): The question to answer.
    
    Returns:
        Dict[str, Any]: Query result.
    """
    logger.info(f"Starting RAG query flow for question: {question}")
    
    try:
        # Initialize embedding model
        embedding_model = initialize_embedding_model()
        
        # Initialize cross-encoder
        try:
            cross_encoder = initialize_cross_encoder()
        except Exception as e:
            logger.warning(f"Cross-encoder initialization failed, proceeding without re-ranking: {e}")
            cross_encoder = None
        
        # Load vector store
        try:
            vectorstore = load_or_create_vectorstore(embedding_model)
        except ValueError:
            return {
                "status": "error",
                "message": "Vector store not initialized. Please process documents first.",
                "answer": "I don't have any documents to work with. Please upload and process documents first.",
                "sources": []
            }
        
        # Retrieve context
        retrieved_context = retrieve_context(vectorstore, question, cross_encoder)
        
        # Generate answer
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
    # Example usage for local testing
    import sys
    
    # If run as a script, check if we have a command line argument
    if len(sys.argv) > 1:
        action = sys.argv[1]
        
        if action == "process" and len(sys.argv) > 2:
            # Process a document
            pdf_path = sys.argv[2]
            result = process_document_rag_flow(pdf_path)
            print(f"Processing result: {result}")
        
        elif action == "query" and len(sys.argv) > 2:
            # Query the system
            question = sys.argv[2]
            result = query_rag_flow(question)
            print(f"Query answer: {result['answer']}")
            print("\nSources:")
            for i, source in enumerate(result['sources']):
                print(f"  {i+1}. {source['title']} (page {source['page']})")
        
        else:
            print("Usage: python -m workflows.pipelines.rag_chatbot process <pdf_path>")
            print("       python -m workflows.pipelines.rag_chatbot query <question>")
    
    else:
        # No args, run a demo
        print("Running RAG chatbot demo...")
        
        # Check if we have an example document in the data directory
        example_pdf = DOCUMENTS_DIR / "example.pdf"
        if not example_pdf.exists():
            # Create a simple example PDF
            print("Creating example PDF...")
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            example_pdf.parent.mkdir(exist_ok=True)
            c = canvas.Canvas(str(example_pdf), pagesize=letter)
            
            c.drawString(100, 750, "RAG Chatbot Example Document")
            c.drawString(100, 700, "This is a test document created for the RAG chatbot demo.")
            c.drawString(100, 650, "Prefect is a workflow orchestration tool designed for data pipelines.")
            c.drawString(100, 600, "It helps manage complex workflows with features like scheduling, retries, and monitoring.")
            c.drawString(100, 550, "FAISS is a library for efficient similarity search developed by Facebook Research.")
            c.drawString(100, 500, "It is commonly used for vector search in RAG applications.")
            
            c.showPage()
            c.drawString(100, 750, "Page 2 of Example Document")
            c.drawString(100, 700, "Transformer models have revolutionized NLP with their attention mechanism.")
            c.drawString(100, 650, "LangChain is a framework for developing applications with language models.")
            c.drawString(100, 600, "It provides abstractions for document loaders, embedding models, and vector stores.")
            
            c.save()
            print(f"Created example PDF at {example_pdf}")
        
        # Process the example document
        print("\nProcessing example document...")
        result = process_document_rag_flow(str(example_pdf), title="RAG Demo Document")
        print(f"Processing result: {result}")
        
        # Query the system
        print("\nQuerying the RAG system...")
        questions = [
            "What is Prefect?",
            "What is FAISS used for?",
            "How does LangChain help with language models?"
        ]
        
        for q in questions:
            print(f"\nQuestion: {q}")
            result = query_rag_flow(q)
            print(f"Answer: {result['answer']}")
            print("Sources:")
            for i, source in enumerate(result['sources']):
                print(f"  {i+1}. {source['title']} (page {source['page']})") 