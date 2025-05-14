"""
Core RAG pipeline tasks: context retrieval and answer generation.
"""
import logging
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv, find_dotenv
# load_dotenv() # REMOVE global load_dotenv() from top of file if present

from langchain_community.vectorstores import FAISS
from langchain.schema import Document, StrOutputParser
from sentence_transformers import CrossEncoder
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate

from .rag_config import INITIAL_RETRIEVAL_K, FINAL_RETRIEVAL_K

logger = logging.getLogger(__name__)

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
        initial_docs = retriever.invoke(question)
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
        # Initialize the LLM with Ollama
        llm = ChatOllama(model="mistral", temperature=0.1) 

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