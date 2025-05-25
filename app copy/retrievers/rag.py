from langchain_community.vectorstores import FAISS
from app.config import (
    EMBEDDING_MODEL, 
    VECTORSTORE_DIR, 
    INITIAL_RETRIEVAL_K, 
    FINAL_RETRIEVAL_K, 
    CROSS_ENCODER_MODEL_NAME
)
import os
import pickle
from typing import List, Tuple, Optional, Dict, Any
from langchain.schema import Document
from sentence_transformers import CrossEncoder # Import CrossEncoder

class RAGRetriever:
    def __init__(self):
        """Initialize the RAG retriever"""
        self.vectorstore = None
        self.embedding_model = EMBEDDING_MODEL
        self.vectorstore_path = VECTORSTORE_DIR
        self.cross_encoder = None
        # Load CrossEncoder model during initialization
        try:
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
            print(f"Cross-encoder model '{CROSS_ENCODER_MODEL_NAME}' loaded successfully.")
        except Exception as e:
            print(f"Error loading cross-encoder model '{CROSS_ENCODER_MODEL_NAME}': {str(e)}")
            self.cross_encoder = None # Ensure it's None if loading fails
        
    def load_vectorstore(self) -> bool:
        """Load the vectorstore from disk if it exists"""
        if self.vectorstore:
            return True
            
        if not self.embedding_model:
            print("Embedding model not available")
            return False
            
        try:
            index_path = os.path.join(self.vectorstore_path, "index.faiss")
            if os.path.exists(index_path):
                self.vectorstore = FAISS.load_local(
                    self.vectorstore_path, 
                    self.embedding_model, 
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded vectorstore from {self.vectorstore_path}")
                return True
            else:
                print(f"No existing vectorstore found at {self.vectorstore_path}")
                return False
        except Exception as e:
            print(f"Error loading vectorstore: {str(e)}")
            return False
    
    def build_vectorstore(self, docs) -> bool:
        """Build a new vectorstore from documents"""
        if not self.embedding_model:
            print("Embedding model not available")
            return False
            
        if not docs:
            print("No documents provided for building vectorstore")
            return False
            
        try:
            print(f"Building vectorstore with {len(docs)} documents")
            self.vectorstore = FAISS.from_documents(docs, self.embedding_model)
            self.save_vectorstore()
            return True
        except Exception as e:
            print(f"Error building vectorstore: {str(e)}")
            return False
    
    def save_vectorstore(self) -> bool:
        """Save the vectorstore to disk"""
        if not self.vectorstore:
            print("No vectorstore to save")
            return False
            
        try:
            self.vectorstore.save_local(self.vectorstore_path)
            print(f"Saved vectorstore to {self.vectorstore_path}")
            return True
        except Exception as e:
            print(f"Error saving vectorstore: {str(e)}")
            return False
    
    def retrieve_context(self, question: str) -> List[Document]:
        """Retrieve relevant document chunks for a question using re-ranking."""
        if not self.vectorstore:
            if not self.load_vectorstore():
                print("[Retriever] Vectorstore not loaded, returning empty list.")
                return []
                
        # Use INITIAL_RETRIEVAL_K for the first pass
        k_initial = INITIAL_RETRIEVAL_K 
        
        try:
            # Initial retrieval from FAISS
            retriever = self.vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": k_initial}
            )
            initial_docs: List[Document] = retriever.get_relevant_documents(question)
            
            print(f"[Retriever] Retrieved {len(initial_docs)} initial candidates for re-ranking.")

            if not initial_docs:
                return [] # No candidates to re-rank

            # Re-ranking step using CrossEncoder
            if self.cross_encoder:
                print(f"[Retriever] Re-ranking {len(initial_docs)} candidates using '{CROSS_ENCODER_MODEL_NAME}'.")
                # Prepare pairs for the cross-encoder: (query, passage)
                pairs = [(question, doc.page_content) for doc in initial_docs]
                
                # Predict scores
                scores = self.cross_encoder.predict(pairs)
                
                # Combine documents with their scores
                docs_with_scores = list(zip(initial_docs, scores))
                
                # Sort documents by score in descending order
                docs_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Select the top N documents after re-ranking
                reranked_docs = [doc for doc, score in docs_with_scores[:FINAL_RETRIEVAL_K]]
                print(f"[Retriever] Returning top {len(reranked_docs)} re-ranked documents.")
                return reranked_docs
            else:
                # Fallback if cross-encoder failed to load: return top N from initial retrieval
                print(f"[Retriever] Warning: Cross-encoder not loaded. Returning top {FINAL_RETRIEVAL_K} initial results.")
                return initial_docs[:FINAL_RETRIEVAL_K]

        except Exception as e:
            print(f"Error retrieving/re-ranking context: {str(e)}")
            return []

# Create a singleton instance
rag_retriever = RAGRetriever() 