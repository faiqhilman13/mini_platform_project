import pytest
from unittest.mock import MagicMock, patch, call

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI # For type hints, mocks

from workflows.pipelines.rag_core import retrieve_context, generate_answer
from workflows.pipelines.rag_config import INITIAL_RETRIEVAL_K, FINAL_RETRIEVAL_K, LLM_MODEL_NAME

@pytest.fixture
def mock_vectorstore() -> MagicMock:
    vs = MagicMock(spec=FAISS)
    retriever = MagicMock()
    vs.as_retriever.return_value = retriever
    return vs

@pytest.fixture
def mock_cross_encoder() -> MagicMock:
    return MagicMock(spec=CrossEncoder)

@pytest.fixture
def mock_llm_chat_openai() -> MagicMock:
    return MagicMock(spec=ChatOpenAI)

@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(page_content="Doc 1 content", metadata={"source": "doc1.pdf", "page": 1, "title": "Doc 1"}),
        Document(page_content="Doc 2 content", metadata={"source": "doc2.pdf", "page": 2, "title": "Doc 2"}),
        Document(page_content="Doc 3 content", metadata={"source": "doc3.pdf", "page": 3, "title": "Doc 3"}),
        Document(page_content="Doc 4 content", metadata={"source": "doc4.pdf", "page": 4, "title": "Doc 4"}),
        Document(page_content="Doc 5 content", metadata={"source": "doc5.pdf", "page": 5, "title": "Doc 5"}),
    ]

# --- Tests for retrieve_context ---

def test_retrieve_context_success_no_reranking(
    mock_vectorstore: MagicMock, 
    sample_documents: list[Document]
):
    """Test context retrieval without reranking."""
    mock_retriever = mock_vectorstore.as_retriever.return_value
    # Simulate retriever returning fewer docs than FINAL_RETRIEVAL_K or exactly FINAL_RETRIEVAL_K
    retrieved_docs = sample_documents[:FINAL_RETRIEVAL_K]
    mock_retriever.get_relevant_documents.return_value = retrieved_docs
    question = "What is test?"

    result_docs = retrieve_context(mock_vectorstore, question, cross_encoder=None, top_k=FINAL_RETRIEVAL_K)

    mock_vectorstore.as_retriever.assert_called_once_with(
        search_type="similarity", 
        search_kwargs={"k": FINAL_RETRIEVAL_K}
    )
    mock_retriever.get_relevant_documents.assert_called_once_with(question)
    assert result_docs == retrieved_docs

def test_retrieve_context_success_with_reranking(
    mock_vectorstore: MagicMock, 
    mock_cross_encoder: MagicMock, 
    sample_documents: list[Document]
):
    """Test context retrieval with cross-encoder reranking."""
    mock_retriever = mock_vectorstore.as_retriever.return_value
    # Ensure initial retrieval is more than FINAL_RETRIEVAL_K for reranking to happen
    # and less than or equal to INITIAL_RETRIEVAL_K
    initial_docs_count = min(INITIAL_RETRIEVAL_K, len(sample_documents))
    initial_retrieved_docs = sample_documents[:initial_docs_count]
    mock_retriever.get_relevant_documents.return_value = initial_retrieved_docs
    
    # Rerank to select fewer than initially retrieved, e.g., FINAL_RETRIEVAL_K
    scores = [0.9, 0.5, 0.8, 0.6, 0.7][:initial_docs_count] # Example scores, length matches initial_docs
    mock_cross_encoder.predict.return_value = scores
    question = "What is test for reranking?"

    # Expected order after reranking (based on scores: doc1, doc3, doc5, doc4, doc2)
    # if initial_docs_count is 5. We want top FINAL_RETRIEVAL_K
    expected_reranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    expected_docs = [initial_retrieved_docs[i] for i in expected_reranked_indices[:FINAL_RETRIEVAL_K]]

    result_docs = retrieve_context(mock_vectorstore, question, cross_encoder=mock_cross_encoder, top_k=FINAL_RETRIEVAL_K)

    mock_vectorstore.as_retriever.assert_called_once_with(
        search_type="similarity", 
        search_kwargs={"k": INITIAL_RETRIEVAL_K} # Uses INITIAL_RETRIEVAL_K for reranking
    )
    mock_retriever.get_relevant_documents.assert_called_once_with(question)
    
    if initial_docs_count > FINAL_RETRIEVAL_K:
        mock_cross_encoder.predict.assert_called_once()
        # Check pairs passed to predict
        expected_pairs = [(question, doc.page_content) for doc in initial_retrieved_docs]
        assert mock_cross_encoder.predict.call_args[0][0] == expected_pairs
        assert len(result_docs) == FINAL_RETRIEVAL_K
        assert result_docs == expected_docs
    else: # If initial docs are already less than or equal to FINAL_K, no reranking happens
        mock_cross_encoder.predict.assert_not_called()
        assert result_docs == initial_retrieved_docs[:FINAL_RETRIEVAL_K]


def test_retrieve_context_no_docs_found(mock_vectorstore: MagicMock):
    """Test retrieval when no documents are found."""
    mock_retriever = mock_vectorstore.as_retriever.return_value
    mock_retriever.get_relevant_documents.return_value = []
    question = "Unknown topic?"

    result_docs = retrieve_context(mock_vectorstore, question)
    assert result_docs == []

def test_retrieve_context_retrieval_error(mock_vectorstore: MagicMock):
    """Test retrieval when the retriever raises an exception."""
    mock_retriever = mock_vectorstore.as_retriever.return_value
    mock_retriever.get_relevant_documents.side_effect = Exception("DB connection error")
    question = "Error question?"

    with patch("workflows.pipelines.rag_core.logger.error") as mock_logger_error:
        result_docs = retrieve_context(mock_vectorstore, question)
    
    assert result_docs == []
    mock_logger_error.assert_called_once_with("Error retrieving context: DB connection error")

# --- Tests for generate_answer ---

@patch("workflows.pipelines.rag_core.ChatOpenAI")
@patch("workflows.pipelines.rag_core.ChatPromptTemplate.from_messages")
@patch("workflows.pipelines.rag_core.StrOutputParser")
def test_generate_answer_success(
    mock_str_output_parser_constructor: MagicMock,
    mock_chat_prompt_template_from_messages: MagicMock,
    mock_chat_open_ai_constructor: MagicMock,
    sample_documents: list[Document]
):
    """Test successful answer generation."""
    question = "What is in doc1?"
    retrieved_context = sample_documents[:2]
    mock_generated_answer_text = "Doc 1 contains important information."

    # Mock instances returned by constructors/methods
    mock_llm_instance = MagicMock()
    mock_chat_open_ai_constructor.return_value = mock_llm_instance
    
    mock_prompt_instance = MagicMock()
    mock_chat_prompt_template_from_messages.return_value = mock_prompt_instance
    
    mock_parser_instance = MagicMock()
    mock_str_output_parser_constructor.return_value = mock_parser_instance

    # Mock the chain behavior
    mock_chain = MagicMock()
    # In the code, the chain is created as: chain = prompt_template | llm | StrOutputParser()
    # Patch the __or__ operator to return our mock_chain
    mock_prompt_instance.__or__.return_value = mock_chain
    mock_chain.__or__.return_value = mock_chain
    mock_chain.invoke.return_value = mock_generated_answer_text

    result = generate_answer(question, retrieved_context)

    mock_chat_open_ai_constructor.assert_called_once_with(model_name=LLM_MODEL_NAME, temperature=0.1)
    mock_chat_prompt_template_from_messages.assert_called_once()
    mock_str_output_parser_constructor.assert_called_once_with() # Ensure parser is instantiated

    # Check that the chain was created correctly
    mock_prompt_instance.__or__.assert_called_once_with(mock_llm_instance)
    mock_chain.__or__.assert_called_once_with(mock_parser_instance)

    # Check the chain invocation
    expected_chain_input_dict = {
        "context": "Doc 1 content\n\n---\n\nDoc 2 content",
        "question": question
    }
    mock_chain.invoke.assert_called_once_with(expected_chain_input_dict)

    assert result["status"] == "success"
    assert result["answer"] == mock_generated_answer_text
    assert len(result["sources"]) == 2
    assert result["sources"][0] == {"source": "doc1.pdf", "page": 1, "title": "Doc 1"}
    assert result["sources"][1] == {"source": "doc2.pdf", "page": 2, "title": "Doc 2"}

def test_generate_answer_no_context():
    """Test answer generation when no context is provided."""
    question = "What if there is no context?"
    result = generate_answer(question, [])
    
    assert result["status"] == "success" # Special success message
    assert result["message"] == "No relevant context found"
    assert "couldn't find any relevant information" in result["answer"]
    assert result["sources"] == []

@patch("workflows.pipelines.rag_core.ChatOpenAI")
@patch("workflows.pipelines.rag_core.ChatPromptTemplate.from_messages")
@patch("workflows.pipelines.rag_core.StrOutputParser")
def test_generate_answer_llm_error(
    mock_str_output_parser_constructor: MagicMock,
    mock_chat_prompt_template_from_messages: MagicMock,
    mock_chat_open_ai_constructor: MagicMock, 
    sample_documents: list[Document]
):
    """Test answer generation when the LLM call fails."""
    question = "What causes an LLM error?"
    retrieved_context = sample_documents[:1]
    llm_exception = Exception("LLM API timeout")

    # Mock instances returned by constructors/methods
    mock_llm_instance = MagicMock()
    mock_chat_open_ai_constructor.return_value = mock_llm_instance
    
    mock_prompt_instance = MagicMock()
    mock_chat_prompt_template_from_messages.return_value = mock_prompt_instance
    
    mock_parser_instance = MagicMock()
    mock_str_output_parser_constructor.return_value = mock_parser_instance

    # Mock the chain behavior, but with an error
    mock_chain = MagicMock()
    mock_prompt_instance.__or__.return_value = mock_chain
    mock_chain.__or__.return_value = mock_chain
    mock_chain.invoke.side_effect = llm_exception

    result = generate_answer(question, retrieved_context)

    mock_chat_open_ai_constructor.assert_called_once_with(model_name=LLM_MODEL_NAME, temperature=0.1)
    
    # Check that the chain was created correctly
    mock_prompt_instance.__or__.assert_called_once_with(mock_llm_instance)
    mock_chain.__or__.assert_called_once_with(mock_parser_instance)
    
    # Verify chain was invoked and raised the exception
    mock_chain.invoke.assert_called_once()

    assert result["status"] == "error"
    assert f"Failed to generate answer using LLM: {str(llm_exception)}" in result["message"]
    assert "Sorry, I encountered an error" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "doc1.pdf" 