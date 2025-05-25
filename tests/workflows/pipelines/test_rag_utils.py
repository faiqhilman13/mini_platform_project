import pytest
import os
from unittest.mock import MagicMock, patch, mock_open

from langchain.schema import Document

from workflows.pipelines.rag_utils import extract_text_from_pdf, create_document_chunks
# Assuming CHUNK_SIZE and CHUNK_OVERLAP are accessible for some tests if needed,
# or we rely on their default values from rag_config
# from workflows.pipelines.rag_config import CHUNK_SIZE, CHUNK_OVERLAP

# --- Tests for extract_text_from_pdf ---

def test_extract_text_from_pdf_success():
    """Test successful text extraction from a PDF."""
    mock_pdf_content = b"dummy pdf content"
    mock_pdf_path = "/fake/path/to/test.pdf"

    # Mock PdfReader and its pages
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Text from page 1."
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Text from page 2."
    
    mock_pdf_reader_instance = MagicMock()
    mock_pdf_reader_instance.pages = [mock_page1, mock_page2]

    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=mock_pdf_content)), \
         patch("workflows.pipelines.rag_utils.PdfReader", return_value=mock_pdf_reader_instance) as mock_pdf_reader_constructor:
        
        extracted_pages = extract_text_from_pdf(mock_pdf_path)

    assert extracted_pages == [
        ("Text from page 1.", 1),
        ("Text from page 2.", 2)
    ]
    mock_pdf_reader_constructor.assert_called_once()
    # open should be called with mock_pdf_path and "rb"
    # os.path.exists should be called with mock_pdf_path

def test_extract_text_from_pdf_empty_page():
    """Test extraction when a page has no text."""
    mock_pdf_content = b"dummy pdf content"
    mock_pdf_path = "/fake/path/to/empty_page.pdf"

    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Text from page 1."
    mock_page_empty = MagicMock()
    mock_page_empty.extract_text.return_value = None # Simulate empty page
    mock_page3 = MagicMock()
    mock_page3.extract_text.return_value = "Text from page 3."

    mock_pdf_reader_instance = MagicMock()
    mock_pdf_reader_instance.pages = [mock_page1, mock_page_empty, mock_page3]

    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=mock_pdf_content)), \
         patch("workflows.pipelines.rag_utils.PdfReader", return_value=mock_pdf_reader_instance):
        
        extracted_pages = extract_text_from_pdf(mock_pdf_path)

    assert extracted_pages == [
        ("Text from page 1.", 1),
        ("Text from page 3.", 3) # Page 2 is skipped
    ]

def test_extract_text_from_pdf_file_not_found():
    """Test FileNotFoundError when PDF does not exist."""
    mock_pdf_path = "/fake/path/to/nonexistent.pdf"
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError) as exc_info:
            extract_text_from_pdf(mock_pdf_path)
    assert mock_pdf_path in str(exc_info.value)

def test_extract_text_from_pdf_extraction_error():
    """Test handling of an error during PdfReader processing."""
    mock_pdf_content = b"dummy pdf content"
    mock_pdf_path = "/fake/path/to/corrupted.pdf"
    
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=mock_pdf_content)), \
         patch("workflows.pipelines.rag_utils.PdfReader", side_effect=Exception("PDF parsing error")):
        
        with pytest.raises(Exception) as exc_info:
            extract_text_from_pdf(mock_pdf_path)
    assert "PDF parsing error" in str(exc_info.value)

# --- Tests for create_document_chunks ---

@patch("workflows.pipelines.rag_utils.RecursiveCharacterTextSplitter")
def test_create_document_chunks_success(mock_text_splitter_constructor):
    """Test successful creation of document chunks."""
    extracted_pages = [
        ("This is the first page. It has some text.", 1),
        ("This is the second page. More text here.", 2)
    ]
    metadata = {"source": "test.pdf", "doc_id": "123"}

    # Mock the behavior of the text splitter instance
    mock_splitter_instance = MagicMock()
    
    # Simulate create_documents behavior for each page
    # Page 1 chunks
    doc1_page1 = Document(page_content="This is the first page.", metadata={"source": "test.pdf", "doc_id": "123", "page": 1})
    doc2_page1 = Document(page_content="It has some text.", metadata={"source": "test.pdf", "doc_id": "123", "page": 1})
    # Page 2 chunks
    doc1_page2 = Document(page_content="This is the second page.", metadata={"source": "test.pdf", "doc_id": "123", "page": 2})
    doc2_page2 = Document(page_content="More text here.", metadata={"source": "test.pdf", "doc_id": "123", "page": 2})

    # This needs to be more robust if CHUNK_SIZE/OVERLAP are small.
    # For this test, assume each page's text is split into two illustrative chunks.
    def create_documents_side_effect(texts, metadatas):
        page_num = metadatas[0]["page"] # Get page from the provided metadata
        if page_num == 1:
            return [doc1_page1, doc2_page1]
        elif page_num == 2:
            return [doc1_page2, doc2_page2]
        return []
        
    mock_splitter_instance.create_documents.side_effect = create_documents_side_effect
    mock_text_splitter_constructor.return_value = mock_splitter_instance
    
    chunks = create_document_chunks(extracted_pages, metadata)

    assert len(chunks) == 4
    assert chunks[0] == doc1_page1
    assert chunks[1] == doc2_page1
    assert chunks[2] == doc1_page2
    assert chunks[3] == doc2_page2
    
    # Check that the constructor was called with parameters from rag_config
    # This requires rag_config.CHUNK_SIZE and CHUNK_OVERLAP to be imported or mocked
    # For now, let's assume they are used. We can refine this if direct access is an issue.
    mock_text_splitter_constructor.assert_called_once() # Potentially check args too

    # Check calls to create_documents
    assert mock_splitter_instance.create_documents.call_count == 2
    
    first_call_args = mock_splitter_instance.create_documents.call_args_list[0]
    assert first_call_args[0][0] == ["This is the first page. It has some text."]
    assert first_call_args[0][1] == [{"source": "test.pdf", "doc_id": "123", "page": 1}]

    second_call_args = mock_splitter_instance.create_documents.call_args_list[1]
    assert second_call_args[0][0] == ["This is the second page. More text here."]
    assert second_call_args[0][1] == [{"source": "test.pdf", "doc_id": "123", "page": 2}]


def test_create_document_chunks_no_pages():
    """Test chunk creation with no extracted pages."""
    extracted_pages = []
    metadata = {"source": "test.pdf"}
    chunks = create_document_chunks(extracted_pages, metadata)
    assert chunks == []

@patch("workflows.pipelines.rag_utils.RecursiveCharacterTextSplitter")
def test_create_document_chunks_short_text(mock_text_splitter_constructor):
    """Test chunk creation with text shorter than chunk size (likely one chunk)."""
    extracted_pages = [("Short text.", 1)]
    metadata = {"source": "test.pdf"}
    
    mock_splitter_instance = MagicMock()
    doc_short = Document(page_content="Short text.", metadata={"source": "test.pdf", "page": 1})
    mock_splitter_instance.create_documents.return_value = [doc_short]
    mock_text_splitter_constructor.return_value = mock_splitter_instance
    
    chunks = create_document_chunks(extracted_pages, metadata)
    
    assert len(chunks) == 1
    assert chunks[0] == doc_short
    mock_splitter_instance.create_documents.assert_called_once_with(
        ["Short text."], 
        [{"source": "test.pdf", "page": 1}]
    ) 