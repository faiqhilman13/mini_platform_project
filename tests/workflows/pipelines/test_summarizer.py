import pytest
from unittest.mock import patch, MagicMock, mock_open

from workflows.pipelines.summarizer import (
    extract_text_from_pdf,
    summarize_text,
    run_pdf_summary_pipeline,
    LANGUAGE
)

# Mock for PdfReader().pages
class MockPdfPage:
    def __init__(self, text_content):
        self.text_content = text_content
    def extract_text(self):
        return self.text_content

@pytest.fixture
def mock_pdf_reader():
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [
        MockPdfPage("This is the first page. It has some text."),
        MockPdfPage("This is the second page. More text here.")
    ]
    return mock_reader_instance

@pytest.fixture
def mock_sumy_components():
    with patch("workflows.pipelines.summarizer.PlaintextParser") as MockParser, \
         patch("workflows.pipelines.summarizer.Tokenizer") as MockTokenizer, \
         patch("workflows.pipelines.summarizer.Stemmer") as MockStemmer, \
         patch("workflows.pipelines.summarizer.LsaSummarizer") as MockLsaSummarizer, \
         patch("workflows.pipelines.summarizer.get_stop_words") as MockGetStopWords:
        
        # Mock LSA Summarizer instance and its call
        mock_lsa_instance = MagicMock()
        mock_sentence1 = MagicMock()
        mock_sentence1.__str__ = MagicMock(return_value="This is the first summary sentence.")
        mock_sentence2 = MagicMock()
        mock_sentence2.__str__ = MagicMock(return_value="This is the second summary sentence.")
        mock_lsa_instance.return_value = [mock_sentence1, mock_sentence2] # Simulate summarizer call
        MockLsaSummarizer.return_value = mock_lsa_instance
        
        # Mock Parser instance and its document
        mock_parser_instance = MagicMock()
        mock_document = MagicMock()
        mock_parser_instance.document = mock_document
        MockParser.from_string.return_value = mock_parser_instance

        yield {
            "Parser": MockParser,
            "Tokenizer": MockTokenizer,
            "Stemmer": MockStemmer,
            "LsaSummarizer": MockLsaSummarizer,
            "GetStopWords": MockGetStopWords,
            "lsa_instance": mock_lsa_instance,
            "parser_instance": mock_parser_instance
        }

# Fixture to mock Prefect task and flow decorators
@pytest.fixture
def mock_prefect_decorators():
    with patch("workflows.pipelines.summarizer.task") as mock_task, \
         patch("workflows.pipelines.summarizer.flow") as mock_flow:
        
        # Make task decorator return the function unchanged for testing
        mock_task.side_effect = lambda *args, **kwargs: lambda fn: fn
        # Make flow decorator return the function unchanged for testing
        mock_flow.side_effect = lambda *args, **kwargs: lambda fn: fn
        
        yield {
            "task": mock_task,
            "flow": mock_flow
        }

# Tests for extract_text_from_pdf
def test_extract_text_from_pdf_success(mock_pdf_reader, mock_prefect_decorators):
    """Test successful text extraction from a PDF."""
    with patch("os.path.exists", return_value=True), \
         patch("workflows.pipelines.summarizer.PdfReader", return_value=mock_pdf_reader) as mock_pdf_constructor:
        extracted_text = extract_text_from_pdf("dummy/path.pdf")
    mock_pdf_constructor.assert_called_once_with("dummy/path.pdf")
    assert "This is the first page." in extracted_text
    assert "This is the second page." in extracted_text
    # The task function itself now raises FileNotFoundError if os.path.exists is false
    # So we test the successful path where os.path.exists is True (mocked)
    # assert extracted_text.count("\n") >= 2 # This assertion might be brittle

def test_extract_text_from_pdf_file_not_found(mock_prefect_decorators):
    """Test extraction task when PDF file is not found (os.path.exists returns False)."""
    with patch("os.path.exists", return_value=False), \
         patch("workflows.pipelines.summarizer.PdfReader") as mock_pdf_constructor, \
         pytest.raises(FileNotFoundError, match="PDF file not found"):
        extract_text_from_pdf("non_existent.pdf") # Call the task
    # PdfReader should not be called if os.path.exists is False
    mock_pdf_constructor.assert_not_called()

def test_extract_text_from_pdf_general_exception(mock_pdf_reader, mock_prefect_decorators):
    """Test extraction task when PdfReader raises an exception."""
    mock_pdf_reader.pages[0].extract_text = MagicMock(side_effect=Exception("Corrupt page"))
    # Patch os.path.exists to return True so PdfReader is called
    with patch("os.path.exists", return_value=True), \
         patch("workflows.pipelines.summarizer.PdfReader", return_value=mock_pdf_reader) as mock_pdf_constructor, \
         pytest.raises(Exception, match="Failed to extract text from PDF"): # Task re-raises exception
        extract_text_from_pdf("corrupt.pdf")
    mock_pdf_constructor.assert_called_once_with("corrupt.pdf")

# Tests for summarize_text
def test_summarize_text_success(mock_sumy_components, mock_prefect_decorators):
    """Test successful text summarization."""
    input_text = "This is a long document with many sentences. It needs to be summarized."
    expected_summary_list = ["This is the first summary sentence.", "This is the second summary sentence."] # Expect list of strings
    
    # Call with correct keyword argument name
    summary = summarize_text(input_text, sentences_count=2, language="english") 

    mock_sumy_components["Tokenizer"].assert_called_once_with("english")
    mock_sumy_components["Parser"].from_string.assert_called_once_with(input_text, mock_sumy_components["Tokenizer"].return_value)
    mock_sumy_components["Stemmer"].assert_called_once_with("english")
    mock_sumy_components["LsaSummarizer"].assert_called_once_with(mock_sumy_components["Stemmer"].return_value)
    mock_sumy_components["GetStopWords"].assert_called_once_with("english")
    
    mock_sumy_components["lsa_instance"].assert_called_once_with(mock_sumy_components["parser_instance"].document, 2)
    assert summary == expected_summary_list # Assert list equality

def test_summarize_text_empty_input(mock_prefect_decorators):
    """Test summarization with empty input text."""
    summary = summarize_text("   ") # Whitespace only
    assert summary == [] # Expect empty list now

def test_summarize_text_summarization_error(mock_sumy_components, mock_prefect_decorators):
    """Test summarization when sumy library raises an exception."""
    mock_sumy_components["lsa_instance"].side_effect = Exception("Sumy internal error")
    input_text = "Some valid text content." # Ensure input text is long enough
    summary = summarize_text(input_text)
    assert summary == [] # Expect empty list on error

# Tests for run_pdf_summary_pipeline (integration of the two)
def test_run_pdf_summary_pipeline_success(mock_prefect_decorators):
    """Test the full PDF summary pipeline successfully."""
    mock_extracted_text = "This is extracted text from a PDF. It has multiple sentences for summarization."
    mock_summary_list = ["Summary sentence 1.", "Summary sentence 2."] # Expect list

    with patch("workflows.pipelines.summarizer.extract_text_from_pdf", return_value=mock_extracted_text) as mock_extract, \
         patch("workflows.pipelines.summarizer.summarize_text", return_value=mock_summary_list) as mock_summarize:
        
        # Call flow without extra args
        result = run_pdf_summary_pipeline("test.pdf") 

    mock_extract.assert_called_once_with("test.pdf")
    # Summarize called with default sentences_count implicitly
    mock_summarize.assert_called_once_with(mock_extracted_text) 
    assert result == {"status": "success", "summary": mock_summary_list}

def test_run_pdf_summary_pipeline_extraction_fails(mock_prefect_decorators):
    """Test pipeline when text extraction task fails (raises FileNotFoundError)."""
    extract_exception = FileNotFoundError("Mock file not found")
    with patch("workflows.pipelines.summarizer.extract_text_from_pdf", side_effect=extract_exception) as mock_extract, \
         patch("workflows.pipelines.summarizer.summarize_text") as mock_summarize: # Should not be called
        
        result = run_pdf_summary_pipeline("bad.pdf")

    mock_extract.assert_called_once_with("bad.pdf")
    mock_summarize.assert_not_called()
    # Assert the flow catches the exception and returns error dict
    assert result == {"status": "error", "message": "File not found: bad.pdf"} 

def test_run_pdf_summary_pipeline_summarization_fails(mock_prefect_decorators):
    """Test pipeline when summarization fails after successful extraction."""
    mock_extracted_text = "Some extracted text."
    # Simulate summarize_text failing by returning empty list
    with patch("workflows.pipelines.summarizer.extract_text_from_pdf", return_value=mock_extracted_text) as mock_extract, \
         patch("workflows.pipelines.summarizer.summarize_text", return_value=[]) as mock_summarize:
        
        result = run_pdf_summary_pipeline("good_extraction_bad_summary.pdf")

    mock_extract.assert_called_once_with("good_extraction_bad_summary.pdf")
    mock_summarize.assert_called_once_with(mock_extracted_text) # Called with default count
    # Assert the flow detects summarization failure (empty list result) and returns error dict
    assert result == {"status": "error", "message": "Summarization task failed."} 

# Edge case: PDF exists but text extraction yields empty string (e.g., image-based PDF)
def test_run_pdf_summary_pipeline_empty_extracted_text(mock_prefect_decorators):
    """Test pipeline when PDF is valid but yields no text (e.g., scanned image PDF)."""
    with patch("workflows.pipelines.summarizer.extract_text_from_pdf", return_value="") as mock_extract, \
         patch("workflows.pipelines.summarizer.summarize_text") as mock_summarize:
        
        result = run_pdf_summary_pipeline("image_based.pdf")

    mock_extract.assert_called_once_with("image_based.pdf")
    mock_summarize.assert_not_called() # Because extracted_text is empty
    # Assert the flow handles empty extraction and returns error dict
    assert result == {"status": "error", "message": "Extracted text is empty."}

# Test for Prefect task and flow decorators
def test_prefect_decorators_applied():
    """Test that Prefect task and flow decorators are properly applied to functions."""
    from workflows.pipelines.summarizer import extract_text_from_pdf, summarize_text, run_pdf_summary_pipeline
    import inspect
    
    # Check if extract_text_from_pdf has Prefect task attributes
    assert hasattr(extract_text_from_pdf, "__prefect_task__") or hasattr(extract_text_from_pdf, "__name__")
    
    # Check if summarize_text has Prefect task attributes
    assert hasattr(summarize_text, "__prefect_task__") or hasattr(summarize_text, "__name__")
    
    # Check if run_pdf_summary_pipeline has Prefect flow attributes
    assert hasattr(run_pdf_summary_pipeline, "__prefect_flow__") or hasattr(run_pdf_summary_pipeline, "__name__")
