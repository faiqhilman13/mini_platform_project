import pytest
from unittest.mock import patch, MagicMock, mock_open

from workflows.pipelines.summarizer import (
    extract_text_from_pdf,
    summarize_text,
    run_pdf_summary_pipeline,
    DEFAULT_SENTENCE_COUNT,
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

# Tests for extract_text_from_pdf
def test_extract_text_from_pdf_success(mock_pdf_reader):
    """Test successful text extraction from a PDF."""
    with patch("workflows.pipelines.summarizer.PdfReader", return_value=mock_pdf_reader) as mock_pdf_constructor:
        extracted_text = extract_text_from_pdf("dummy/path.pdf")
    mock_pdf_constructor.assert_called_once_with("dummy/path.pdf")
    assert "This is the first page." in extracted_text
    assert "This is the second page." in extracted_text
    assert extracted_text.count("\n") >= 2 # Each page text adds a newline

def test_extract_text_from_pdf_file_not_found():
    """Test extraction when PDF file is not found."""
    with patch("workflows.pipelines.summarizer.PdfReader", side_effect=FileNotFoundError("File not found")) as mock_pdf_constructor:
        extracted_text = extract_text_from_pdf("non_existent.pdf")
    mock_pdf_constructor.assert_called_once_with("non_existent.pdf")
    assert extracted_text == ""

def test_extract_text_from_pdf_general_exception(mock_pdf_reader):
    """Test extraction when a general error occurs during PDF parsing."""
    mock_pdf_reader.pages[0].extract_text = MagicMock(side_effect=Exception("Corrupt page"))
    with patch("workflows.pipelines.summarizer.PdfReader", return_value=mock_pdf_reader) as mock_pdf_constructor:
        extracted_text = extract_text_from_pdf("corrupt.pdf")
    mock_pdf_constructor.assert_called_once_with("corrupt.pdf")
    assert extracted_text == "" # Should return empty string on error

# Tests for summarize_text
def test_summarize_text_success(mock_sumy_components):
    """Test successful text summarization."""
    input_text = "This is a long document with many sentences. It needs to be summarized."
    expected_summary = "This is the first summary sentence. This is the second summary sentence."
    
    summary = summarize_text(input_text, num_sentences=2, language="english")

    mock_sumy_components["Tokenizer"].assert_called_once_with("english")
    mock_sumy_components["Parser"].from_string.assert_called_once_with(input_text, mock_sumy_components["Tokenizer"].return_value)
    mock_sumy_components["Stemmer"].assert_called_once_with("english")
    mock_sumy_components["LsaSummarizer"].assert_called_once_with(mock_sumy_components["Stemmer"].return_value)
    mock_sumy_components["GetStopWords"].assert_called_once_with("english")
    
    mock_sumy_components["lsa_instance"].assert_called_once_with(mock_sumy_components["parser_instance"].document, 2)
    assert summary == expected_summary

def test_summarize_text_empty_input():
    """Test summarization with empty input text."""
    summary = summarize_text("   ") # Whitespace only
    assert summary == ""

def test_summarize_text_summarization_error(mock_sumy_components):
    """Test summarization when sumy library raises an exception."""
    mock_sumy_components["lsa_instance"].side_effect = Exception("Sumy internal error")
    input_text = "Some valid text content."
    summary = summarize_text(input_text)
    assert summary == ""

# Tests for run_pdf_summary_pipeline (integration of the two)
def test_run_pdf_summary_pipeline_success():
    """Test the full PDF summary pipeline successfully."""
    mock_extracted_text = "This is extracted text from a PDF. It has multiple sentences for summarization."
    mock_summary = "This is a summary."

    with patch("workflows.pipelines.summarizer.extract_text_from_pdf", return_value=mock_extracted_text) as mock_extract, \
         patch("workflows.pipelines.summarizer.summarize_text", return_value=mock_summary) as mock_summarize:
        
        result = run_pdf_summary_pipeline("test.pdf", num_summary_sentences=1)

    mock_extract.assert_called_once_with("test.pdf")
    mock_summarize.assert_called_once_with(mock_extracted_text, 1)
    assert result == {"summary": mock_summary, "status": "success"}

def test_run_pdf_summary_pipeline_extraction_fails():
    """Test pipeline when text extraction fails."""
    with patch("workflows.pipelines.summarizer.extract_text_from_pdf", return_value="") as mock_extract, \
         patch("workflows.pipelines.summarizer.summarize_text") as mock_summarize: # Should not be called
        
        result = run_pdf_summary_pipeline("bad.pdf")

    mock_extract.assert_called_once_with("bad.pdf")
    mock_summarize.assert_not_called()
    assert "Failed to extract text" in result["status"]
    assert result["summary"] == ""

def test_run_pdf_summary_pipeline_summarization_fails():
    """Test pipeline when summarization fails after successful extraction."""
    mock_extracted_text = "Some extracted text."
    with patch("workflows.pipelines.summarizer.extract_text_from_pdf", return_value=mock_extracted_text) as mock_extract, \
         patch("workflows.pipelines.summarizer.summarize_text", return_value="") as mock_summarize:
        
        result = run_pdf_summary_pipeline("good_extraction_bad_summary.pdf")

    mock_extract.assert_called_once_with("good_extraction_bad_summary.pdf")
    mock_summarize.assert_called_once_with(mock_extracted_text, DEFAULT_SENTENCE_COUNT)
    assert "Failed to generate summary" in result["status"]
    assert result["summary"] == ""

# Edge case: PDF exists but text extraction yields empty string (e.g., image-based PDF)
def test_run_pdf_summary_pipeline_empty_extracted_text():
    """Test pipeline when PDF is valid but yields no text (e.g., scanned image PDF)."""
    # extract_text_from_pdf would return "" in this case, not raise FileNotFoundError
    with patch("workflows.pipelines.summarizer.extract_text_from_pdf", return_value="") as mock_extract, \
         patch("workflows.pipelines.summarizer.summarize_text") as mock_summarize:
        
        result = run_pdf_summary_pipeline("image_based.pdf")

    mock_extract.assert_called_once_with("image_based.pdf")
    mock_summarize.assert_not_called() # Because extracted_text is empty
    assert "Failed to extract text" in result["status"]
    assert result["summary"] == ""
