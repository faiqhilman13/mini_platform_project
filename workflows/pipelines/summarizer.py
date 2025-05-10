"""
PDF Summarization Pipeline Logic
"""

import logging
import os
from pypdf import PdfReader # Using pypdf, the successor to PyPDF2
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer # Using LSA as a starting point
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LANGUAGE = "english"
DEFAULT_SENTENCE_COUNT = 3

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text content from a given PDF file.

    Args:
        pdf_path (str): The file path to the PDF.

    Returns:
        str: The extracted text, or an empty string if extraction fails.
    """
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        logger.info(f"Successfully extracted {len(text)} characters from '{pdf_path}'.")
    except FileNotFoundError:
        logger.error(f"Error parsing PDF: File not found at '{pdf_path}'.")
        return "" # Or raise specific error
    except Exception as e:
        logger.error(f"Error parsing PDF '{pdf_path}': {e}")
        return "" # Or raise specific error
    return text

def summarize_text(text: str, num_sentences: int = DEFAULT_SENTENCE_COUNT, language: str = LANGUAGE) -> str:
    """
    Generates an extractive summary from the given text.

    Args:
        text (str): The text to summarize.
        num_sentences (int): The desired number of sentences in the summary.
        language (str): The language of the text (for tokenizer, stemmer, stop words).

    Returns:
        str: The generated summary, or an empty string if summarization fails.
    """
    if not text.strip():
        logger.warning("Cannot summarize empty or whitespace-only text.")
        return ""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer(language))
        stemmer = Stemmer(language)
        summarizer = LsaSummarizer(stemmer)
        summarizer.stop_words = get_stop_words(language)
        
        summary_sentences = summarizer(parser.document, num_sentences)
        summary = " ".join([str(sentence) for sentence in summary_sentences])
        logger.info(f"Successfully generated summary with {len(summary_sentences)} sentences.")
        return summary
    except Exception as e:
        logger.error(f"Error during text summarization: {e}")
        return "" # Or raise specific error

def run_pdf_summary_pipeline(pdf_path: str, num_summary_sentences: int = DEFAULT_SENTENCE_COUNT) -> dict:
    """
    Orchestrates the PDF summarization pipeline: extracts text and summarizes it.

    Args:
        pdf_path (str): Path to the PDF file.
        num_summary_sentences (int): Desired number of sentences in the summary.

    Returns:
        dict: A dictionary containing the summary and a status message.
              Example: {"summary": "...", "status": "success"} or 
                       {"summary": "", "status": "error: reason..."}
    """
    logger.info(f"Starting PDF summary pipeline for '{pdf_path}'.")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    if not extracted_text:
        error_message = f"Failed to extract text from PDF '{pdf_path}'."
        logger.error(error_message)
        return {"summary": "", "status": f"error: {error_message}"}
    
    summary = summarize_text(extracted_text, num_summary_sentences)
    
    if not summary:
        error_message = f"Failed to generate summary for PDF '{pdf_path}'."
        logger.error(error_message)
        return {"summary": "", "status": f"error: {error_message}"}
        
    logger.info(f"PDF summary pipeline completed successfully for '{pdf_path}'.")
    return {"summary": summary, "status": "success"}

if __name__ == '__main__':
    # Example usage (for testing purposes)
    # Create a dummy PDF for testing if you don't have one.
    # For example, save some text to a .txt file and convert to .pdf using an online tool.
    # Or, ensure you have a PDF in your project directory for testing.
    test_pdf_path = "example.pdf" # Replace with a path to a real PDF file for testing
    
    # Create a dummy example.pdf if it doesn't exist for basic testing
    # Note: This dummy PDF will likely be empty or very simple. 
    # For real testing, use a proper PDF document.
    try:
        with open(test_pdf_path, "rb") as f:
            pass # Just check if it exists
        logger.info(f"Found test PDF: {test_pdf_path}")
    except FileNotFoundError:
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            c = canvas.Canvas(test_pdf_path, pagesize=letter)
            c.drawString(72, 720, "This is a simple test PDF for the summarizer pipeline.")
            c.drawString(72, 700, "It contains a few sentences to see if text extraction works.")
            c.drawString(72, 680, "Sumy should be able to pick out the most salient points hopefully.")
            c.save()
            logger.info(f"Created dummy test PDF: {test_pdf_path}. Please replace with a real PDF for meaningful testing.")
            # Add reportlab to requirements.txt if you keep this dummy creation code for long.
        except ImportError:
            logger.warning(f"Could not create dummy PDF {test_pdf_path} because reportlab is not installed.")
            logger.warning("Please create a PDF manually or install reportlab for this example to run.")
        except Exception as e:
            logger.error(f"Could not create dummy PDF {test_pdf_path}: {e}")

    if os.path.exists(test_pdf_path):
        result = run_pdf_summary_pipeline(test_pdf_path, num_summary_sentences=2)
        print("\n--- Pipeline Result ---")
        print(f"Status: {result['status']}")
        print(f"Summary: {result['summary']}")
    else:
        print(f"\nSkipping example run as test PDF '{test_pdf_path}' was not found and could not be created.")

    # To use NLTK data like 'punkt' for tokenization, you might need to download it once:
    # import nltk
    # try:
    #     nltk.data.find('tokenizers/punkt')
    # except nltk.downloader.DownloadError:
    #     nltk.download('punkt')
    # try:
    #     nltk.data.find('corpora/stopwords')
    # except nltk.downloader.DownloadError:
    #     nltk.download('stopwords') # if your sumy summarizer uses NLTK stopwords 