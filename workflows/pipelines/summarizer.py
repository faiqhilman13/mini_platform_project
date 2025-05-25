"""
PDF Summarization Pipeline Logic
"""

import logging
import os
from typing import List, Tuple, Dict, Any
from pypdf import PdfReader # Using pypdf, the successor to PyPDF2
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer # Using LSA as a starting point
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from nltk.tokenize import sent_tokenize
import nltk
from prefect import task, flow # Added Prefect imports

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logging.info("NLTK 'punkt' resource not found. Downloading...")
    nltk.download('punkt', quiet=True)
except LookupError:
    logging.info("NLTK 'punkt' resource not found. Downloading...")
    nltk.download('punkt', quiet=True)

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LANGUAGE = "english"
SENTENCES_COUNT = 5  # Number of sentences in the summary

@task # Decorated with Prefect task
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text content from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text content.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        Exception: If there is an error reading the PDF.
    """
    logger.info(f"Extracting text from PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"  # Add newline between pages
        logger.info(f"Successfully extracted text from {pdf_path}.")
        return text.strip() # Remove leading/trailing whitespace
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        raise Exception(f"Failed to extract text from PDF: {e}")

@task # Decorated with Prefect task
def summarize_text(text: str, language: str = LANGUAGE, sentences_count: int = SENTENCES_COUNT) -> List[str]:
    """
    Summarizes the given text using the LSA algorithm.

    Args:
        text (str): The text to summarize.
        language (str): The language of the text (default: "english").
        sentences_count (int): The desired number of sentences in the summary (default: 5).

    Returns:
        List[str]: A list of sentences forming the summary.

    Raises:
        ValueError: If the input text is empty or too short.
        Exception: For errors during summarization.
    """
    logger.info(f"Summarizing text ({language}, {sentences_count} sentences).")
    # Ensure NLTK sentence tokenizer is available
    try:
        sent_tokenize(text) # Use it to check for errors / availability
    except (LookupError, FileNotFoundError):
        logger.exception(f"NLTK 'punkt' data not found or failed to load for language '{language}'.")
        return [] # Return empty list if tokenizer isn't available

    # Check if text is too short or just whitespace after initial check
    if not text.strip() or len(sent_tokenize(text)) < sentences_count:
        logger.warning("Input text is too short for summarization or empty.")
        return [] # Return empty list if too short or empty

    try:
        parser = PlaintextParser.from_string(text, Tokenizer(language))
        stemmer = Stemmer(language)
        summarizer = LsaSummarizer(stemmer)
        summarizer.stop_words = get_stop_words(language)

        summary_sentences = []
        for sentence in summarizer(parser.document, sentences_count):
            summary_sentences.append(str(sentence))

        logger.info("Successfully generated summary.")
        return summary_sentences
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        # Return empty list on error instead of raising Exception
        return []

@flow(name="PDF Summarization Flow") # New Prefect flow
def run_pdf_summary_pipeline(pdf_path: str) -> Dict[str, Any]:
    """
    Runs the PDF summarization pipeline as a Prefect flow.

    Args:
        pdf_path (str): Path to the input PDF file.

    Returns:
        Dict[str, Any]: A dictionary containing the status and summary.
                      {'status': 'success', 'summary': [sentence1, ...]} or
                      {'status': 'error', 'message': 'error details'}
    """
    logger.info(f"Starting PDF summary flow for: {pdf_path}")
    try:
        # Call tasks using .submit() for potential parallelism or use default sequential execution
        extracted_text = extract_text_from_pdf(pdf_path) 
        if not extracted_text:
            logger.warning("Extracted text is empty. Cannot summarize.")
            # Returning error status within the flow's result
            return {"status": "error", "message": "Extracted text is empty."}

        # Call summarize_text without sentences_count, using its default
        summary_list = summarize_text(extracted_text) 
        
        # Check if summarization failed (returned empty list due to error)
        if not summary_list and extracted_text: # Check extracted_text to differentiate from genuinely empty PDFs
             logger.error(f"Summarization task failed for {pdf_path}")
             return {"status": "error", "message": "Summarization task failed."}
             
        logger.info(f"Flow completed successfully for: {pdf_path}")
        # Return list directly in summary field
        return {"status": "success", "summary": summary_list} 

    except FileNotFoundError as e:
        logger.error(f"Flow failed due to missing file: {pdf_path} - {e}")
        return {"status": "error", "message": f"File not found: {pdf_path}"}
    except Exception as e:
        # Log exceptions happening during task execution or flow logic
        logger.exception(f"PDF summary flow failed for {pdf_path}: {e}")
        # Prefect automatically captures task failures, but we can return a structured error
        return {"status": "error", "message": f"Flow failed: {e}"}

if __name__ == '__main__':
    # Example usage for local testing
    # Create a dummy pdf if it doesn't exist
    if not os.path.exists("example.pdf"):
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        c = canvas.Canvas("example.pdf", pagesize=letter)
        textobject = c.beginText(50, 750)
        textobject.textLines('''
        This is the first sentence of the example PDF document.
        Prefect is a workflow orchestration tool designed for modern data stacks.
        It helps build, run, and monitor data pipelines.
        This document exists solely for testing the summarization pipeline.
        LSA summarization will attempt to find the most salient sentences.
        We expect the summary to contain key concepts about Prefect.
        The final sentence provides closure.
        ''')
        c.drawText(textobject)
        c.save()
        print("Created dummy example.pdf")

    pdf_file = "example.pdf"
    print(f"Running pipeline for {pdf_file}...")
    # result = run_pdf_summary_pipeline_celery(pdf_file) # Removed test for old function
    result = run_pdf_summary_pipeline(pdf_file) # Test the new Prefect flow
    print("\nResult:")
    print(result)

    if result["status"] == "success":
        print("\nSummary:")
        for sentence in result["summary"]:
            print(f"- {sentence}")

    # Test case for non-existent file
    print("\nRunning pipeline for non_existent.pdf...")
    result_non_existent = run_pdf_summary_pipeline("non_existent.pdf")
    print("\nResult (Non-existent file):")
    print(result_non_existent)

    # Test case for empty file (difficult to create directly, simulate with empty text)
    print("\nRunning pipeline for empty text...")
    try:
        empty_text_summary = summarize_text("")
        print("\nResult (Empty text):")
        print({"status": "success", "summary": empty_text_summary})
    except Exception as e:
        print("\nResult (Empty text):")
        print({"status": "error", "message": str(e)})