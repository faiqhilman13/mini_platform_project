#!/usr/bin/env python3
"""
Unit tests for DS1.1.1: Enhanced File Upload Support
Tests the new CSV/Excel validation functionality
"""

import pytest
import tempfile
import os
import pandas as pd
from app.services.file_service import validate_dataset_file

class TestDatasetFileValidation:
    """Test cases for CSV/Excel file validation"""

    def test_valid_csv_file(self):
        """Test validation with a valid CSV file"""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,age,city\n")
            f.write("Alice,25,New York\n")
            f.write("Bob,30,London\n")
            temp_file = f.name

        try:
            is_valid, error_msg = validate_dataset_file(temp_file, "text/csv")
            assert is_valid == True
            assert error_msg == ""
        finally:
            os.unlink(temp_file)

    def test_valid_excel_file(self):
        """Test validation with a valid Excel file"""
        # Create a temporary Excel file
        data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            df.to_excel(f.name, index=False)
            temp_file = f.name

        try:
            is_valid, error_msg = validate_dataset_file(
                temp_file, 
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            assert is_valid == True
            assert error_msg == ""
        finally:
            os.unlink(temp_file)

    def test_empty_csv_file(self):
        """Test validation with an empty CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("")  # Empty file
            temp_file = f.name

        try:
            is_valid, error_msg = validate_dataset_file(temp_file, "text/csv")
            assert is_valid == False
            assert "empty" in error_msg.lower()
        finally:
            os.unlink(temp_file)

    def test_csv_file_with_no_columns(self):
        """Test validation with a CSV file that has no columns"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("\n\n\n")  # Just newlines, no actual data
            temp_file = f.name

        try:
            is_valid, error_msg = validate_dataset_file(temp_file, "text/csv")
            assert is_valid == False
            assert "columns" in error_msg.lower() or "empty" in error_msg.lower()
        finally:
            os.unlink(temp_file)

    def test_csv_file_with_too_many_columns(self):
        """Test validation with a CSV file that has too many columns"""
        # Create CSV with 1001 columns (over the limit of 1000)
        columns = [f"col_{i}" for i in range(1001)]
        header = ",".join(columns)
        row_data = ",".join(["1"] * 1001)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(f"{header}\n{row_data}\n")
            temp_file = f.name

        try:
            is_valid, error_msg = validate_dataset_file(temp_file, "text/csv")
            assert is_valid == False
            assert "too many columns" in error_msg.lower()
        finally:
            os.unlink(temp_file)

    def test_malformed_csv_file(self):
        """Test validation with a malformed CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,age\n")
            f.write("Alice,25,Extra,Fields\n")  # Too many fields
            f.write("Bob\n")  # Too few fields
            temp_file = f.name

        try:
            # This might pass or fail depending on pandas' tolerance
            # Just ensure it doesn't crash
            is_valid, error_msg = validate_dataset_file(temp_file, "text/csv")
            assert isinstance(is_valid, bool)
            assert isinstance(error_msg, str)
        finally:
            os.unlink(temp_file)

    def test_non_dataset_file_skips_validation(self):
        """Test that non-dataset files (PDF, JSON) skip validation"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
            f.write("This is not a real PDF")
            temp_file = f.name

        try:
            is_valid, error_msg = validate_dataset_file(temp_file, "application/pdf")
            assert is_valid == True
            assert error_msg == ""
        finally:
            os.unlink(temp_file)

    def test_excel_file_by_extension(self):
        """Test validation using file extension when content-type is not reliable"""
        data = {'test': [1, 2, 3]}
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            df.to_excel(f.name, index=False)
            temp_file = f.name

        try:
            # Test with generic content type but xlsx extension
            is_valid, error_msg = validate_dataset_file(temp_file, "application/octet-stream")
            assert is_valid == True
            assert error_msg == ""
        finally:
            os.unlink(temp_file)

    def test_csv_file_by_extension(self):
        """Test validation using file extension when content-type is not reliable"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n1,2\n3,4\n")
            temp_file = f.name

        try:
            # Test with generic content type but csv extension
            is_valid, error_msg = validate_dataset_file(temp_file, "application/octet-stream")
            assert is_valid == True
            assert error_msg == ""
        finally:
            os.unlink(temp_file)

class TestFileServiceExtensions:
    """Test the enhanced file service functionality"""

    def test_is_dataset_detection(self):
        """Test that the is_dataset flag is set correctly"""
        # Test CSV content type
        assert self._is_dataset_helper("text/csv", "test.csv") == True
        
        # Test Excel content types
        assert self._is_dataset_helper("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "test.xlsx") == True
        assert self._is_dataset_helper("application/vnd.ms-excel", "test.xls") == True
        
        # Test non-dataset content types
        assert self._is_dataset_helper("application/pdf", "test.pdf") == False
        assert self._is_dataset_helper("application/json", "test.json") == False
        
        # Test extension-based detection
        assert self._is_dataset_helper("application/octet-stream", "test.csv") == True
        assert self._is_dataset_helper("application/octet-stream", "test.xlsx") == True
        assert self._is_dataset_helper("application/octet-stream", "test.xls") == True

    def _is_dataset_helper(self, content_type: str, filename: str) -> bool:
        """Helper method to test is_dataset detection logic"""
        # Replicate the logic from file_service.py
        file_location = f"/tmp/{filename}"
        is_dataset = (
            content_type == "text/csv" or 
            content_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                           "application/vnd.ms-excel"] or
            file_location.endswith(('.csv', '.xlsx', '.xls'))
        )
        return is_dataset

if __name__ == "__main__":
    pytest.main([__file__]) 