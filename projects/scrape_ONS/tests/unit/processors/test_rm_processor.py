import pytest
import pandas as pd
import os
import tempfile
import json
from unittest.mock import patch, Mock, MagicMock, mock_open

# Import the RM processor
from ons_data.processors.rm_processor import RMProcessor
from ons_data.models.rm_models import RMResponse, RMObservation, RMDimension
from ons_data.models.common import Dimension, DimensionOption

class TestRMProcessor:

    @pytest.fixture
    def sample_rm_api_response(self):
        """Create sample RM API response data for testing"""
        return {
            "dimensions": [
                {
                    "name": "geography",
                    "options": [
                        {"id": "E92000001", "name": "England"}
                    ]
                },
                {
                    "name": "age",
                    "options": [
                        {"id": "1", "name": "0-15"},
                        {"id": "2", "name": "16-64"}
                    ]
                },
                {
                    "name": "sex",
                    "options": [
                        {"id": "1", "name": "Male"},
                        {"id": "2", "name": "Female"}
                    ]
                }
            ],
            "observations": {
                "E92000001,1,1": "5123456",
                "E92000001,1,2": "4987654",
                "E92000001,2,1": "18123456",
                "E92000001,2,2": "19234567"
            }
        }

    def test_init(self):
        """Test processor initialization"""
        processor = RMProcessor()
        assert isinstance(processor, RMProcessor)

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_process_response(self, mock_json_dump, mock_file_open, sample_rm_api_response):
        """Test processing API response to a file"""
        processor = RMProcessor()
        output_file = "test_output.csv"

        # Mock CSV file writing
        with patch('csv.writer') as mock_writer:
            mock_csv = Mock()
            mock_writer.return_value = mock_csv

            # Process the data
            result = processor.process_response(sample_rm_api_response, output_file)

            # Check that debug file was written
            mock_file_open.assert_any_call(f"{output_file}.debug.json", 'w')
            mock_json_dump.assert_called_once()

            # Check that CSV file was created with values
            mock_file_open.assert_any_call(output_file, 'w', newline='')
            mock_writer.assert_called_once()

            # Result should be the output file path
            assert result == output_file

    def test_flatten_data(self):
        """Test the flatten_data method"""
        processor = RMProcessor()

        # Create temp files for testing
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_input:
            temp_input_path = temp_input.name
            # Write a CSV file with 'value' column
            temp_input.write(b"value\n5123456\n4987654\n18123456\n19234567\n")

        # Create debug JSON file with metadata
        debug_file_path = f"{temp_input_path}.debug.json"
        with open(debug_file_path, 'w') as debug_file:
            # Sample metadata
            sample_data = {
                "dimensions": [
                    {
                        "dimension_name": "geography",
                        "options": [
                            {"id": "E92000001", "label": "England"}
                        ]
                    },
                    {
                        "dimension_name": "age",
                        "options": [
                            {"id": "1", "label": "0-15"},
                            {"id": "2", "label": "16-64"}
                        ]
                    },
                    {
                        "dimension_name": "sex",
                        "options": [
                            {"id": "1", "label": "Male"},
                            {"id": "2", "label": "Female"}
                        ]
                    }
                ],
                "observations": {
                    "E92000001,1,1": "5123456",
                    "E92000001,1,2": "4987654",
                    "E92000001,2,1": "18123456",
                    "E92000001,2,2": "19234567"
                }
            }
            json.dump(sample_data, debug_file)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_output:
            temp_output_path = temp_output.name

        try:
            # Test the function
            result = processor.flatten_data(temp_input_path, temp_output_path)
            assert isinstance(result, str)
            assert result == temp_output_path

            # Verify output file contains data
            with open(temp_output_path, "r") as f:
                output_content = f.read()
                assert len(output_content) > 0

                # Check for expected header content
                assert "geography,geography_code,age,age_code,sex,sex_code,observation" in output_content

        finally:
            # Clean up temp files
            os.unlink(temp_input_path)
            os.unlink(debug_file_path)
            os.unlink(temp_output_path)

    def test_create_dimension_combinations(self, sample_rm_api_response):
        """Test creating dimension combinations from metadata"""
        processor = RMProcessor()

        # Test values
        values = ["5123456", "4987654", "18123456", "19234567"]

        # Mock the dimension_data to use dimension_name
        modified_response = sample_rm_api_response.copy()
        modified_response["dimensions"] = [
            {
                "dimension_name": "geography",
                "options": [
                    {"id": "E92000001", "label": "England"}
                ]
            },
            {
                "dimension_name": "age",
                "options": [
                    {"id": "1", "label": "0-15"},
                    {"id": "2", "label": "16-64"}
                ]
            },
            {
                "dimension_name": "sex",
                "options": [
                    {"id": "1", "label": "Male"},
                    {"id": "2", "label": "Female"}
                ]
            }
        ]

        # Call the method
        flat_rows, fieldnames = processor._create_dimension_combinations(modified_response, values)

        # Check results
        assert len(flat_rows) == 4
        assert len(fieldnames) == 7

        # Check fieldnames
        assert "geography" in fieldnames
        assert "geography_code" in fieldnames
        assert "age" in fieldnames
        assert "age_code" in fieldnames
        assert "sex" in fieldnames
        assert "sex_code" in fieldnames
        assert "observation" in fieldnames

        # Check row contents
        assert flat_rows[0]["observation"] == "5123456"
        assert flat_rows[3]["observation"] == "19234567"

    @patch('os.path.exists', return_value=True)
    @patch('os.path.getsize', return_value=100)  # Non-empty file
    def test_validate_file_exists(self, mock_getsize, mock_exists):
        """Test file validation"""
        processor = RMProcessor()
        assert processor.validate_file_exists("test_file.csv") == True

        # Test empty file
        mock_getsize.return_value = 0
        assert processor.validate_file_exists("empty_file.csv") == False

        # Test non-existent file
        mock_exists.return_value = False
        assert processor.validate_file_exists("missing_file.csv") == False
