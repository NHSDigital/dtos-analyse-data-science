import pytest
import pandas as pd
import os
import tempfile
import json
from unittest.mock import patch, Mock, MagicMock, mock_open

# Import the processor under test
from ons_client.processors.ts_processor import TSProcessor
from ons_client.models.ts_models import TSResponse, TSObservation, TSFlattenedRow
from ons_client.models.common import Dimension, DimensionOption

class TestTSProcessor:

    @pytest.fixture
    def sample_ts_api_response(self):
        """Create sample TS API response data for testing"""
        return {
            "dimensions": [
                {
                    "name": "geography",
                    "options": [
                        {"id": "E92000001", "name": "England"}
                    ]
                },
                {
                    "name": "religion_tb",
                    "options": [
                        {"id": "1", "name": "No religion"},
                        {"id": "2", "name": "Christian"}
                    ]
                }
            ],
            "observations": {
                "E92000001,1": "20715664",
                "E92000001,2": "26167900"
            }
        }

    def test_init(self):
        """Test processor initialization"""
        processor = TSProcessor()
        assert isinstance(processor, TSProcessor)

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_process_response(self, mock_json_dump, mock_file_open, sample_ts_api_response):
        """Test processing API response to a file"""
        processor = TSProcessor()
        output_file = "test_output.csv"

        # Mock CSV file writing
        with patch('csv.writer') as mock_writer:
            mock_csv = Mock()
            mock_writer.return_value = mock_csv

            # Process the data
            result = processor.process_response(sample_ts_api_response, output_file)

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
        processor = TSProcessor()

        # Create temp files for testing
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_input:
            temp_input_path = temp_input.name
            # Write a CSV file with 'value' column
            temp_input.write(b"value\n20715664\n26167900\n")

        # Create debug JSON file with metadata
        debug_file_path = f"{temp_input_path}.debug.json"
        with open(debug_file_path, 'w') as debug_file:
            # Sample metadata
            sample_data = {
                "dimensions": [
                    {
                        "name": "geography",
                        "options": [
                            {"id": "E92000001", "name": "England"}
                        ]
                    },
                    {
                        "name": "religion_tb",
                        "options": [
                            {"id": "1", "name": "No religion"},
                            {"id": "2", "name": "Christian"}
                        ]
                    }
                ],
                "observations": {
                    "E92000001,1": "20715664",
                    "E92000001,2": "26167900"
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
                assert "geography" in output_content
                assert "religion_tb" in output_content

        finally:
            # Clean up temp files
            os.unlink(temp_input_path)
            os.unlink(debug_file_path)
            os.unlink(temp_output_path)

    def test_create_dimension_combinations(self, sample_ts_api_response):
        """Test creating dimension combinations from metadata"""
        processor = TSProcessor()

        # Test values
        values = ["20715664", "26167900"]

        # Call the method
        flat_rows, fieldnames = processor._create_dimension_combinations(sample_ts_api_response, values)

        # Check results
        assert len(flat_rows) == 2
        assert len(fieldnames) == 5

        # Check fieldnames
        assert "geography" in fieldnames
        assert "geography_code" in fieldnames
        assert "religion_tb" in fieldnames
        assert "religion_tb_code" in fieldnames
        assert "observation" in fieldnames

        # Check row contents
        assert flat_rows[0]["observation"] == "20715664"
        assert flat_rows[1]["observation"] == "26167900"

    @patch('os.path.exists', return_value=True)
    @patch('os.path.getsize', return_value=100)  # Non-empty file
    def test_validate_file_exists(self, mock_getsize, mock_exists):
        """Test file validation"""
        processor = TSProcessor()
        assert processor.validate_file_exists("test_file.csv") == True

        # Test empty file
        mock_getsize.return_value = 0
        assert processor.validate_file_exists("empty_file.csv") == False

        # Test non-existent file
        mock_exists.return_value = False
        assert processor.validate_file_exists("missing_file.csv") == False
