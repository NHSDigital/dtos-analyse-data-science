import pytest
from unittest.mock import patch, Mock, call
import requests

# Import the RM client
from ons_client.api.rm_client import RMApiClient

class TestRMApiClient:

    def test_init(self):
        """Test RM client initialization"""
        client = RMApiClient()
        # Base URL should match ONSApiClient base URL
        assert client.base_url == "https://api.beta.ons.gov.uk/v1"

    @patch('requests.get')
    def test_get_dataset_url(self, mock_get):
        """Test URL construction using batch_get_dataset_data method"""
        # Create the client
        client = RMApiClient()

        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": "data"}
        mock_get.return_value = mock_response

        # Call batch_get_dataset_data to test URL construction
        dataset_id = "RM097"
        area_codes = ["E92000001"]
        geo_level = "ctry"

        client.batch_get_dataset_data(dataset_id, area_codes, geo_level)

        # The URL should include the dataset ID, edition, version and comma-separated areas
        expected_url = "https://api.beta.ons.gov.uk/v1/datasets/RM097/editions/2021/versions/1/json?area-type=ctry,E92000001"
        mock_get.assert_called_with(expected_url)

    @patch('requests.get')
    def test_get_areas_for_level(self, mock_get):
        """Test getting areas for a specific geographic level"""
        # Configure the mock to return sample areas
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {"id": "E92000001", "label": "England"},
                {"id": "W92000004", "label": "Wales"}
            ],
            "links": {}  # No next page
        }
        mock_get.return_value = mock_response

        # Create client and patch file operations
        with patch('os.path.exists', return_value=False):
            with patch('os.makedirs'):
                with patch('builtins.open', Mock()):
                    with patch('json.dump'):
                        client = RMApiClient()
                        areas = client.get_areas_for_level("ctry", "UR")

                        # Check API called with correct URL
                        expected_url = "https://api.beta.ons.gov.uk/v1/population-types/UR/area-types/ctry/areas"
                        mock_get.assert_called_with(expected_url)

                        # Check areas extracted correctly
                        assert len(areas) == 2
                        assert areas[0]["id"] == "E92000001"
                        assert areas[0]["label"] == "England"

    @patch('requests.get')
    def test_batch_get_dataset_data(self, mock_get):
        """Test getting dataset data in batches"""
        # Configure the mock to return a sample response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "dimensions": [
                {
                    "name": "geography",
                    "options": [{"id": "E92000001", "name": "England"}]
                },
                {
                    "name": "age",
                    "options": [{"id": "1", "name": "0-15"}]
                },
                {
                    "name": "sex",
                    "options": [{"id": "1", "name": "Male"}]
                }
            ],
            "observations": {"E92000001,1,1": "5123456"}
        }
        mock_get.return_value = mock_response

        # Create client
        client = RMApiClient()

        # Test batch processing
        dataset_id = "RM097"
        area_codes = ["E92000001", "W92000004", "S92000003", "N92000002"]
        geo_level = "ctry"
        batch_size = 2

        result = client.batch_get_dataset_data(
            dataset_id=dataset_id,
            area_codes=area_codes,
            geo_level=geo_level,
            batch_size=batch_size
        )

        # Should make 2 API calls for 2 batches
        assert mock_get.call_count == 2

        # Get the actual URLs that were called
        first_call_url = mock_get.call_args_list[0][0][0]
        second_call_url = mock_get.call_args_list[1][0][0]

        # Verify the URLs match our expected patterns
        assert "ctry,E92000001,W92000004" in first_call_url
        assert "ctry,S92000003,N92000002" in second_call_url

        # Check result structure
        assert isinstance(result, list) or isinstance(result, dict)
        # The exact return type may vary based on implementation
