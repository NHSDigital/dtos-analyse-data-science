import pytest
import time
from unittest.mock import patch, Mock, call
import requests
import tempfile
import os
import json

# Import module under test
from ons_client.api.client import ONSApiClient, with_retry

class TestONSApiClient:

    def test_init(self):
        """Test client initialization"""
        client = ONSApiClient()
        assert client.base_url == "https://api.beta.ons.gov.uk/v1"

        # Test with custom base URL
        custom_client = ONSApiClient(base_url="https://custom-api.example.com")
        assert custom_client.base_url == "https://custom-api.example.com"

    @patch('requests.get')
    def test_get_datasets(self, mock_get):
        """Test getting datasets list"""
        # Configure the mock to return successful responses for all API calls
        mock_response = Mock()
        mock_response.status_code = 200

        # For datasets pagination
        mock_response.json.return_value = {
            "total_count": 2,
            "items": [
                {"id": "TS030", "title": "Religion", "description": "Religion data"},
                {"id": "RM097", "title": "Occupancy", "description": "Occupancy data"}
            ]
        }

        # Make all requests return the same mock response
        mock_get.return_value = mock_response

        client = ONSApiClient()
        datasets = client.get_datasets()

        # Don't check specific API call count or URLs since the implementation
        # now makes multiple API calls for discovery and pagination
        assert mock_get.call_count >= 1

        # Check datasets extracted correctly - we should have at least
        # the two datasets from our mock response
        assert len(datasets) >= 2
        # Find our test datasets in the results
        ts_dataset = next((d for d in datasets if d.id == "TS030"), None)
        rm_dataset = next((d for d in datasets if d.id == "RM097"), None)

        assert ts_dataset is not None
        assert ts_dataset.title == "Religion"
        assert rm_dataset is not None
        assert rm_dataset.description == "Occupancy data"

    @patch('requests.get')
    def test_get_dimensions(self, mock_get):
        """Test getting dimensions"""
        # First response for dimensions list
        dimensions_response = Mock()
        dimensions_response.status_code = 200
        dimensions_response.json.return_value = {
            "items": [
                {"id": "religion", "label": "Religion"},
                {"id": "geography", "label": "Geography"}
            ]
        }

        # Second response for religion options
        religion_options_response = Mock()
        religion_options_response.status_code = 200
        religion_options_response.json.return_value = {
            "items": [
                {"id": "1", "label": "No religion"},
                {"id": "2", "label": "Christian"}
            ]
        }

        # Third response for geography options
        geography_options_response = Mock()
        geography_options_response.status_code = 200
        geography_options_response.json.return_value = {
            "items": [
                {"id": "E92000001", "label": "England"}
            ]
        }

        # Set up the mock to return different responses
        mock_get.side_effect = [
            dimensions_response,
            religion_options_response,
            geography_options_response
        ]

        client = ONSApiClient()
        dimensions = client.get_dimensions(population_type="UR")

        # Check API called correctly
        assert mock_get.call_count == 3
        mock_get.assert_has_calls([
            call("https://api.beta.ons.gov.uk/v1/population-types/UR/dimensions"),
            call("https://api.beta.ons.gov.uk/v1/population-types/UR/dimensions/religion/options"),
            call("https://api.beta.ons.gov.uk/v1/population-types/UR/dimensions/geography/options")
        ])

        # Check dimensions extracted correctly
        assert len(dimensions) == 2
        assert dimensions[0].id == "religion"
        assert dimensions[0].label == "Religion"
        assert len(dimensions[0].options) == 2
        assert dimensions[0].options[0].id == "1"
        assert dimensions[0].options[0].label == "No religion"

        assert dimensions[1].id == "geography"
        assert dimensions[1].options[0].id == "E92000001"

    @patch('requests.get')
    def test_get_areas_for_level(self, mock_get):
        """Test getting areas for a specific geographic level"""
        # Configure mock to return areas
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

        # Create a temporary cache directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Path for cache
            cache_dir = os.path.join(temp_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)

            # Patch os.path.exists to return False (no cache)
            with patch('os.path.exists', return_value=False):
                # Patch open and json.dump to avoid file operations
                with patch('builtins.open', mock_open := Mock()):
                    with patch('json.dump') as mock_json_dump:
                        # Patch cache dir location
                        with patch('os.makedirs') as mock_makedirs:
                            client = ONSApiClient()
                            areas = client.get_areas_for_level("ctry", "UR")

                            # Verify API was called
                            mock_get.assert_called_with("https://api.beta.ons.gov.uk/v1/population-types/UR/area-types/ctry/areas")

                            # Check areas extracted correctly
                            assert len(areas) == 2
                            assert areas[0]["id"] == "E92000001"
                            assert areas[0]["label"] == "England"
                            assert areas[1]["id"] == "W92000004"

    @patch('requests.get')
    @patch('time.sleep')
    def test_pagination(self, mock_sleep, mock_get):
        """Test handling of paginated responses"""
        # First page with next link
        first_response = Mock()
        first_response.status_code = 200
        first_response.json.return_value = {
            "items": [
                {"id": "1", "label": "First page item"}
            ],
            "links": {
                "next": {
                    "href": "https://api.beta.ons.gov.uk/v1/test-endpoint?page=2"
                }
            }
        }

        # Second page with no next link
        second_response = Mock()
        second_response.status_code = 200
        second_response.json.return_value = {
            "items": [
                {"id": "2", "label": "Second page item"}
            ],
            "links": {}  # No next page
        }

        # Set up the mock to return different responses
        mock_get.side_effect = [first_response, second_response]

        # Create a temporary cache directory and patch functions
        with tempfile.TemporaryDirectory() as temp_dir:
            # Patch os.path.exists and other file operations
            with patch('os.path.exists', return_value=False):
                with patch('builtins.open', Mock()):
                    with patch('json.dump'):
                        with patch('os.makedirs'):
                            client = ONSApiClient()
                            # Test pagination by calling get_areas_for_level
                            # which uses pagination internally
                            url = f"{client.base_url}/population-types/UR/area-types/la/areas"
                            areas = client.get_areas_for_level("la", "UR")

                            # Check that requests.get was called twice (for pagination)
                            assert mock_get.call_count == 2
                            # Verify collected items
                            assert len(areas) == 2

    def test_retry_decorator(self):
        """Test the retry decorator directly"""
        # Create mock response with status code
        mock_response = Mock()
        mock_response.status_code = 429

        # Create HTTPError with the response
        http_error = requests.exceptions.HTTPError("Rate limited")
        http_error.response = mock_response

        # Create a mock function that first raises an error, then succeeds
        mock_func = Mock()
        mock_func.side_effect = [http_error, {"success": True}]

        # Apply the decorator to our test function
        @with_retry(max_retries=2, initial_delay=0.1)
        def test_func():
            result = mock_func()
            if isinstance(result, requests.exceptions.HTTPError):
                raise result
            return result

        # Patch time.sleep to avoid actual delays
        with patch('time.sleep'):
            with patch('random.uniform', return_value=1.0):
                # Call the decorated function
                result = test_func()

                # Verify it was called twice and succeeded
                assert mock_func.call_count == 2
                assert result == {"success": True}
