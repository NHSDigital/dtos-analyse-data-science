import os
import json
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def sample_ts_response():
    """Sample Time Series API response"""
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
            "E92000001,1": 20715664,
            "E92000001,2": 26167900
        }
    }

@pytest.fixture
def sample_rm_response():
    """Sample Regular Matrix API response"""
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
            "E92000001,1,1": 5123456,
            "E92000001,1,2": 4987654,
            "E92000001,2,1": 18123456,
            "E92000001,2,2": 19234567
        }
    }

@pytest.fixture
def sample_areas_response():
    """Sample areas API response"""
    return {
        "items": [
            {"id": "E92000001", "name": "England"},
            {"id": "W92000004", "name": "Wales"}
        ]
    }

@pytest.fixture
def mock_ons_api():
    """Mock for ONS API responses"""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response
        yield mock_get
