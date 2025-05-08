import pytest
from pydantic import ValidationError

# Import the models under test
from ons_client.models.ts_models import TSResponse, TSObservation, TSFlattenedRow
from ons_client.models.common import Dimension, DimensionOption

class TestTSModels:

    def test_ts_observation_model(self):
        """Test the TSObservation model validation"""
        # Valid TSObservation with string value as per implementation
        valid_obs = TSObservation(
            value="20715664",
            dimension_values={"geography": "E92000001", "religion": "1"}
        )

        assert valid_obs.value == "20715664"
        assert valid_obs.dimension_values["geography"] == "E92000001"
        assert valid_obs.dimension_values["religion"] == "1"

        # Test validation for required fields
        with pytest.raises(ValidationError):
            TSObservation(dimension_values={"geography": "E92000001"})  # Missing value

    def test_ts_response_model(self):
        """Test the TSResponse model validation"""
        # Create sample dimensions data
        dimensions_data = [
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
        ]

        # Sample observations
        observations_data = ["20715664", "26167900"]

        # Valid TSResponse
        valid_response = TSResponse(
            dimensions=dimensions_data,
            observations=observations_data
        )

        assert len(valid_response.dimensions) == 2
        assert valid_response.dimensions[0]["name"] == "geography"
        assert valid_response.dimensions[1]["name"] == "religion_tb"
        assert len(valid_response.observations) == 2
        assert valid_response.observations[0] == "20715664"
        assert valid_response.observations[1] == "26167900"

    def test_ts_flattened_row_model(self):
        """Test the TSFlattenedRow model"""
        # Create a flattened row
        row = TSFlattenedRow(
            observation="20715664",
            geography="E92000001",
            religion="No religion"
        )

        assert row.observation == "20715664"
        assert row.geography == "E92000001"
        assert row.religion == "No religion"

        # Test extra fields support
        row_dict = row.model_dump()
        assert "observation" in row_dict
        assert "geography" in row_dict
        assert "religion" in row_dict

    def test_json_serialization(self):
        """Test JSON serialization/deserialization of models"""
        # Create object for testing
        obs = TSObservation(
            value="20715664",
            dimension_values={"geography": "E92000001", "religion": "1"}
        )

        # Test JSON serialization
        obs_json = obs.model_dump_json()

        # Test JSON deserialization
        obs_from_json = TSObservation.model_validate_json(obs_json)

        # Verify equality
        assert obs_from_json.value == obs.value
        assert obs_from_json.dimension_values == obs.dimension_values
