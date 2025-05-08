import pytest
from pydantic import ValidationError

# Import the models under test
from ons_client.models.rm_models import RMResponse, RMObservation, RMDimension
from ons_client.models.common import Dimension, DimensionOption

class TestRMModels:

    def test_rm_observation_model(self):
        """Test the RMObservation model validation"""
        # Create dimension objects
        dim1 = RMDimension(dimension_id="geography", option="England", option_id="E92000001")
        dim2 = RMDimension(dimension_id="age", option="0-15", option_id="1")
        dim3 = RMDimension(dimension_id="sex", option="Male", option_id="1")

        # Valid RMObservation
        valid_obs = RMObservation(
            observation="5123456",
            dimensions=[dim1, dim2, dim3]
        )

        assert len(valid_obs.dimensions) == 3
        assert valid_obs.observation == "5123456"
        assert valid_obs.dimensions[0].dimension_id == "geography"
        assert valid_obs.dimensions[0].option == "England"

        # Test validation for required fields
        with pytest.raises(ValidationError):
            RMObservation(dimensions=[dim1, dim2, dim3])  # Missing observation

        with pytest.raises(ValidationError):
            RMObservation(observation="5123456")  # Missing dimensions

    def test_rm_response_model(self):
        """Test the RMResponse model validation"""
        # Create dimensions with proper labels
        geo_option = DimensionOption(id="E92000001", label="England")
        age_option1 = DimensionOption(id="1", label="0-15")
        age_option2 = DimensionOption(id="2", label="16-64")
        sex_option1 = DimensionOption(id="1", label="Male")
        sex_option2 = DimensionOption(id="2", label="Female")

        geo_dim = Dimension(id="geography", label="Geography", options=[geo_option])
        age_dim = Dimension(id="age", label="Age", options=[age_option1, age_option2])
        sex_dim = Dimension(id="sex", label="Sex", options=[sex_option1, sex_option2])

        # Create sample observations for the response
        obs_data = {"observation1": "value1"}

        # Valid RMResponse
        valid_response = RMResponse(
            observations=[obs_data]
        )

        assert len(valid_response.observations) == 1
        assert valid_response.observations[0] == obs_data

    def test_json_serialization(self):
        """Test JSON serialization/deserialization of RMDimension"""
        # Create dimension for testing
        dim = RMDimension(dimension_id="geography", option="England", option_id="E92000001")

        # Test JSON serialization
        dim_json = dim.model_dump_json()

        # Test JSON deserialization
        dim_from_json = RMDimension.model_validate_json(dim_json)

        # Verify equality
        assert dim_from_json.dimension_id == dim.dimension_id
        assert dim_from_json.option == dim.option
        assert dim_from_json.option_id == dim.option_id

    def test_rm_observation_serialization(self):
        """Test RMObservation serialization"""
        # Create dimension objects
        dim1 = RMDimension(dimension_id="geography", option="England", option_id="E92000001")
        dim2 = RMDimension(dimension_id="age", option="0-15", option_id="1")

        # Create observation
        obs = RMObservation(
            observation="5123456",
            dimensions=[dim1, dim2]
        )

        # Test serialization
        obs_json = obs.model_dump_json()

        # Test deserialization
        obs_from_json = RMObservation.model_validate_json(obs_json)

        # Verify equality
        assert obs_from_json.observation == obs.observation
        assert len(obs_from_json.dimensions) == len(obs.dimensions)
        assert obs_from_json.dimensions[0].dimension_id == obs.dimensions[0].dimension_id
