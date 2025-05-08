import pytest
from pydantic import ValidationError

# Import the actual models used in the implementation
from ons_data.models.common import (
    DatasetType,
    GeoLevel,
    Dimension,
    DimensionOption,
    DimensionWithOptions,
    Dataset,
    BatchSizeConfig,
    ONSConfig
)

class TestCommonModels:

    def test_geo_level_model(self):
        """Test the GeoLevel model validation"""
        # Valid GeoLevel
        valid_geo_level = GeoLevel(id="ctry", name="Country", batch_size=200)
        assert valid_geo_level.id == "ctry"
        assert valid_geo_level.name == "Country"
        assert valid_geo_level.batch_size == 200

        # Test validation for required fields
        with pytest.raises(ValidationError):
            GeoLevel(id="ctry", batch_size=200)  # Missing name

        with pytest.raises(ValidationError):
            GeoLevel(name="Country", batch_size=200)  # Missing id

    def test_dimension_option_model(self):
        """Test the DimensionOption model validation"""
        # Valid DimensionOption
        valid_option = DimensionOption(id="1", label="No religion")
        assert valid_option.id == "1"
        assert valid_option.label == "No religion"

        # Test validation for required fields
        with pytest.raises(ValidationError):
            DimensionOption(id="1")  # Missing label

        with pytest.raises(ValidationError):
            DimensionOption(label="No religion")  # Missing id

    def test_dimension_model(self):
        """Test the Dimension model validation"""
        # Valid Dimension
        valid_dimension = Dimension(id="religion_tb", label="Religion")

        assert valid_dimension.id == "religion_tb"
        assert valid_dimension.label == "Religion"

        # Test validation for required fields
        with pytest.raises(ValidationError):
            Dimension(label="Religion")  # Missing id

        with pytest.raises(ValidationError):
            Dimension(id="religion_tb")  # Missing label

    def test_dimension_with_options_model(self):
        """Test DimensionWithOptions model"""
        # Create options
        option1 = DimensionOption(id="1", label="No religion")
        option2 = DimensionOption(id="2", label="Christian")

        # Valid DimensionWithOptions
        dimension = DimensionWithOptions(
            id="religion_tb",
            label="Religion",
            options=[option1, option2]
        )

        assert dimension.id == "religion_tb"
        assert dimension.label == "Religion"
        assert len(dimension.options) == 2
        assert dimension.options[0].id == "1"
        assert dimension.options[1].label == "Christian"

        # Empty options list should be valid
        dimension_empty = DimensionWithOptions(id="test_dim", label="Test Dimension")
        assert dimension_empty.options == []

    def test_dataset_model(self):
        """Test Dataset model and dataset_type property"""
        # Create TS dataset
        ts_dataset = Dataset(
            id="TS030",
            title="Religion by geographic area",
            description="Census 2021 time series data on religion"
        )

        assert ts_dataset.id == "TS030"
        assert ts_dataset.title == "Religion by geographic area"
        assert ts_dataset.description == "Census 2021 time series data on religion"
        assert ts_dataset.dataset_type == DatasetType.TOPIC_SUMMARY

        # Create RM dataset
        rm_dataset = Dataset(
            id="RM097",
            title="Occupancy rating by ethnic group",
            description="Census 2021 data on occupancy rating"
        )

        assert rm_dataset.dataset_type == DatasetType.REGULAR_MATRIX

        # Test dataset with other ID prefix
        other_dataset = Dataset(id="XX123", title="Other dataset")
        assert other_dataset.dataset_type == DatasetType.OTHER

    def test_batch_size_config(self):
        """Test BatchSizeConfig model"""
        # Default configuration
        config = BatchSizeConfig()
        assert config.ctry == 200
        assert config.rgn == 200
        assert config.la == 200
        assert config.msoa == 200
        assert config.lsoa == 200
        assert config.oa == 200

        # Custom configuration
        custom_config = BatchSizeConfig(
            ctry=100,
            rgn=100,
            la=50,
            msoa=25,
            lsoa=20,
            oa=10
        )

        assert custom_config.ctry == 100
        assert custom_config.oa == 10

        # Test get_for_level method
        assert config.get_for_level("ctry") == 200
        assert config.get_for_level("la") == 200
        assert config.get_for_level("unknown_level") == 50  # Default value

    def test_ons_config(self):
        """Test ONSConfig model"""
        # Default configuration
        config = ONSConfig(dataset_id="TS030")
        assert config.dataset_id == "TS030"
        assert config.geo_levels == []
        assert config.population_type == "UR"
        assert config.output_dir == "data"
        assert isinstance(config.batch_sizes, BatchSizeConfig)

        # Custom configuration
        custom_config = ONSConfig(
            dataset_id="RM097",
            geo_levels=["ctry", "rgn"],
            population_type="UR",
            output_dir="custom_data",
            batch_sizes=BatchSizeConfig(ctry=50, rgn=50)
        )

        assert custom_config.dataset_id == "RM097"
        assert custom_config.geo_levels == ["ctry", "rgn"]
        assert custom_config.output_dir == "custom_data"
        assert custom_config.batch_sizes.ctry == 50
