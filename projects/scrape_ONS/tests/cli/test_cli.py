import pytest
import argparse
import os
import tempfile
from unittest.mock import patch, MagicMock, Mock

# Import the CLI module
from ons_client.cli import (
    setup_parser,
    validate_args,
    create_config,
    ensure_output_dir,
    download_data_for_level,
    main,
)
from ons_client.models.common import ONSConfig, BatchSizeConfig


class TestCLI:

    def test_setup_parser(self):
        """Test argument parser setup"""
        parser = setup_parser()

        # Check that it's an ArgumentParser instance
        assert isinstance(parser, argparse.ArgumentParser)

        # Parse some test arguments to check it works
        args = parser.parse_args(["--dataset", "TS030"])
        assert args.dataset == "TS030"
        assert args.geo_level is None  # Default
        assert args.debug is False  # Default

        # Test with more arguments
        args = parser.parse_args(
            [
                "--dataset",
                "RM097",
                "--geo-level",
                "ctry",
                "--output-dir",
                "/tmp/test",
                "--batch-size",
                "50",
                "--population-type",
                "UR",
                "--debug",
            ]
        )

        assert args.dataset == "RM097"
        assert args.geo_level == "ctry"
        assert args.output_dir == "/tmp/test"
        assert args.batch_size == 50
        assert args.population_type == "UR"
        assert args.debug is True

    def test_validate_args(self):
        """Test argument validation"""
        # Mock args object
        valid_args = Mock(dataset="TS030", geo_level="ctry")
        assert validate_args(valid_args) is True

        # Test with empty dataset ID
        invalid_args = Mock(dataset="", geo_level="ctry")
        with patch("logging.Logger.error"):
            assert validate_args(invalid_args) is False

        # Test with too short dataset ID
        invalid_args = Mock(dataset="TS", geo_level="ctry")
        with patch("logging.Logger.error"):
            assert validate_args(invalid_args) is False

        # Test with invalid dataset type
        invalid_args = Mock(dataset="XX123", geo_level="ctry")
        with patch("logging.Logger.error"):
            assert validate_args(invalid_args) is False

        # Test with invalid geo level
        invalid_args = Mock(dataset="TS030", geo_level="invalid")
        with patch("logging.Logger.error"):
            assert validate_args(invalid_args) is False

    def test_create_config(self):
        """Test config creation from args"""
        # Mock args object for complete config
        args = Mock(
            dataset="TS030",
            geo_level="ctry",
            output_dir="/tmp/test",
            batch_size=50,
            population_type="UR",
        )

        config = create_config(args)

        # Check config properties
        assert isinstance(config, ONSConfig)
        assert config.dataset_id == "TS030"
        assert config.geo_levels == ["ctry"]
        assert config.output_dir == "/tmp/test"
        assert config.population_type == "UR"
        assert isinstance(config.batch_sizes, BatchSizeConfig)
        assert config.batch_sizes.ctry == 50  # Overridden by args.batch_size
        assert config.batch_sizes.rgn == 50  # Overridden by args.batch_size

        # Test with no geo_level specified (should use all levels)
        args = Mock(
            dataset="TS030",
            geo_level=None,
            output_dir=None,
            batch_size=None,
            population_type="UR",
        )

        config = create_config(args)
        assert config.geo_levels == ["ctry", "rgn", "la", "msoa", "lsoa", "oa"]
        assert config.output_dir == "data/TS030"  # Default
        assert config.batch_sizes.ctry == 200  # Default

    def test_ensure_output_dir(self):
        """Test output directory creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = os.path.join(tmpdir, "test_output")

            # Directory should not exist yet
            assert not os.path.exists(test_dir)

            # Create the directory
            with patch("logging.Logger.info"):
                ensure_output_dir(test_dir)

            # Directory should now exist
            assert os.path.exists(test_dir)
            assert os.path.isdir(test_dir)

            # Running again should not cause errors
            ensure_output_dir(test_dir)

    @patch("ons_client.cli.setup_parser")
    @patch("ons_client.cli.validate_args")
    @patch("ons_client.cli.create_config")
    @patch("ons_client.cli.ensure_output_dir")
    @patch("ons_client.cli.download_data_for_level")
    @patch("ons_client.cli.configure_logging")
    def test_main_function(
        self,
        mock_configure_logging,
        mock_download,
        mock_ensure_dir,
        mock_create_config,
        mock_validate_args,
        mock_setup_parser,
    ):
        """Test the main function"""
        # Set up mocks
        mock_parser = Mock()
        mock_args = Mock(dataset="TS030", geo_level="ctry", debug=False)
        mock_config = Mock(
            dataset_id="TS030",
            geo_levels=["ctry"],
            output_dir="/tmp/test",
            batch_sizes=Mock(get_for_level=lambda x: 200),
            population_type="UR",
        )

        mock_setup_parser.return_value = mock_parser
        mock_parser.parse_args.return_value = mock_args
        mock_validate_args.return_value = True
        mock_create_config.return_value = mock_config
        mock_download.return_value = "/tmp/test/TS030_ctry.csv"

        # Call main function
        with patch("builtins.print"):
            with patch("time.sleep"):
                main()

        # Verify function calls
        mock_setup_parser.assert_called_once()
        mock_parser.parse_args.assert_called_once()
        mock_validate_args.assert_called_once_with(mock_args)
        mock_create_config.assert_called_once_with(mock_args)
        mock_ensure_dir.assert_called_once_with("/tmp/test")
        mock_download.assert_called_once_with(
            dataset_id="TS030",
            geo_level="ctry",
            output_dir="/tmp/test",
            batch_size=200,
            population_type="UR",
        )

    @patch("ons_client.cli.validate_args")
    def test_main_validation_failure(self, mock_validate_args):
        """Test main function when validation fails"""
        # Mock validation to fail
        mock_validate_args.return_value = False

        # Call main and expect SystemExit
        with patch("ons_client.cli.setup_parser"):
            with patch("ons_client.cli.configure_logging"):
                with pytest.raises(SystemExit) as excinfo:
                    main()

                assert excinfo.value.code == 1
