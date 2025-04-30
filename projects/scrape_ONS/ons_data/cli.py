#!/usr/bin/env python3
"""
Command-line interface for the ONS Census Data Tool
"""

import argparse
import logging
import sys
import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from .api import ApiClientFactory
from .processors import ProcessorFactory
from .models.common import ONSConfig, BatchSizeConfig, DatasetType

# Set up a custom formatter that includes timestamp but minimizes other info
class MinimalFormatter(logging.Formatter):
    """Formatter that only shows minimal information to avoid disrupting progress bars"""
    def format(self, record):
        # For debug level, show more details
        if record.levelno <= logging.DEBUG:
            return super().format(record)

        # For info and above, just show the message
        return f"{record.levelname}: {record.getMessage()}"

# Configure logging - Use minimal formatter to avoid disrupting progress bars
def configure_logging(debug_mode=False):
    """Configure logging with appropriate level and format based on debug mode"""
    root_logger = logging.getLogger()

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set level based on debug mode
    level = logging.DEBUG if debug_mode else logging.INFO
    root_logger.setLevel(level)

    # Create console handler with appropriate formatter
    handler = logging.StreamHandler()

    if debug_mode:
        # In debug mode, show full details
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        # In normal mode, use minimal format
        formatter = MinimalFormatter("%(levelname)s: %(message)s")

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

# Get module-level logger
logger = logging.getLogger("ons_data")


def setup_parser():
    """
    Set up the command-line argument parser.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description='Retrieve and process census data from ONS API')

    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset ID to retrieve (e.g., TS030 for Time Series, RM097 for Regular Matrix)')

    parser.add_argument('--geo-level', type=str, default=None,
                        help='Geographic level to retrieve (e.g., ctry, rgn, la, msoa, lsoa, oa). '
                             'If not specified, all standard levels will be processed.')

    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for data files. Defaults to data/{dataset_id}/')

    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override the default batch size for area processing')

    parser.add_argument('--population-type', type=str, default="UR",
                        help='Population type for census data (default: UR)')

    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    return parser


def validate_args(args):
    """
    Validate command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        bool: True if valid, False otherwise
    """
    # Check dataset ID format
    if not args.dataset:
        logger.error("Dataset ID is required")
        return False

    if len(args.dataset) < 3:
        logger.error(f"Invalid dataset ID format: {args.dataset}")
        return False

    dataset_type_prefix = args.dataset[:2].upper()
    if dataset_type_prefix != "TS" and dataset_type_prefix != "RM":
        logger.error(f"Unsupported dataset type: {dataset_type_prefix}")
        return False

    # Check geo level if specified
    valid_geo_levels = ['ctry', 'rgn', 'la', 'msoa', 'lsoa', 'oa']
    if args.geo_level and args.geo_level not in valid_geo_levels:
        logger.error(f"Invalid geographic level: {args.geo_level}")
        logger.info(f"Valid levels are: {', '.join(valid_geo_levels)}")
        return False

    return True


def create_config(args) -> ONSConfig:
    """
    Create a configuration object from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        ONSConfig: Configuration object
    """
    # Set up default geo levels if none specified
    geo_levels = []
    if args.geo_level:
        geo_levels = [args.geo_level]
    else:
        geo_levels = ['ctry', 'rgn', 'la', 'msoa', 'lsoa', 'oa']

    # Set up output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = f"data/{args.dataset}"

    # Create batch size config (with override if specified)
    batch_sizes = BatchSizeConfig()
    if args.batch_size:
        # Override all batch sizes
        for level in ['ctry', 'rgn', 'la', 'msoa', 'lsoa', 'oa']:
            setattr(batch_sizes, level, args.batch_size)

    # Create and return the config
    return ONSConfig(
        dataset_id=args.dataset,
        geo_levels=geo_levels,
        population_type=args.population_type,
        output_dir=output_dir,
        batch_sizes=batch_sizes
    )


def ensure_output_dir(output_dir: str) -> None:
    """
    Ensure the output directory exists.

    Args:
        output_dir: Path to the output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")


def download_data_for_level(
    dataset_id: str,
    geo_level: str,
    output_dir: str,
    batch_size: int,
    population_type: str = "UR"
) -> Optional[str]:
    """
    Download data for a specific geographic level.

    Args:
        dataset_id: Dataset ID
        geo_level: Geographic level
        output_dir: Output directory
        batch_size: Batch size for processing
        population_type: Population type

    Returns:
        str: Path to the output file or None if failed
    """
    logger.info(f"Processing dataset {dataset_id} for geographic level {geo_level}")

    # Get the API client for this dataset type
    api_client = ApiClientFactory.get_client(dataset_id)
    if not api_client:
        logger.error(f"Failed to create API client for dataset {dataset_id}")
        return None

    # Get the processor for this dataset type
    processor = ProcessorFactory.get_processor(dataset_id)
    if not processor:
        logger.error(f"Failed to create processor for dataset {dataset_id}")
        return None

    try:
        # Define temporary file for API response (we won't keep this)
        temp_file = os.path.join(output_dir, f"{dataset_id}_{geo_level}_raw.csv")

        # Define output file path (this will be the flattened file)
        output_file = os.path.join(output_dir, f"{dataset_id}_{geo_level}.csv")

        # Clean up any old files from previous runs
        old_flat_file = os.path.join(output_dir, f"{dataset_id}_{geo_level}_flat.csv")
        if os.path.exists(old_flat_file):
            os.remove(old_flat_file)
            logger.debug(f"Removed old flattened file: {old_flat_file}")

        # Get all areas for this geographic level
        logger.info(f"Fetching areas for level {geo_level}")
        areas = api_client.get_areas_for_level(geo_level, population_type)
        if not areas:
            logger.warning(f"No areas found for level {geo_level}")
            return None

        logger.info(f"Found {len(areas)} areas for level {geo_level}")

        # Extract area codes
        area_codes = [area["id"] for area in areas]

        # Get dataset data in batches
        if dataset_id.startswith("TS"):
            response = api_client.batch_get_dataset_data(
                dataset_id=dataset_id,
                area_codes=area_codes,
                geo_level=geo_level,
                batch_size=batch_size,
                population_type=population_type
            )
        else:  # RM dataset
            response = api_client.batch_get_dataset_data(
                dataset_id=dataset_id,
                area_codes=area_codes,
                geo_level=geo_level,
                batch_size=batch_size,
                population_type=population_type
            )

        # Process the response: First save to temp file
        temp_result = processor.process_response(response, temp_file)

        if not temp_result:
            logger.error(f"Failed to process data for level {geo_level}")
            return None

        # Then directly flatten it to the final output file
        flat_result = processor.flatten_data(temp_file, output_file)

        # Remove the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            logger.debug(f"Removed temporary file {temp_file}")

        # Also remove the debug JSON if it exists
        debug_file = f"{temp_file}.debug.json"
        if os.path.exists(debug_file):
            # We need to keep the debug file for the flattening to work correctly
            # Just rename it to be associated with the output file
            new_debug_file = f"{output_file}.debug.json"
            os.rename(debug_file, new_debug_file)
            logger.debug(f"Renamed debug file to {new_debug_file}")

        if flat_result:
            logger.info(f"Successfully processed and flattened data for level {geo_level}")
            return output_file
        else:
            logger.error(f"Failed to flatten data for level {geo_level}")
            return None

    except Exception as e:
        logger.error(f"Error processing level {geo_level}: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def main():
    """Main entry point for the CLI"""
    # Parse command-line arguments
    parser = setup_parser()
    args = parser.parse_args()

    # Configure logging based on debug mode
    configure_logging(args.debug)

    # Validate arguments
    if not validate_args(args):
        sys.exit(1)

    # Create configuration
    config = create_config(args)
    logger.info(f"Processing dataset {config.dataset_id}")

    # Ensure output directory exists
    ensure_output_dir(config.output_dir)

    # Track results
    results = {}
    failed_levels = []

    # Process each geographic level
    for geo_level in config.geo_levels:
        try:
            # Get batch size for this level
            batch_size = config.batch_sizes.get_for_level(geo_level)
            logger.info(f"Using batch size of {batch_size} for {geo_level}")

            # Download and flatten data for this level
            output_file = download_data_for_level(
                dataset_id=config.dataset_id,
                geo_level=geo_level,
                output_dir=config.output_dir,
                batch_size=batch_size,
                population_type=config.population_type
            )

            if output_file:
                results[geo_level] = output_file
            else:
                failed_levels.append(geo_level)

            # Add a brief pause between levels to avoid rate limiting
            if geo_level != config.geo_levels[-1]:  # If not the last level
                pause_time = 5  # 5 seconds
                logger.info(f"Pausing for {pause_time} seconds before next level...")
                time.sleep(pause_time)

        except Exception as e:
            logger.error(f"Error processing level {geo_level}: {str(e)}")
            failed_levels.append(geo_level)

    # Print summary
    print("\n=== DOWNLOAD SUMMARY ===")
    print(f"Successfully processed {len(results)} geographic levels")
    for level, file_path in results.items():
        print(f"✅ {level}: {file_path}")

    if failed_levels:
        print(f"❌ Failed to process {len(failed_levels)} levels: {', '.join(failed_levels)}")

    logger.info(f"Completed processing for dataset {config.dataset_id}")


if __name__ == "__main__":
    main()
