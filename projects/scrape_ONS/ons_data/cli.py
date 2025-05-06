#!/usr/bin/env python3
"""
Command-line interface for the ONS Census Data Tool
"""

import argparse
import logging
import sys
import os
import time
import json
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
    parser = argparse.ArgumentParser(
        description="Retrieve and process census data from ONS API"
    )

    # Create a mutually exclusive group for main commands
    main_group = parser.add_mutually_exclusive_group(required=True)

    main_group.add_argument(
        "--dataset",
        type=str,
        help="Dataset ID to retrieve (e.g., TS030 for Time Series, RM097 for Regular Matrix)",
    )

    main_group.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets with IDs and names",
    )

    main_group.add_argument(
        "--print-datasets",
        action="store_true",
        help="Print all available datasets in a simple format",
    )

    parser.add_argument(
        "--geo-level",
        type=str,
        default=None,
        help="Geographic level to retrieve (e.g., ctry, rgn, la, msoa, lsoa, oa). "
        "If not specified, all standard levels will be processed.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for data files. Defaults to data/{dataset_id}/",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the default batch size for area processing",
    )

    parser.add_argument(
        "--population-type",
        type=str,
        default="UR",
        help="Population type for census data (default: UR)",
    )

    parser.add_argument(
        "--use-filter",
        action="store_true",
        help="Use ONS Filter API for data retrieval (recommended for large datasets)",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser


def validate_args(args):
    """
    Validate command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        bool: True if valid, False otherwise
    """
    # If we're just listing datasets or printing datasets, no further validation needed
    if args.list_datasets or (hasattr(args, "print_datasets") and args.print_datasets):
        return True

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
    valid_geo_levels = ["ctry", "rgn", "la", "msoa", "lsoa", "oa"]
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
        geo_levels = ["ctry", "rgn", "la", "msoa", "lsoa", "oa"]

    # Set up output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = f"data/{args.dataset}"

    # Create batch size config (with override if specified)
    batch_sizes = BatchSizeConfig()
    if args.batch_size:
        # Override all batch sizes
        for level in ["ctry", "rgn", "la", "msoa", "lsoa", "oa"]:
            setattr(batch_sizes, level, args.batch_size)

    # Create and return the config
    return ONSConfig(
        dataset_id=args.dataset,
        geo_levels=geo_levels,
        population_type=args.population_type,
        output_dir=output_dir,
        batch_sizes=batch_sizes,
        use_filter=args.use_filter,
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
    population_type: str = "UR",
    use_filter: bool = False,
) -> Optional[str]:
    """
    Download data for a specific geographic level.

    Args:
        dataset_id: Dataset ID
        geo_level: Geographic level
        output_dir: Output directory
        batch_size: Batch size for processing
        population_type: Population type
        use_filter: Whether to use the filter API instead of batch download

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
        # First check if this dataset is available at this geographic level
        logger.info(
            f"Checking if dataset {dataset_id} is available at geographic level {geo_level}"
        )
        availability = api_client.check_dataset_availability(
            dataset_id, geo_level, population_type
        )

        if not availability.is_available:
            error_msg = (
                availability.error_message
                or f"Dataset {dataset_id} is not available at geographic level {geo_level}"
            )
            logger.warning(error_msg)
            logger.warning(
                f"Skipping geographic level {geo_level} for dataset {dataset_id}"
            )
            return None

        logger.info(
            f"Dataset {dataset_id} is available at geographic level {geo_level}, proceeding with download"
        )

        # Define temporary file for API response (we won't keep this)
        temp_file = os.path.join(output_dir, f"{dataset_id}_{geo_level}_raw.csv")

        # Define output file path (this will be the flattened file)
        output_file = os.path.join(output_dir, f"{dataset_id}_{geo_level}.csv")

        # Clean up any old files from previous runs
        old_flat_file = os.path.join(output_dir, f"{dataset_id}_{geo_level}_flat.csv")
        if os.path.exists(old_flat_file):
            os.remove(old_flat_file)
            logger.debug(f"Removed old flattened file: {old_flat_file}")

        # Large geographic levels that are better processed using filter API
        large_geo_levels = ["lsoa", "msoa", "oa"]

        # Decide whether to use filter API
        should_use_filter = use_filter or (geo_level in large_geo_levels)

        if should_use_filter:
            logger.info(f"Using filter API for level {geo_level} (large dataset)")

            # Use the filter API directly
            filtered_file = api_client.get_dataset_using_filter(
                dataset_id=dataset_id,
                geo_level=geo_level,
                output_dir=output_dir,
                population_type=population_type,
            )

            if not filtered_file:
                logger.error(
                    f"Failed to retrieve data using filter API for level {geo_level}"
                )
                return None

            # Also create a debug.json file for consistency
            debug_file = f"{output_file}.debug.json"
            try:
                # Get minimal metadata for debug file
                areas = api_client.get_areas_for_level(geo_level, population_type)
                sample_area_codes = [
                    area["id"] for area in areas[:5]
                ]  # Just get a few areas

                sample_data = api_client.get_dataset_data(
                    dataset_id=dataset_id,
                    area_codes=sample_area_codes,
                    geo_level=geo_level,
                    population_type=population_type,
                )

                # Save minimal debug info
                with open(debug_file, "w") as f:
                    json.dump(sample_data, f, indent=2)
                logger.debug(f"Created debug file with sample metadata: {debug_file}")
            except Exception as e:
                logger.warning(
                    f"Failed to create debug file, but data was retrieved: {str(e)}"
                )

            logger.info(
                f"Successfully retrieved data for level {geo_level} using filter API"
            )
            return filtered_file
        else:
            # Use the original batch processing method for smaller datasets
            # Get all areas for this geographic level
            logger.info(f"Using batch processing for level {geo_level}")
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
                    population_type=population_type,
                )
            else:  # RM dataset
                response = api_client.batch_get_dataset_data(
                    dataset_id=dataset_id,
                    area_codes=area_codes,
                    geo_level=geo_level,
                    batch_size=batch_size,
                    population_type=population_type,
                )

            # Process the response: First save to temp file
            temp_result = processor.process_response(
                response, temp_file, area_metadata={"area_codes": area_codes}
            )

            if not temp_result:
                logger.error(f"Failed to process data for level {geo_level}")
                return None

            # Then directly flatten it to the final output file with area metadata
            flat_result = processor.flatten_data(
                temp_file,
                output_file,
                area_metadata={"area_codes": area_codes, "geo_level": geo_level},
            )

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
                logger.info(
                    f"Successfully processed and flattened data for level {geo_level}"
                )
                return output_file
            else:
                logger.error(f"Failed to flatten data for level {geo_level}")
                return None

    except Exception as e:
        logger.error(f"Error processing level {geo_level}: {str(e)}")
        import traceback

        logger.debug(traceback.format_exc())
        return None


def list_datasets():
    """
    Fetch and display all available datasets from the ONS API.

    Lists datasets with their IDs, dataset type (TS/RM), and titles.
    Provides information about Census 2021 datasets and how to find them.
    """
    try:
        # Create a base API client (dataset type doesn't matter for listing)
        from .api.client import ONSApiClient

        client = ONSApiClient()

        print("\n=== ONS DATASETS ===\n")

        # First, provide information about Census datasets
        print("### CENSUS 2021 DATASETS ###")
        print("Census datasets typically use the following prefixes:")
        print("- TS: Time Series datasets (e.g., TS030 for Religion)")
        print("- RM: Regular Matrix datasets (e.g., RM097 for Occupancy Rating)")
        print()
        print("Common Census 2021 datasets include:")
        print("- TS030: Religion by geographic area")
        print("- TS008: Sex by geographic area")
        print("- TS021: Age by geographic area")
        print("- RM097: Occupancy rating (bedrooms) by ethnic group")
        print("- RM040: Age by sex by long-term health condition")
        print("- RM052: Country of birth by sex")
        print()
        print(
            "Note: The ONS API may not list all Census datasets in the general datasets endpoint."
        )
        print("You may need to search for specific dataset IDs on the ONS website:")
        print("https://www.ons.gov.uk/census/census2021dictionary")
        print()

        # Fetch general datasets
        logger.info("Fetching available datasets from ONS API...")
        datasets = client.get_datasets()

        if not datasets:
            logger.error("No datasets found in the general API endpoint")
            return

        # Sort datasets by ID
        datasets.sort(key=lambda x: x.id)

        # Group by dataset type
        ts_datasets = [ds for ds in datasets if ds.id.startswith("TS")]
        rm_datasets = [ds for ds in datasets if ds.id.startswith("RM")]
        other_datasets = [
            ds
            for ds in datasets
            if not ds.id.startswith("TS") and not ds.id.startswith("RM")
        ]

        # Display results
        print("### AVAILABLE DATASETS FROM API ###")

        if ts_datasets:
            print(f"\n--- TIME SERIES DATASETS ({len(ts_datasets)}) ---")
            for ds in ts_datasets:
                print(f"{ds.id}: {ds.title}")
                if ds.description:
                    print(f"   {ds.description}")
        else:
            print("\n--- TIME SERIES DATASETS ---")
            print("No Time Series datasets found in the general API endpoint.")

        if rm_datasets:
            print(f"\n--- REGULAR MATRIX DATASETS ({len(rm_datasets)}) ---")
            for ds in rm_datasets:
                print(f"{ds.id}: {ds.title}")
                if ds.description:
                    print(f"   {ds.description}")
        else:
            print("\n--- REGULAR MATRIX DATASETS ---")
            print("No Regular Matrix datasets found in the general API endpoint.")

        if other_datasets:
            print(f"\n--- OTHER DATASETS ({len(other_datasets)}) ---")
            for i, ds in enumerate(other_datasets):
                # Only show the first 5 datasets to avoid overwhelming output
                if i < 5:
                    print(f"{ds.id}: {ds.title}")
                elif i == 5:
                    print(f"... and {len(other_datasets) - 5} more datasets")
                    break

        print(f"\nTotal: {len(datasets)} datasets found in the general API endpoint")
        print("\n### USAGE ###")
        print("To use a specific dataset, run the command with:")
        print("python run_census.py --dataset DATASET_ID --geo-level LEVEL")
        print("\nExample for a country-level religion dataset:")
        print("python run_census.py --dataset TS030 --geo-level ctry")

    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        import traceback

        logger.debug(traceback.format_exc())


def print_datasets():
    """
    Fetch and display all datasets from the ONS API in a simple format.

    This function creates an ONSApiClient, calls get_datasets(),
    and prints each dataset with its ID, title, and description.
    Displays results in pages to handle large numbers of datasets.
    """
    try:
        # Create a base API client
        from .api.client import ONSApiClient

        client = ONSApiClient()

        print("\n=== ONS DATASETS ===\n")

        # Fetch datasets
        logger.info("Fetching datasets from ONS API...")
        datasets = client.get_datasets()

        if not datasets:
            logger.error("No datasets found")
            return

        # Separate TS and RM datasets
        ts_datasets = [ds for ds in datasets if ds.id.startswith("TS")]
        rm_datasets = [ds for ds in datasets if ds.id.startswith("RM")]
        other_datasets = [
            ds
            for ds in datasets
            if not ds.id.startswith("TS") and not ds.id.startswith("RM")
        ]

        # Function to display datasets in pages
        def display_datasets(dataset_list, title):
            if not dataset_list:
                print(f"\nNO {title} DATASETS FOUND")
                return

            print(f"\n=== {title} DATASETS ({len(dataset_list)}) ===\n")

            # Display in pages of 10 datasets each
            page_size = 10
            total_pages = (len(dataset_list) + page_size - 1) // page_size
            current_page = 1

            while True:
                start_idx = (current_page - 1) * page_size
                end_idx = min(start_idx + page_size, len(dataset_list))

                for dataset in dataset_list[start_idx:end_idx]:
                    print(f"ID: {dataset.id}")
                    print(f"Type: {dataset.dataset_type}")
                    print(f"Title: {dataset.title}")
                    if dataset.description:
                        # Truncate long descriptions
                        desc = dataset.description
                        if len(desc) > 200:
                            desc = desc[:197] + "..."
                        print(f"Description: {desc}")
                    print("---")

                # If we have more than one page, show navigation options
                if total_pages > 1:
                    print(f"\nPage {current_page} of {total_pages} | ", end="")

                    if current_page > 1:
                        print("[p] Previous | ", end="")

                    if current_page < total_pages:
                        print("[n] Next | ", end="")

                    print("[a] All Types | [s] Skip | [q] Quit")

                    choice = input("Enter choice: ").strip().lower()

                    if choice == "p" and current_page > 1:
                        current_page -= 1
                    elif choice == "n" and current_page < total_pages:
                        current_page += 1
                    elif choice == "a":
                        return "all"  # Signal to go back to all types
                    elif choice == "s":
                        return "skip"  # Signal to skip to next type
                    elif choice == "q":
                        return "quit"  # Signal to quit
                    # Any other input just refreshes the current page
                else:
                    # If only one page, just ask to continue
                    if len(dataset_list) > 5:  # Only prompt if we have enough items
                        print("\n[Enter] Continue | [a] All Types | [q] Quit")
                        choice = input("Enter choice: ").strip().lower()
                        if choice == "a":
                            return "all"
                        elif choice == "q":
                            return "quit"
                    return "skip"  # Move to next type by default

        # Display all dataset types
        while True:
            print("\n=== DATASET TYPES ===")
            print(f"[1] TIME SERIES (TS) DATASETS ({len(ts_datasets)})")
            print(f"[2] REGULAR MATRIX (RM) DATASETS ({len(rm_datasets)})")
            print(f"[3] OTHER DATASETS ({len(other_datasets)})")
            print("[4] SUMMARY ONLY")
            print("[q] QUIT")

            choice = input("\nEnter choice (1-4 or q): ").strip().lower()

            if choice == "1":
                result = display_datasets(ts_datasets, "TIME SERIES (TS)")
                if result == "quit":
                    break
            elif choice == "2":
                result = display_datasets(rm_datasets, "REGULAR MATRIX (RM)")
                if result == "quit":
                    break
            elif choice == "3":
                result = display_datasets(other_datasets, "OTHER")
                if result == "quit":
                    break
            elif choice == "4" or choice == "":
                # Just show summary
                break
            elif choice == "q":
                return
            else:
                print("Invalid choice. Please try again.")

        # Print summary
        print(f"\n=== SUMMARY ===")
        print(f"Total: {len(datasets)} datasets")
        print(f"- {len(ts_datasets)} Time Series (TS) datasets")
        print(f"- {len(rm_datasets)} Regular Matrix (RM) datasets")
        print(f"- {len(other_datasets)} Other datasets")
        print("\nNOTE: Common Census datasets are included in this list.")
        print("Additional TS/RM datasets may be available on the ONS website:")
        print("https://www.ons.gov.uk/census/census2021dictionary")

    except Exception as e:
        logger.error(f"Error fetching datasets: {str(e)}")
        import traceback

        logger.debug(traceback.format_exc())


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

    # Handle list datasets command
    if args.list_datasets:
        list_datasets()
        return

    # Handle print datasets command
    if hasattr(args, "print_datasets") and args.print_datasets:
        print_datasets()
        return

    # Create configuration
    config = create_config(args)
    logger.info(f"Processing dataset {config.dataset_id}")

    # Ensure output directory exists
    ensure_output_dir(config.output_dir)

    # Track results
    results = {}
    failed_levels = []
    skipped_levels = []  # New: Track skipped levels due to dataset unavailability

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
                population_type=config.population_type,
                use_filter=config.use_filter,
            )

            if output_file:
                results[geo_level] = output_file
            else:
                # Check if this was a dataset availability issue
                api_client = ApiClientFactory.get_client(config.dataset_id)
                if api_client:
                    availability = api_client.check_dataset_availability(
                        config.dataset_id, geo_level, config.population_type
                    )
                    if not availability.is_available:
                        skipped_levels.append(
                            (
                                geo_level,
                                availability.error_message
                                or "Dataset not available at this level",
                            )
                        )
                    else:
                        failed_levels.append(geo_level)
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
    print(f"Dataset: {config.dataset_id}")

    if results:
        print(f"\nSuccessfully processed {len(results)} geographic levels:")
        for level, file_path in results.items():
            print(f"✅ {level}: {file_path}")
    else:
        print("\nNo geographic levels were successfully processed.")

    if skipped_levels:
        print(f"\nSkipped {len(skipped_levels)} levels due to dataset unavailability:")
        for level, reason in skipped_levels:
            print(f"⚠️ {level}: {reason}")

    if failed_levels:
        print(f"\nFailed to process {len(failed_levels)} levels:")
        for level in failed_levels:
            print(f"❌ {level}")

    # Final success/failure message
    if results:
        logger.info(f"Completed processing for dataset {config.dataset_id}")
    else:
        logger.error(
            f"Failed to process any geographic levels for dataset {config.dataset_id}"
        )
        if skipped_levels and not failed_levels:
            # If all failures were due to dataset unavailability
            print(
                "\nSuggestion: This dataset may not be available at the requested geographic levels."
            )
            print("Try using different geographic levels or a different dataset.")
        elif (
            len(config.geo_levels) == 1
            and config.geo_levels[0] == "oa"
            and skipped_levels
        ):
            # Special case for OA level, which is often not available
            print(
                "\nNote: Output Area (OA) level data is not available for all datasets."
            )
            print(
                "Try using a higher geographic level like LSOA, MSOA, or Local Authority (LA)."
            )

    # Set exit code for system
    if not results:
        sys.exit(1)


def process_dataset(
    dataset_id: str,
    geo_level: str = None,
    output_dir: str = None,
    batch_size: int = None,
    population_type: str = "UR",
    debug: bool = False,
    use_filter: bool = False,
    **kwargs,
) -> Optional[str]:
    """
    Process a dataset for a specific geographic level.
    This is a wrapper around download_data_for_level for backward compatibility.

    Args:
        dataset_id: Dataset ID to process
        geo_level: Geographic level to process
        output_dir: Output directory for data files
        batch_size: Batch size for processing
        population_type: Population type
        debug: Enable debug mode
        use_filter: Whether to use the filter API instead of batch download
        **kwargs: Additional arguments

    Returns:
        Path to the output file or None if failed
    """
    # Set defaults
    if output_dir is None:
        output_dir = f"data/{dataset_id}"
    if geo_level is None:
        geo_level = "ctry"  # Default to country level
    if batch_size is None:
        batch_size = 200  # Default batch size

    # Configure logging based on debug mode
    configure_logging(debug)

    # Check if we need to create a nested directory structure
    # The tests expect output_dir/dataset_id/geo_level.csv
    if not output_dir.endswith(dataset_id):
        nested_output_dir = os.path.join(output_dir, dataset_id)
    else:
        nested_output_dir = output_dir

    # Ensure output directory exists
    ensure_output_dir(nested_output_dir)

    # Process the dataset
    result = download_data_for_level(
        dataset_id=dataset_id,
        geo_level=geo_level,
        output_dir=nested_output_dir,
        batch_size=batch_size,
        population_type=population_type,
        use_filter=use_filter,
    )

    # The e2e tests expect a different naming convention
    if result is not None:
        # If download_data_for_level returns a path like output_dir/dataset_id_geo_level.csv
        # we need to check if an alternate format file should be created
        if not os.path.exists(os.path.join(nested_output_dir, f"{geo_level}.csv")):
            alt_path = os.path.join(nested_output_dir, f"{geo_level}.csv")
            # Create a copy with the expected name for backward compatibility
            try:
                import shutil

                shutil.copy(result, alt_path)
                return alt_path
            except Exception as e:
                logger.warning(f"Could not create alternate format file: {e}")

    return result


if __name__ == "__main__":
    main()
