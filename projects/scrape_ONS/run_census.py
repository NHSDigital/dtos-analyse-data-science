#!/usr/bin/env python3
"""
Wrapper script to run the fast_census_data functions
"""

import sys
import os
import logging
import csv
import json
import time
import random
import requests
import argparse
from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ons_client.fast_census_data import download_all_areas_for_level, get_fast_census_data, get_areas_for_level
from ons_client.ons_client import ONSApiClient
from ons_client.models import Dataset, DatasetsResponse, TSDatasetResponse, RMDatasetResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("run_census")

# Optimized batch sizes for different geographic levels
# Smaller areas (more numerous) get smaller batch sizes to avoid URL length limits
BATCH_SIZES = {
    "ctry": 200,    # Countries (few)
    "rgn": 200,     # Regions (few)
    "la": 200,       # Local Authorities (moderate)
    "msoa": 200,     # MSOAs (many) - very small batch size to avoid URL length issues
    "lsoa": 200,     # LSOAs (many) - very small batch size to avoid URL length issues
    "oa": 200        # Output Areas (very many) - one at a time
}

def get_non_ltna_datasets() -> List[Dataset]:
    """
    Get all available datasets from ONS API and filter out those that are ltna-specific.

    Returns:
        List[Dataset]: List of dataset objects that aren't specific to ltna area types
    """
    client = ONSApiClient()
    datasets_response = client.get_datasets()

    # Filter out ltna-specific datasets
    filtered_datasets = []
    for dataset in datasets_response.datasets:
        # Add to the list if it's not ltna-specific (checking title/description)
        if "ltna" not in dataset.title.lower() and (not dataset.description or "ltna" not in dataset.description.lower()):
            filtered_datasets.append(dataset)

    logger.info(f"Found {len(filtered_datasets)} datasets (excluding ltna-specific)")
    return filtered_datasets

def get_available_dataset_dimensions(dataset_id: str, population_type: str = "UR") -> List[Dict[str, str]]:
    """
    Query all available dimensions for a specific dataset from the ONS API.

    Args:
        dataset_id (str): The dataset ID to get dimensions for
        population_type (str): The population type (default: "UR")

    Returns:
        List[Dict[str, str]]: List of dimension dictionaries with id and label
    """
    # Initialize the ONS client
    client = ONSApiClient()

    try:
        # Use the provided population type
        logger.info(f"Retrieving dimensions for population type: {population_type}")
        dimensions = client.get_dimensions(population_type)

        # Convert to list of dictionaries
        dimension_list = [{"id": dim.id, "label": dim.label} for dim in dimensions]

        logger.info(f"Found {len(dimension_list)} dimensions for dataset {dataset_id}")

        # Log the available dimensions
        for dim in dimension_list:
            logger.info(f"  - {dim['id']}: {dim['label']}")

        return dimension_list

    except Exception as e:
        logger.error(f"Error retrieving dimensions for dataset {dataset_id}: {str(e)}")
        return []

# Add a wrapper function for download_all_areas_for_level with retry logic
def download_with_retry(
    dataset_id: str,
    geo_level: str,
    population_type: str = "UR",
    max_retries: int = 3,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    **kwargs
) -> Optional[str]:
    """
    Attempt to download data with retry logic and exponential backoff.

    Args:
        dataset_id: The dataset ID to download
        geo_level: The geographic level
        population_type: The population type
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay between retries
        **kwargs: Additional arguments to pass to download_all_areas_for_level

    Returns:
        Optional[str]: Path to the output file if successful, None otherwise
    """
    attempt = 0
    delay = initial_delay

    while attempt <= max_retries:
        try:
            # Attempt to download the data
            if attempt > 0:
                logger.info(f"Retry attempt {attempt}/{max_retries} for {geo_level} level")

            # If we're on 2nd+ attempt and batch_size is > 1, reduce it by half
            if attempt > 0 and 'batch_size' in kwargs and kwargs['batch_size'] > 1:
                original_batch_size = kwargs['batch_size']
                # Reduce batch size by half on each retry to address URL length issues
                kwargs['batch_size'] = max(1, kwargs['batch_size'] // 2)
                logger.info(f"Reducing batch size from {original_batch_size} to {kwargs['batch_size']} for retry")

            return download_all_areas_for_level(
                dataset_id=dataset_id,
                geo_level=geo_level,
                population_type=population_type,
                **kwargs
            )

        except requests.exceptions.HTTPError as e:
            # Handle different HTTP errors
            status_code = e.response.status_code if hasattr(e, 'response') else None

            if status_code == 429:  # Too Many Requests
                logger.warning(f"Rate limit exceeded for {geo_level} level (429 Too Many Requests)")
                # Always retry with longer delay for rate limiting
                retry = True
                delay = max(delay, 60.0)  # At least 60 seconds for rate limit errors

            elif status_code in [502, 503, 504, 520, 521, 522, 523, 524]:  # Server errors, including Cloudflare
                logger.warning(f"Server error {status_code} for {geo_level} level, may retry")
                retry = attempt < max_retries

                # For these errors, we might have URL length issues - reduce batch size more aggressively
                if 'batch_size' in kwargs and kwargs['batch_size'] > 1:
                    original_batch_size = kwargs['batch_size']
                    kwargs['batch_size'] = max(1, kwargs['batch_size'] // 4)  # More aggressive reduction
                    logger.warning(f"Aggressively reducing batch size from {original_batch_size} to {kwargs['batch_size']} due to server error")

            else:
                # For other HTTP errors, don't retry after first attempt
                logger.error(f"HTTP error {status_code} for {geo_level} level: {str(e)}")
                retry = False

            if not retry or attempt >= max_retries:
                logger.error(f"Giving up on {geo_level} level after {attempt+1} attempts")
                return None

        except Exception as e:
            # For non-HTTP errors, retry up to max_retries
            logger.error(f"Error processing {geo_level} level: {str(e)}")

            # Check if this might be a URL length issue
            if str(e).lower().find("url") >= 0 or len(str(e)) > 1000:
                logger.warning("Possible URL length issue detected")
                if 'batch_size' in kwargs and kwargs['batch_size'] > 1:
                    original_batch_size = kwargs['batch_size']
                    kwargs['batch_size'] = max(1, kwargs['batch_size'] // 4)  # More aggressive reduction
                    logger.warning(f"Aggressively reducing batch size from {original_batch_size} to {kwargs['batch_size']} due to possible URL length issue")

            if attempt >= max_retries:
                logger.error(f"Giving up on {geo_level} level after {attempt+1} attempts")
                return None

        # Calculate delay with jitter for next retry
        jitter = random.uniform(0.8, 1.2)
        sleep_time = delay * jitter
        logger.info(f"Waiting {sleep_time:.1f} seconds before retry...")
        time.sleep(sleep_time)

        # Increase delay for next attempt
        delay *= backoff_factor
        attempt += 1

    # Should not reach here, but just in case
    return None

def flatten_csv(input_file: str, output_file: str = None) -> str:
    """
    Convert the CSV with JSON dimensions to a flat CSV with separate columns.

    Can handle:
    1. Standard CSVs with a 'dimensions' column containing JSON
    2. Simple CSVs with just a 'value' column (uses metadata to create dimensions)

    Args:
        input_file: Path to the input CSV file with JSON dimensions
        output_file: Path to the output flat CSV file (default: append _flat to input filename)

    Returns:
        str: Path to the flattened CSV file, or None if no flattening was done
    """
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_flat{ext}"

    # Read the input CSV
    rows = []
    fieldnames = []
    with open(input_file, 'r', newline='') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    if not rows:
        logger.warning(f"No data found in {input_file}")
        return None

    # Check if this is a simple CSV with only a 'value' column
    if fieldnames == ['value'] or (len(fieldnames) == 1 and 'dimensions' not in fieldnames):
        logger.info(f"CSV at {input_file} is a simple value format with columns: {fieldnames}")

        # For TS datasets, we need to attempt to create dimensions
        # This requires getting metadata from the dataset
        debug_file = f"{input_file}.debug.json"
        metadata = None

        # Check if debug file exists with raw response
        if os.path.exists(debug_file):
            try:
                with open(debug_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Found debug file with metadata: {debug_file}")
            except:
                logger.warning(f"Couldn't parse debug file: {debug_file}")

        # If we have metadata, attempt to create a structured flat file
        if metadata and isinstance(metadata, dict) and 'dimensions' in metadata and metadata['dimensions']:
            try:
                logger.info("Attempting to create a structured flat file from metadata")

                # Extract dimension information
                dimension_data = metadata.get('dimensions', [])
                headers = metadata.get('headers', [])

                # Get all possible dimension values if available
                dimensions = []
                dimension_values = {}

                # First, identify the dimensions we have
                if isinstance(dimension_data, list):
                    for dim in dimension_data:
                        if isinstance(dim, dict) and 'dimension_name' in dim:
                            dim_name = dim.get('dimension_name')
                            dimensions.append(dim_name)
                            # Store dimension values if available
                            if 'options' in dim:
                                dimension_values[dim_name] = dim['options']
                                logger.info(f"Found dimension {dim_name} with {len(dim['options'])} options")

                if dimensions:
                    logger.info(f"Found dimensions in metadata: {dimensions}")

                    # Get the observations
                    observations = [row.get('value', '') for row in rows]

                    # Create structured rows from dimensions and values
                    if dimension_values:
                        # Create fieldnames for all dimensions and their codes
                        fieldnames = []
                        for dim in dimensions:
                            fieldnames.append(dim)
                            fieldnames.append(f"{dim}_code")
                        fieldnames.append('observation')

                        # Generate a row for each observation based on dimension combinations
                        # This requires understanding the ordering of observations in the API response
                        flat_rows = []

                        # Calculate the total number of combinations for each dimension
                        dim_counts = [len(dimension_values.get(dim, [])) for dim in dimensions]
                        logger.info(f"Dimension counts: {dim_counts}")

                        # Calculate product of all dimension counts
                        total_combinations = 1
                        for count in dim_counts:
                            total_combinations *= count

                        # Verify observation count matches expected combinations
                        logger.info(f"Total combinations: {total_combinations}, Observations: {len(observations)}")

                        # For TS datasets, create all possible dimension combinations
                        # and map them to observations
                        if len(observations) == total_combinations or len(observations) < total_combinations:
                            logger.info(f"Creating {total_combinations} rows with dimension combinations")

                            # Initialize indices for each dimension
                            indices = [0] * len(dimensions)
                            obs_index = 0

                            # Generate all dimension combinations
                            # This matches ONS API's ordering logic
                            while obs_index < len(observations) and obs_index < total_combinations:
                                # Create a new row
                                row = {}

                                # Add dimension values and codes for this combination
                                for i, dim in enumerate(dimensions):
                                    options = dimension_values.get(dim, [])
                                    if options and indices[i] < len(options):
                                        opt = options[indices[i]]
                                        row[dim] = opt.get('label', '')
                                        row[f"{dim}_code"] = opt.get('id', '')

                                # Add the observation value
                                row['observation'] = observations[obs_index]
                                flat_rows.append(row)

                                # Increment indices (like counting with carry)
                                # First dimension changes slowest, last dimension changes fastest
                                # This matches how the ONS API returns observations
                                # Outer loop (first dimension) corresponds to first elements
                                for i in range(len(indices) - 1, -1, -1):
                                    indices[i] += 1
                                    if indices[i] < dim_counts[i]:
                                        break
                                    indices[i] = 0

                                # Move to next observation
                                obs_index += 1

                            # Write the flat CSV with dimension info
                            if flat_rows:
                                with open(output_file, 'w', newline='') as outfile:
                                    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                                    writer.writeheader()
                                    writer.writerows(flat_rows)

                                logger.info(f"Created structured flat file with {len(flat_rows)} rows")
                                return output_file

            except Exception as e:
                logger.error(f"Error creating flat file from metadata: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # If we couldn't create a structured file, just copy the original
        logger.info(f"Creating a simple copy as the flattened version")
        import shutil
        shutil.copy(input_file, output_file)
        logger.info(f"Created flattened copy at: {output_file}")
        return output_file

    # Check if 'dimensions' column exists for regular processing
    if 'dimensions' not in fieldnames:
        logger.warning(f"No 'dimensions' column found in {input_file}. Columns: {fieldnames}")
        return None

    # Extract column names from dimensions
    flat_rows = []
    dimension_keys = set()

    for row in rows:
        flat_row = {}

        # Parse the dimensions from JSON string to Python object
        try:
            dimensions = json.loads(row['dimensions'].replace("'", '"'))
        except:
            # print(row)
            # logger.warning(f"Could not parse dimensions: {row['dimensions'][:50]}...")
            continue

        # Extract values from dimensions
        for dim in dimensions:
            if 'dimension_id' in dim and 'option' in dim:
                key = dim['dimension_id']
                dimension_keys.add(key)
                flat_row[key] = dim['option']

                # Also add the area code if available
                if 'option_id' in dim:
                    area_key = f"{key}_code"
                    dimension_keys.add(area_key)
                    flat_row[area_key] = dim['option_id']

        # Add the observation value
        flat_row['observation'] = row['observation']
        flat_rows.append(flat_row)

    # Write the flattened CSV
    if flat_rows:
        fieldnames = sorted(list(dimension_keys)) + ['observation']
        with open(output_file, 'w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)

        logger.info(f"Flattened CSV saved to: {output_file}")
        return output_file
    else:
        logger.warning("No rows to write to flattened CSV")
        return None

def inspect_dataset_response(response_file):
    """
    Inspect and log details about the dataset response from a JSON file

    Args:
        response_file (str): Path to the JSON response file
    """
    try:
        if not os.path.exists(response_file):
            logger.warning(f"Response file not found: {response_file}")
            return

        with open(response_file, 'r') as f:
            data = json.load(f)

        logger.info("=== DATASET RESPONSE INSPECTION ===")
        logger.info(f"Response file: {response_file}")
        logger.info(f"Response keys: {list(data.keys())}")

        # Check for dimensions
        if 'dimensions' in data:
            dimensions = data['dimensions']
            logger.info(f"Dimensions type: {type(dimensions).__name__}")

            if isinstance(dimensions, list):
                logger.info(f"Found {len(dimensions)} dimensions")

                # Log dimension details
                for i, dim in enumerate(dimensions[:3]):  # Log first 3 dimensions
                    logger.info(f"Dimension {i} type: {type(dim).__name__}")
                    if isinstance(dim, dict):
                        logger.info(f"Dimension {i} keys: {list(dim.keys())}")

                        # Look for dimension name
                        dim_name = dim.get('dimension_name', dim.get('name', 'unknown'))
                        logger.info(f"Dimension {i} name: {dim_name}")

                        # Check for options
                        if 'options' in dim:
                            options = dim['options']
                            logger.info(f"Dimension {i} has {len(options)} options")

                            if options:
                                logger.info(f"First option type: {type(options[0]).__name__}")
                                if isinstance(options[0], dict):
                                    logger.info(f"First option keys: {list(options[0].keys())}")

                # Dimensions should have the same number of options as observations length
                if 'observations' in data:
                    expected_length = 1
                    for dim in dimensions:
                        if isinstance(dim, dict) and 'options' in dim:
                            expected_length *= len(dim['options'])

                    actual_length = len(data['observations'])
                    logger.info(f"Expected observations count: {expected_length}")
                    logger.info(f"Actual observations count: {actual_length}")

                    if expected_length != actual_length:
                        logger.warning(f"Mismatch between expected and actual observation count")

        # Check for observations
        if 'observations' in data:
            observations = data['observations']
            logger.info(f"Observations type: {type(observations).__name__}")
            logger.info(f"Observations count: {len(observations)}")

            if observations:
                first_obs_type = type(observations[0]).__name__
                logger.info(f"First observation type: {first_obs_type}")

                if first_obs_type == 'str':
                    logger.info(f"String observation sample: {observations[0]}")
                elif first_obs_type == 'int' or first_obs_type == 'float':
                    logger.info(f"Numeric observation sample: {observations[0]}")
                elif first_obs_type == 'dict':
                    logger.info(f"Dict observation keys: {list(observations[0].keys())}")

                # Log observation distribution
                if len(observations) > 10:
                    # Sample a few observations from different parts of the array
                    sample_indices = [0, len(observations)//4, len(observations)//2, 3*len(observations)//4, len(observations)-1]
                    for idx in sample_indices:
                        logger.info(f"Observation at index {idx}: {observations[idx]}")

        logger.info("=== END DATASET RESPONSE INSPECTION ===")

    except Exception as e:
        logger.error(f"Error inspecting dataset response: {e}")
        import traceback
        logger.error(traceback.format_exc())

def check_flattened_file(flat_file):
    """
    Check if a flattened CSV file exists and examine its structure

    Args:
        flat_file (str): Path to the flattened CSV file
    """
    try:
        if not os.path.exists(flat_file):
            logger.warning(f"Flattened file not found: {flat_file}")
            return

        # Read the CSV file
        df = pd.read_csv(flat_file)

        logger.info("=== FLATTENED FILE INSPECTION ===")
        logger.info(f"Flattened file: {flat_file}")
        logger.info(f"File size: {os.path.getsize(flat_file)} bytes")
        logger.info(f"Number of rows: {len(df)}")
        logger.info(f"Number of columns: {len(df.columns)}")
        logger.info(f"Columns: {list(df.columns)}")

        # Check if we have more than just 'value' column
        if len(df.columns) <= 1:
            logger.warning("Only found a single column in the flattened file!")
            logger.warning(f"Column: {df.columns[0]}")

            # Check first few values
            logger.info(f"First 5 values: {df.iloc[:5, 0].tolist()}")
        else:
            # Print sample of data
            logger.info("First 2 rows of data:")
            logger.info(df.head(2))

            # Check for dimension columns
            dim_cols = [col for col in df.columns if not col.endswith('_code') and col != 'observation']
            logger.info(f"Found {len(dim_cols)} dimension columns: {dim_cols}")

            # Check for code columns
            code_cols = [col for col in df.columns if col.endswith('_code')]
            logger.info(f"Found {len(code_cols)} code columns: {code_cols}")

            # Check observation column
            if 'observation' in df.columns:
                logger.info(f"Observation column data types: {df['observation'].dtype}")
                logger.info(f"Observation min: {df['observation'].min()}, max: {df['observation'].max()}")

        logger.info("=== END FLATTENED FILE INSPECTION ===")

    except Exception as e:
        logger.error(f"Error checking flattened file: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Retrieve and process census data from ONS API')
    parser.add_argument('--dataset', type=str, default='TS030',
                        help='Dataset ID to retrieve (e.g., TS030, RM097)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--geo-level', type=str, default=None,
                        help='Geographic level to retrieve (e.g., ctry, rgn, la, msoa, lsoa, oa). If not specified, all levels will be processed.')
    args = parser.parse_args()

    # Enable debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("ons_client").setLevel(logging.DEBUG)

    # Set up output directory
    dataset_id = args.dataset
    geo_level = args.geo_level
    base_dir = f"data/{dataset_id}"
    os.makedirs(base_dir, exist_ok=True)

    logger.info(f"Processing dataset {dataset_id}")

    if geo_level:
        # Process a single geographic level
        logger.info(f"Processing single geographic level: {geo_level}")

        # Define output file path
        output_file = os.path.join(base_dir, f"{dataset_id}_{geo_level}.csv")

        # Download all areas for this level using batch processing
        logger.info(f"Downloading all areas for level {geo_level}")

        # Get the optimized batch size for this geographic level
        batch_size = BATCH_SIZES.get(geo_level, 50)
        logger.info(f"Using batch size of {batch_size} for {geo_level}")

        result = download_all_areas_for_level(
            dataset_id=dataset_id,
            geo_level=geo_level,
            batch_size=batch_size,
            output_dir=base_dir,
            output_file=output_file
        )

        if result:
            logger.info(f"Successfully downloaded all areas for level {geo_level}")

            # Verify file exists
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                logger.info(f"Output CSV size: {file_size} bytes")

                # Check file content if it's valid
                if file_size > 0:
                    try:
                        df = pd.read_csv(output_file)
                        logger.info(f"CSV has {len(df)} rows and {len(df.columns)} columns")
                        logger.info(f"Columns: {list(df.columns)}")

                        # Sample a few rows if available
                        if len(df) > 0:
                            sample_size = min(5, len(df))
                            logger.info(f"Sample of {sample_size} rows:")
                            logger.info(df.head(sample_size))
                    except Exception as e:
                        logger.error(f"Error reading CSV: {e}")
            else:
                logger.warning(f"Output file not created: {output_file}")
        else:
            logger.error(f"Failed to download data for level {geo_level}")
    else:
        # Process all geographic levels
        logger.info("No specific geographic level provided. Processing all levels.")

        results = {}
        skipped_levels = []

        # Process each geographic level defined in BATCH_SIZES
        for level in BATCH_SIZES.keys():
            logger.info(f"Processing geographic level: {level}")

            try:
                # Define output file path
                output_file = os.path.join(base_dir, f"{dataset_id}_{level}.csv")

                # Get the optimized batch size for this geographic level
                batch_size = BATCH_SIZES.get(level, 50)
                logger.info(f"Using batch size of {batch_size} for {level}")

                # Download all areas for this level
                logger.info(f"Downloading all areas for level {level}")
                result = download_all_areas_for_level(
                    dataset_id=dataset_id,
                    geo_level=level,
                    batch_size=batch_size,
                    output_dir=base_dir,
                    output_file=output_file
                )

                if result:
                    results[level] = output_file
                    logger.info(f"Successfully downloaded all areas for level {level}")

                    # Verify file exists and log basic stats
                    if os.path.exists(output_file):
                        file_size = os.path.getsize(output_file)
                        logger.info(f"  - CSV file size: {file_size} bytes")

                        # Check file content if it's valid
                        if file_size > 0:
                            try:
                                df = pd.read_csv(output_file)
                                logger.info(f"  - CSV has {len(df)} rows and {len(df.columns)} columns")
                                logger.info(f"  - Columns: {list(df.columns)}")
                            except Exception as e:
                                logger.error(f"Error reading CSV: {e}")
                else:
                    logger.error(f"Failed to download data for level {level}")
                    skipped_levels.append(level)

                # Add a brief pause between levels to avoid rate limiting
                if level != list(BATCH_SIZES.keys())[-1]:  # If not the last level
                    pause_time = 10  # 10 seconds
                    logger.info(f"Pausing for {pause_time} seconds before next level...")
                    time.sleep(pause_time)

            except Exception as e:
                logger.error(f"Error processing level {level}: {str(e)}")
                logger.info("Continuing with next level...")
                skipped_levels.append(level)

        # Summary report
        logger.info(f"=== DOWNLOAD SUMMARY ===")
        logger.info(f"Successfully processed {len(results)} geographic levels")
        for level, file_path in results.items():
            logger.info(f"✅ {level}: {file_path}")

        if skipped_levels:
            logger.info(f"❌ Skipped {len(skipped_levels)} levels due to errors: {', '.join(skipped_levels)}")

        logger.info(f"Completed processing all geographic levels for dataset {dataset_id}")
