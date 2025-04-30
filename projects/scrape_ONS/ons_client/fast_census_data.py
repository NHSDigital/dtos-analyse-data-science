#!/usr/bin/env python3
"""
Fast Census Data Retrieval Utility

This module provides functions to quickly retrieve ONS census data
without using filter jobs, which can be slow and unreliable.

It provides helpers to download data at various geographic levels
using the census-observations endpoint, which returns data immediately.
"""

import os
import csv
import json
import logging
from typing import List, Dict, Any, Optional, Union
import time
import requests
import pandas as pd
from urllib.parse import urljoin

from ons_client import ONSApiClient
from .models import RMDatasetResponse, RMObservation, RMDimensionInfo, RMDimensionItem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dictionary of geographic levels that work with census-observations
# Based on our testing, these are the levels that work reliably
WORKING_GEO_LEVELS = {
    "ctry": "Country",          # England, Wales, etc.
    "rgn": "Region",            # North East, London, etc.
    "la": "Local Authority",    # Local Authority Districts
    "msoa": "MSOA",             # Middle Super Output Area
    "lsoa": "LSOA",             # Lower Super Output Area
    "oa": "Output Area"         # Output Area (smallest geographic level)
}

# These are the reliable dimensions to use with census-observations
RELIABLE_DIMENSIONS = [
    "health_in_general",
    "highest_qualification"
]


def get_fast_census_data(dataset_id: str, geo_level: str, area_codes: List[str],
                        dimensions: Optional[List[str]] = None,
                        output_dir: str = "data/census_data",
                        save_files: bool = True,
                        output_file: Optional[str] = None,
                        json_output: Optional[str] = None,
                        log_level: int = logging.INFO) -> Dict[str, Any]:
    """
    Quickly retrieve census data for the specified geographic level and areas.

    This function uses the census-observations endpoint which returns data
    immediately without the need for filter jobs. It works reliably for certain
    geographic levels (country, region, MSOA, LSOA) and dimension combinations.

    Args:
        dataset_id (str): The dataset ID (e.g., "TS003"). Used for file naming.
        geo_level (str): The geographic level (ctry, rgn, msoa, lsoa).
        area_codes (List[str]): List of area codes to retrieve data for.
        dimensions (Optional[List[str]]): Dimensions to include. If None, uses
                                         reliable default dimensions.
        output_dir (str): Directory to save output files to.
        save_files (bool): Whether to save JSON and CSV files of the data.
        output_file (Optional[str]): Path to save CSV output. If None, generated automatically.
        json_output (Optional[str]): Path to save JSON output. If None, generated automatically.
        log_level (int): Logging level to use.

    Returns:
        Dict[str, Any]: The full response data including observations.

    Raises:
        ValueError: If an unsupported geographic level is specified.
        Exception: If the API call fails.
    """
    # Set the logging level
    logger.setLevel(log_level)

    # Validate geographic level
    if geo_level not in WORKING_GEO_LEVELS:
        supported = ", ".join(WORKING_GEO_LEVELS.keys())
        raise ValueError(f"Geographic level '{geo_level}' not supported. Use one of: {supported}")

    # Use default dimensions if none provided
    if dimensions is None:
        dimensions = RELIABLE_DIMENSIONS
        logger.info(f"Using default dimensions: {dimensions}")

    # Initialize the ONS client
    client = ONSApiClient()

    # Make the API call to get the data
    logger.info(f"Retrieving data for {len(area_codes)} areas at {geo_level} level...")
    try:
        data = client.get_dataset_observations_by_area_type(
            dataset_id=dataset_id,
            edition="2021",
            version="1",
            area_type=geo_level,
            area_codes=area_codes
        )
    except Exception as e:
        logger.error(f"Error retrieving data: {e}")
        raise

    # Check if we got observations
    obs_count = len(data.get('observations', []))
    logger.info(f"Retrieved {obs_count} observations")

    # Save files if requested
    if save_files and obs_count > 0:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create filename base or use provided paths
        if json_output is None:
            base_filename = f"{dataset_id}_{geo_level}_{len(area_codes)}_areas"
            json_file = os.path.join(output_dir, f"{base_filename}.json")
        else:
            json_file = json_output

        if output_file is None:
            base_filename = f"{dataset_id}_{geo_level}_{len(area_codes)}_areas"
            csv_file = os.path.join(output_dir, f"{base_filename}.csv")
        else:
            csv_file = output_file

        # Save JSON only if json_output is provided
        if json_output is not None:
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved JSON data to {json_file}")

        # Determine if this is a TS (Topic Summary) or RM (Multivariate) dataset
        is_rm_dataset = dataset_id.startswith("RM")
        logger.info(f"Dataset {dataset_id} identified as {'RM (Multivariate)' if is_rm_dataset else 'TS (Topic Summary)'}")

        # Save CSV with all dimensions for both TS and RM datasets
        # The flat file with all dimensions becomes the main output file
        transform_rm_dataset_to_flat(data, csv_file)

        # Also save debug.json to help with troubleshooting only if JSON output is enabled
        if json_output is not None:
            debug_file = f"{csv_file}.debug.json"
            with open(debug_file, 'w') as f:
                json.dump(data, f, indent=2)

    return data


def get_areas_for_level(geo_level: str, population_type: str = "UR") -> List[Dict[str, str]]:
    """
    Get all available areas for a specified geographic level.

    Args:
        geo_level (str): The geographic level (ctry, rgn, msoa, lsoa).
        population_type (str): The population type (default: "UR").

    Returns:
        List[Dict[str, str]]: List of area dictionaries with id and label.

    Raises:
        ValueError: If an unsupported geographic level is specified.
    """
    # Validate geographic level
    if geo_level not in WORKING_GEO_LEVELS:
        supported = ", ".join(WORKING_GEO_LEVELS.keys())
        raise ValueError(f"Geographic level '{geo_level}' not supported. Use one of: {supported}")

    # Initialize the ONS client
    client = ONSApiClient()

    # Get areas
    logger.info(f"Retrieving areas for {geo_level} level...")
    try:
        areas = client.get_cached_areas(population_type, geo_level)
        logger.info(f"Found {len(areas)} areas for {geo_level} level")
        return [{"id": area.id, "label": area.label} for area in areas]
    except Exception as e:
        logger.error(f"Error retrieving areas: {e}")
        raise


def observations_to_csv(data: Dict[str, Any], output_file: str) -> None:
    """
    Convert observations data to CSV format.

    Args:
        data (Dict[str, Any]): The observations data.
        output_file (str): The output CSV file path.
    """
    try:
        if 'observations' not in data or not data['observations']:
            logger.warning("No observations to convert to CSV")
            return

        observations = data['observations']
        logger.info(f"Converting {len(observations)} observations to CSV")

        # DEBUG: Print out observation structure for debugging
        if observations:
            if isinstance(observations[0], dict):
                logger.info(f"First observation keys: {list(observations[0].keys())}")
                for key in observations[0].keys():
                    logger.info(f"  Key: {key}, Value type: {type(observations[0][key]).__name__}, Sample: {str(observations[0][key])[:100]}")
            else:
                logger.info(f"Observations are simple values, type: {type(observations[0]).__name__}, sample: {observations[0]}")

        # Open the CSV file for writing
        with open(output_file, 'w', newline='') as csvfile:
            # Check if observations are simple values (strings, numbers)
            if observations and not isinstance(observations[0], dict):
                logger.info("Observations are simple values, looking for headers")
                headers = data.get('headers', [])

                if headers:
                    logger.info(f"Found headers: {headers}")
                    # Write a simple CSV with headers and values
                    writer = csv.writer(csvfile)
                    writer.writerow(headers)

                    # If observations are a flat list, write them in rows according to header length
                    if len(headers) > 0:
                        for i in range(0, len(observations), len(headers)):
                            if i + len(headers) <= len(observations):
                                row = observations[i:i+len(headers)]
                                writer.writerow(row)
                    else:
                        # If no headers length, just write each observation as a row
                        for value in observations:
                            writer.writerow([value])
                else:
                    logger.warning("No headers found, writing simple value column")
                    writer = csv.writer(csvfile)
                    writer.writerow(['value'])
                    for value in observations:
                        writer.writerow([value])
            # Create a structured CSV for dictionary observations
            elif isinstance(observations[0], dict) and 'dimensions' in observations[0]:
                logger.info("Processing observations with 'dimensions' field")
                # Extract all potential dimension IDs for columns
                dimension_ids = set()
                for obs in observations:
                    if isinstance(obs.get('dimensions'), list):
                        for dim in obs.get('dimensions', []):
                            if isinstance(dim, dict) and 'dimension_id' in dim:
                                dimension_ids.add(dim['dimension_id'])

                logger.info(f"Found dimension IDs: {dimension_ids}")
                # Create fieldnames: one column for each dimension ID + observation value
                fieldnames = sorted(list(dimension_ids)) + ['observation_value']

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # Write each observation as a row
                for obs in observations:
                    row = {'observation_value': obs.get('observation', '')}

                    # Extract dimension values
                    if isinstance(obs.get('dimensions'), list):
                        for dim in obs.get('dimensions', []):
                            if isinstance(dim, dict) and 'dimension_id' in dim and 'option' in dim:
                                row[dim['dimension_id']] = dim.get('option', '')

                    writer.writerow(row)
            else:
                logger.info("Using fallback CSV structure (no dimensions field found)")
                # Fallback to using the original structure with all fields
                if isinstance(observations[0], dict):
                    all_keys = set()
                    for obs in observations:
                        if isinstance(obs, dict):
                            all_keys.update(obs.keys())

                    logger.info(f"Using these CSV columns: {all_keys}")
                    writer = csv.DictWriter(csvfile, fieldnames=sorted(all_keys))
                    writer.writeheader()
                    writer.writerows(observations)
                else:
                    # Final fallback for any other structure
                    writer = csv.writer(csvfile)
                    writer.writerow(['value'])
                    for value in observations:
                        writer.writerow([value])

        # Verify if the file was written successfully
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            logger.info(f"CSV file created: {output_file}, size: {file_size} bytes")
            if file_size == 0:
                logger.warning("WARNING: CSV file is empty!")

    except Exception as e:
        logger.error(f"Error converting to CSV: {e}")
        import traceback
        logger.error(traceback.format_exc())


def get_available_dimensions(population_type: str = "UR") -> List[Dict[str, str]]:
    """
    Get all available dimensions for a population type.

    Args:
        population_type (str): The population type (default: "UR").

    Returns:
        List[Dict[str, str]]: List of dimension dictionaries with id and label.
    """
    # Initialize the ONS client
    client = ONSApiClient()

    # Get dimensions
    logger.info(f"Retrieving dimensions for {population_type}...")
    try:
        dimensions = client.get_dimensions(population_type)
        logger.info(f"Found {len(dimensions)} dimensions")
        return [{"id": dim.id, "label": dim.label} for dim in dimensions]
    except Exception as e:
        logger.error(f"Error retrieving dimensions: {e}")
        raise


def download_multiple_levels(dataset_id: str, area_sets: Dict[str, List[str]],
                           dimensions: Optional[List[str]] = None,
                           output_dir: str = "data/census_data") -> Dict[str, Dict[str, Any]]:
    """
    Download data for multiple geographic levels in one function call.

    Args:
        dataset_id (str): The dataset ID (used for file naming).
        area_sets (Dict[str, List[str]]): Dictionary mapping geo level to list of area codes.
        dimensions (Optional[List[str]]): Dimensions to include.
        output_dir (str): Directory to save output files to.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping geo level to response data.
    """
    results = {}

    for geo_level, area_codes in area_sets.items():
        if geo_level not in WORKING_GEO_LEVELS:
            logger.warning(f"Skipping unsupported level: {geo_level}")
            continue

        try:
            logger.info(f"Processing {geo_level} level with {len(area_codes)} areas...")
            data = get_fast_census_data(
                dataset_id=dataset_id,
                geo_level=geo_level,
                area_codes=area_codes,
                dimensions=dimensions,
                output_dir=output_dir
            )
            results[geo_level] = data
            logger.info(f"Successfully retrieved data for {geo_level}")

            # Brief pause to avoid rate limiting
            time.sleep(1)

        except Exception as e:
            logger.error(f"Error retrieving data for {geo_level}: {e}")

    return results


def download_all_areas_for_level(
        dataset_id: str,
        geo_level: str,
        population_type: str = "UR",
        dimensions: Optional[List[str]] = None,
        edition: str = "2021",
        version: str = "1",
        output_dir: Optional[str] = None,
        output_file: Optional[str] = None,
        batch_size: int = 50
    ) -> str:
    """
    Download data for ALL areas of a specific geographic level.

    This function retrieves all areas for the specified geographic level and downloads
    data for all of them using the appropriate endpoint based on the dataset type.
    Data for all areas is combined into a single CSV file.

    Args:
        dataset_id (str): The dataset ID (e.g., "TS003" or "RM097").
        geo_level (str): The geographic level (ctry, rgn, msoa, lsoa).
        population_type (str): The population type (default: "UR").
        dimensions (Optional[List[str]]): Dimensions to include if using census-observations.
        edition (str): The dataset edition. Defaults to "2021".
        version (int): The dataset version. Defaults to 1.
        output_dir (Optional[str]): Directory to save output file. If None, uses "data/{dataset_id}".
        output_file (Optional[str]): Specific path to save the output CSV. If provided, overrides the default.
        batch_size (int): Number of areas to process in each API call. Defaults to 50.

    Returns:
        str: Path to the combined CSV file with data for all areas.
    """
    # Validate geographic level
    if geo_level not in WORKING_GEO_LEVELS:
        supported = ", ".join(WORKING_GEO_LEVELS.keys())
        raise ValueError(f"Geographic level '{geo_level}' not supported. Use one of: {supported}")

    # Set output directory
    if output_dir is None:
        output_dir = f"data/{dataset_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Define output file path
    if output_file is None:
        output_file = os.path.join(output_dir, f"{dataset_id}_{geo_level}_all.csv")

    # Create a temporary directory for batch files
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Get all areas for this level
    client = ONSApiClient()
    areas = client.get_cached_areas(population_type, geo_level)

    if not areas:
        logger.warning(f"No areas found for {geo_level} level")
        return None

    # Extract area codes
    area_codes = [area.id for area in areas]
    logger.info(f"Found {len(area_codes)} areas for {geo_level} level")

    # Process in batches to avoid URL length limits and rate limits
    area_batches = [area_codes[i:i+batch_size] for i in range(0, len(area_codes), batch_size)]
    logger.info(f"Processing {len(area_batches)} batches of up to {batch_size} areas each")

    # Determine if this is a TS (Topic Summary) or RM (Multivariate) dataset
    is_rm_dataset = dataset_id.startswith("RM")
    logger.info(f"Dataset {dataset_id} identified as {'RM (Multivariate)' if is_rm_dataset else 'TS (Topic Summary)'}")

    # If it's an RM dataset, we'll combine all batch data and output a single CSV at the end
    all_data = {}
    all_observations = []
    dimension_info = {}
    header_written = False  # Track if header has been written for append mode

    # Process each batch
    for i, batch in enumerate(area_batches):
        logger.info(f"Processing batch {i+1}/{len(area_batches)} with {len(batch)} areas")

        try:
            # Get data for this batch
            data = get_dataset_data(
                dataset_id=dataset_id,
                geo_level=geo_level,
                area_codes=batch,
                edition=edition,
                version=str(version),
                output_dir=temp_dir,
                save_files=False  # Don't save individual files
            )

            # Store dimension info from the first batch (should be the same for all)
            if not dimension_info and 'dimensions' in data:
                dimension_info = data['dimensions']
                logger.info(f"Stored dimension info from batch {i+1}")

            # Accumulate observations
            if 'observations' in data and data['observations']:
                all_observations.extend(data['observations'])
                batch_observations_count = len(data['observations'])
                logger.info(f"Added {batch_observations_count} observations from batch {i+1}")

                # For each batch, process and append to the main CSV file
                batch_file = os.path.join(temp_dir, f"batch_{i+1}.csv")

                # Process batch data
                batch_data = {'observations': data['observations']}
                if 'dimensions' in data:
                    batch_data['dimensions'] = data['dimensions']

                # Transform to CSV format with proper dimensions
                transform_rm_dataset_to_batch_csv(batch_data, batch_file)

                # Append this batch to the main output file
                if os.path.exists(batch_file):
                    append_batch_to_output(batch_file, output_file, header_written)
                    header_written = True  # After first batch, header is written
                    logger.info(f"Appended batch {i+1} data to {output_file}")

            # Brief pause to avoid rate limiting
            if i < len(area_batches) - 1:  # If not the last batch
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error processing batch {i+1}: {str(e)}")
            logger.info("Continuing with next batch...")

    # Check if we've accumulated any observations
    total_observations = len(all_observations)
    logger.info(f"Total observations collected: {total_observations}")

    if total_observations > 0:
        logger.info(f"Successfully downloaded data for {geo_level} level to {output_file}")
        return output_file
    else:
        logger.warning(f"No observations collected for {geo_level} level")
        return None


def download_all_geographic_levels(
        dataset_id: str,
        population_type: str = "UR",
        dimensions: Optional[List[str]] = None,
        edition: str = "2021",
        version: int = 1,
        output_dir: Optional[str] = None,
        batch_size: int = 50
    ) -> Dict[str, str]:
    """
    Download data for ALL supported geographic levels.

    This function downloads data for all areas within each supported geographic level
    (country, region, MSOA, LSOA). It's the equivalent of download_all_area_types but
    uses the fast census-observations endpoint instead of filters.

    Args:
        dataset_id (str): The dataset ID (e.g., "TS003").
        population_type (str): The population type (default: "UR").
        dimensions (Optional[List[str]]): Dimensions to include. If None, uses defaults.
        edition (str): The dataset edition. Defaults to "2021".
        version (int): The dataset version. Defaults to 1.
        output_dir (Optional[str]): Directory to save output files. If None, uses "data/{dataset_id}".
        batch_size (int): Number of areas to process in each API call. Defaults to 50.

    Returns:
        Dict[str, str]: Dictionary mapping geographic level to the path of the downloaded CSV file.
    """
    # Set output directory
    if output_dir is None:
        output_dir = f"data/{dataset_id}"
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Process each supported geographic level
    for geo_level in WORKING_GEO_LEVELS.keys():
        logger.info(f"Processing {geo_level} level ({WORKING_GEO_LEVELS[geo_level]})")

        try:
            # Download data for all areas at this level
            output_file = download_all_areas_for_level(
                dataset_id=dataset_id,
                geo_level=geo_level,
                population_type=population_type,
                dimensions=dimensions,
                edition=edition,
                version=version,
                output_dir=output_dir,
                batch_size=batch_size
            )

            if output_file:
                results[geo_level] = output_file
                logger.info(f"Successfully downloaded data for {geo_level}")
            else:
                logger.warning(f"No data downloaded for {geo_level}")

            # Brief pause between levels to avoid rate limiting
            time.sleep(2)

        except Exception as e:
            logger.error(f"Error processing {geo_level} level: {e}")

    return results


def create_filter_job(
    dataset_id: str,
    geo_level: str,
    population_type: str = "UR",
    dimensions: Optional[List[str]] = None,
    edition: str = "2021",
    version: int = 1,
    output_dir: Optional[str] = None
) -> str:
    """
    Create a filter job to download an entire geographic level dataset at once.

    This is much more efficient for large area types (LSOA, OA) than making
    thousands of API calls. It creates a filter job and downloads the resulting CSV.

    Args:
        dataset_id: The dataset ID (e.g., "TS003").
        geo_level: The geographic level (ctry, rgn, msoa, lsoa, oa).
        population_type: The population type. Defaults to "UR" (usual residents).
        dimensions: Additional dimensions to include. If None, uses default dimensions.
        edition: The dataset edition. Defaults to "2021".
        version: The dataset version. Defaults to 1.
        output_dir: Directory to save output file. If None, uses "data/{dataset_id}".

    Returns:
        str: Path to the downloaded CSV file.
    """
    # Set output directory
    if output_dir is None:
        output_dir = f"data/{dataset_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Create a descriptive filename for the output
    output_file = os.path.join(output_dir, f"{dataset_id}_{geo_level}_all_filter.csv")

    # Initialize client
    client = ONSApiClient()

    # Use default dimensions if none specified
    if dimensions is None:
        dimensions = RELIABLE_DIMENSIONS
        logger.info(f"Using default dimensions: {dimensions}")

    logger.info(f"Creating filter job for {geo_level} level...")

    # Create filter request payload
    filter_payload = {
        "dataset": {
            "id": dataset_id,
            "edition": edition,
            "version": version
        },
        "population_type": population_type,
        "dimensions": [
            {
                "name": geo_level,
                "is_area_type": True
            }
        ]
    }

    # Add any additional dimensions
    for dim in dimensions:
        filter_payload["dimensions"].append({"name": dim})

    # Create the filter
    try:
        # POST to create filter
        filter_response = client.session.post(
            f"{client.base_url}/filters",
            json=filter_payload
        )
        filter_response.raise_for_status()

        filter_data = filter_response.json()
        filter_id = filter_data.get("filter_id")

        if not filter_id:
            logger.error("No filter_id in response")
            return None

        logger.info(f"Created filter with ID: {filter_id}")

        # Submit the filter job for processing
        submit_response = client.session.post(
            f"{client.base_url}/filters/{filter_id}/submit"
        )
        submit_response.raise_for_status()

        filter_output_data = submit_response.json()
        filter_output_id = filter_output_data.get("filter_output_id")

        if not filter_output_id:
            logger.error("No filter_output_id in response")
            return None

        logger.info(f"Submitted filter, filter_output_id: {filter_output_id}")

        # Poll until the filter job is complete
        max_attempts = 30
        attempt = 0
        complete = False

        while attempt < max_attempts and not complete:
            time.sleep(5)  # Wait 5 seconds between checks
            attempt += 1

            status_response = client.session.get(
                f"{client.base_url}/filter-outputs/{filter_output_id}"
            )
            status_response.raise_for_status()

            status_data = status_response.json()
            state = status_data.get("state")

            logger.info(f"Filter job state: {state} (attempt {attempt}/{max_attempts})")

            if state == "completed":
                complete = True
                break

        if not complete:
            logger.error(f"Filter job did not complete within {max_attempts} attempts")
            return None

        # Get the download URL for the CSV file
        download_url = None
        for download in status_data.get("downloads", {}).values():
            if download.get("size"):  # Only get non-empty downloads
                download_url = download.get("href")
                if download_url and download_url.endswith(".csv"):
                    break

        if not download_url:
            logger.error("No CSV download URL found in filter output")
            return None

        # Download the CSV file
        logger.info(f"Downloading CSV from {download_url}")
        download_response = client.session.get(download_url)
        download_response.raise_for_status()

        with open(output_file, "wb") as f:
            f.write(download_response.content)

        logger.info(f"Downloaded CSV to {output_file}")
        return output_file

    except Exception as e:
        logger.error(f"Error in filter process: {e}")
        return None


def get_dataset_data(dataset_id: str, geo_level: str, area_codes: List[str],
                 edition: str = "2021", version: str = "1",
                 output_dir: str = "data/census_data",
                 save_files: bool = True) -> Dict[str, Any]:
    """
    Get dataset data directly using the dataset endpoint which works reliably for both TS and RM datasets.

    This function uses the datasets/{datasetId}/editions/{edition}/versions/{version}/json endpoint
    which returns data immediately and works for both Topic Summaries (TS) and Multivariate Datasets (RM).

    Args:
        dataset_id (str): The dataset ID (e.g., "TS008" or "RM097").
        geo_level (str): The geographic level (ctry, rgn, la, msoa, lsoa, oa).
        area_codes (List[str]): List of area codes to retrieve data for.
        edition (str): The dataset edition. Defaults to "2021".
        version (str): The dataset version. Defaults to "1".
        output_dir (str): Directory to save output files to.
        save_files (bool): Whether to save JSON and CSV files of the data.

    Returns:
        Dict[str, Any]: The full response data including observations.

    Raises:
        ValueError: If an unsupported geographic level is specified.
        Exception: If the API call fails.
    """
    # Validate geographic level
    if geo_level not in WORKING_GEO_LEVELS:
        supported = ", ".join(WORKING_GEO_LEVELS.keys())
        raise ValueError(f"Geographic level '{geo_level}' not supported. Use one of: {supported}")

    # Initialize the ONS client
    client = ONSApiClient()

    # Detect if this is an RM dataset based on ID prefix
    is_rm_dataset = dataset_id.startswith("RM")
    logger.info(f"Dataset {dataset_id} identified as {'RM (Multivariate)' if is_rm_dataset else 'TS (Topic Summary)'}")

    # Make the API call to get the data
    logger.info(f"Retrieving data for {len(area_codes)} areas at {geo_level} level using dataset endpoint...")
    try:
        data = client.get_dataset_observations_by_area_type(
            dataset_id=dataset_id,
            edition=edition,
            version=version,
            area_type=geo_level,
            area_codes=area_codes
        )

        # DEBUG: Print out the structure of the response
        logger.info(f"Response keys: {list(data.keys())}")
        if 'observations' in data and data['observations']:
            logger.info(f"Number of observations: {len(data['observations'])}")
            logger.info(f"Sample observation keys: {list(data['observations'][0].keys()) if isinstance(data['observations'][0], dict) else 'Not a dict'}")
            logger.info(f"Sample observation: {json.dumps(data['observations'][0], indent=2)[:500]}...")
        else:
            logger.warning(f"No observations found in response. Full response: {json.dumps(data, indent=2)[:500]}...")

    except Exception as e:
        logger.error(f"Error retrieving data: {e}")
        raise

    # Check if we got observations
    obs_count = len(data.get('observations', []))
    logger.info(f"Retrieved {obs_count} observations")

    # Save files if requested
    if save_files and obs_count > 0:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create filename base
        base_filename = f"{dataset_id}_{geo_level}_{len(area_codes)}_areas"

        # Save JSON
        json_file = os.path.join(output_dir, f"{base_filename}.json")
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved JSON data to {json_file}")

        # Save CSV - use appropriate handler based on dataset type
        csv_file = os.path.join(output_dir, f"{base_filename}.csv")
        if is_rm_dataset:
            # Use the RM dataset specific handler for better CSV generation
            handle_rm_dataset_response(data, csv_file)
        else:
            # Use the standard observations_to_csv function for TS datasets
            observations_to_csv(data, csv_file)

            # For TS datasets, also save the debug.json file to help with flattening
            debug_file = f"{csv_file}.debug.json"
            with open(debug_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved TS dataset debug data to {debug_file} for better flattening")

        logger.info(f"Saved CSV data to {csv_file}")

    return data


def transform_rm_dataset_to_flat(data: Dict[str, Any], output_file: str) -> None:
    """
    Transform RM dataset responses into a flat CSV format with all dimensions and codes.

    This function handles the RM dataset format where observations are a simple array of values
    and dimensions are provided separately. It creates a proper tabular structure similar to
    the TS dataset flat CSVs.

    Args:
        data (Dict[str, Any]): The RM dataset response data with observations and dimensions
        output_file (str): Path to save the flat CSV file
    """
    try:
        logger.info("=== TRANSFORM_RM_DATASET_TO_FLAT DEBUGGING ===")

        if not data.get('observations'):
            logger.warning("No observations to transform")
            return

        if not data.get('dimensions'):
            logger.warning("No dimensions found in the data")
            return

        observations = data['observations']
        dimensions = data['dimensions']

        logger.info(f"Transforming dataset with {len(observations)} observations and processing dimensions")
        logger.info(f"Dimensions type: {type(dimensions).__name__}")
        if isinstance(dimensions, list):
            logger.info(f"Found {len(dimensions)} dimensions")

        # Print sample of first dimension
        if isinstance(dimensions, list) and dimensions:
            dim = dimensions[0]
            logger.info(f"First dimension type: {type(dim).__name__}")
            if isinstance(dim, dict):
                logger.info(f"First dimension keys: {list(dim.keys())}")
                # Look for dimension name field
                dim_name = dim.get('dimension_name', dim.get('name', 'unknown'))
                logger.info(f"First dimension name: {dim_name}")

                # Check for options structure
                if 'options' in dim:
                    options = dim['options']
                    logger.info(f"First dimension has {len(options)} options")
                    if options:
                        logger.info(f"First option type: {type(options[0]).__name__}")
                        if isinstance(options[0], dict):
                            logger.info(f"First option keys: {list(options[0].keys())}")

        # Extract dimension information
        dim_info = []
        for dim in dimensions:
            if isinstance(dim, dict):
                # For debugging, print dimension structure
                logger.info(f"Processing dimension: {json.dumps(dim)[:200]}...")

                # Look for dimension name field - support multiple possible keys
                dim_name = None
                if 'dimension_name' in dim:
                    dim_name = dim['dimension_name']
                elif 'name' in dim:
                    dim_name = dim['name']

                # Look for options field
                options = []
                if 'options' in dim:
                    raw_options = dim['options']

                    # Log options structure
                    logger.info(f"Found options array with {len(raw_options)} items")
                    if raw_options:
                        first_option = raw_options[0]
                        logger.info(f"Option type: {type(first_option).__name__}")
                        if isinstance(first_option, dict):
                            logger.info(f"Option keys: {list(first_option.keys())}")

                    options = raw_options

                    # If we have a dimension name and options, add to dim_info
                    if dim_name:
                        dim_info.append({
                            'name': dim_name,
                            'options': options
                        })
                        logger.info(f"Added dimension '{dim_name}' with {len(options)} options to dim_info")

        if not dim_info:
            logger.warning("Could not extract dimension information")
            logger.warning("Raw dimensions data: " + json.dumps(dimensions)[:500])
            return

        logger.info(f"Successfully extracted {len(dim_info)} dimensions: {[d['name'] for d in dim_info]}")

        # Calculate the shape of the multidimensional data
        shape = [len(dim['options']) for dim in dim_info]
        logger.info(f"Data shape: {shape}")

        # Generate all dimension combinations
        indices = [0] * len(dim_info)
        rows = []

        # Calculate the total number of combinations (should match observations length)
        total_combinations = 1
        for s in shape:
            total_combinations *= s

        if total_combinations != len(observations):
            logger.warning(f"Observation count ({len(observations)}) doesn't match expected combinations ({total_combinations})")
            logger.warning(f"Will use available observations up to the maximum index")

        # For each observation, create a row with all dimension values
        obs_index = 0
        while obs_index < len(observations):
            # Stop if we've processed all expected combinations
            if obs_index >= total_combinations:
                break

            row = {}

            # Add dimension values and codes
            for i, dim in enumerate(dim_info):
                opt_index = indices[i]
                if opt_index < len(dim['options']):
                    option = dim['options'][opt_index]
                    dim_name = dim['name']

                    # Handle different option structures
                    label = ""
                    code = ""

                    if isinstance(option, dict):
                        # Log option keys for debugging
                        logger.debug(f"Option keys: {list(option.keys())}")

                        # Look for label field with fallbacks
                        if 'label' in option:
                            label = option['label']
                        elif 'name' in option:
                            label = option['name']

                        # Look for ID field with fallbacks
                        if 'id' in option:
                            code = option['id']
                        elif 'option' in option:
                            code = option['option']
                    else:
                        # If option is not a dict, use the raw value
                        label = str(option)
                        code = str(option)

                    row[dim_name] = label
                    row[f"{dim_name}_code"] = code

            # Add the observation value
            row['observation'] = observations[obs_index]
            rows.append(row)

            # Increment indices (like counting with carry)
            for i in range(len(indices) - 1, -1, -1):
                indices[i] += 1
                if indices[i] < shape[i]:
                    break
                indices[i] = 0

            obs_index += 1

        # Generate field names preserving order
        fieldnames = []
        for dim in dim_info:
            fieldnames.append(dim['name'])
            fieldnames.append(f"{dim['name']}_code")
        fieldnames.append('observation')

        logger.info(f"Generated {len(rows)} rows with {len(fieldnames)} columns")

        # Log first few rows for debugging
        if rows:
            logger.info(f"First row: {rows[0]}")

        # Write to CSV
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Successfully wrote {len(rows)} rows to {output_file} with dimensions: {fieldnames}")
        logger.info(f"Saved full dimensional data to CSV file: {output_file}")
        logger.info("=== END TRANSFORM_RM_DATASET_TO_FLAT ===")

    except Exception as e:
        logger.error(f"Error transforming RM dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())


def handle_rm_dataset_response(data: Dict[str, Any], output_file: str) -> None:
    """
    Handle the specific structure of RM (Multivariate) dataset responses.

    The ONS API response structure differs between TS and RM datasets.
    This function handles both RM and TS dataset responses using Pydantic models.

    Args:
        data (Dict[str, Any]): The observations data from the ONS API.
        output_file (str): The output CSV file path.
    """
    try:
        logger.info("=== HANDLE_RM_DATASET_RESPONSE DEBUGGING ===")

        # Save raw response for debugging
        debug_file = f"{output_file}.debug.json"
        with open(debug_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved raw response to {debug_file} for debugging")

        # Create a flat file path - replace .csv with _flat.csv
        flat_file = os.path.splitext(output_file)[0] + "_flat.csv"

        # Log the basic structure of the input data
        logger.info(f"Input data keys: {list(data.keys())}")
        logger.info(f"Has 'observations': {'observations' in data}")
        logger.info(f"Has 'dimensions': {'dimensions' in data}")

        if 'observations' in data:
            observations = data['observations']
            logger.info(f"Number of observations: {len(observations)}")

            if observations:
                logger.info(f"First observation type: {type(observations[0]).__name__}")
                if not isinstance(observations[0], dict):
                    logger.info(f"First few observations: {observations[:5]}")

        if 'dimensions' in data:
            dimensions = data['dimensions']
            logger.info(f"Dimensions type: {type(dimensions).__name__}")

            if isinstance(dimensions, list):
                logger.info(f"Number of dimensions: {len(dimensions)}")

                if dimensions:
                    logger.info(f"First dimension type: {type(dimensions[0]).__name__}")
                    if isinstance(dimensions[0], dict):
                        # Print dimension keys
                        logger.info(f"First dimension keys: {list(dimensions[0].keys())}")

                        # Print dimension name field
                        dim_name = dimensions[0].get('dimension_name', dimensions[0].get('name', 'unknown'))
                        logger.info(f"First dimension name: {dim_name}")

                        # Check for options
                        if 'options' in dimensions[0]:
                            options = dimensions[0]['options']
                            logger.info(f"First dimension has {len(options)} options")

                            if options:
                                # Print option structure
                                logger.info(f"First option type: {type(options[0]).__name__}")
                                if isinstance(options[0], dict):
                                    logger.info(f"First option keys: {list(options[0].keys())}")
                                    # Print label and ID fields
                                    logger.info(f"Option label: {options[0].get('label', 'N/A')}")
                                    logger.info(f"Option ID: {options[0].get('id', 'N/A')}")

        # Try to parse the response as either TS or RM dataset
        is_ts_dataset = False
        is_rm_dataset = False

        # Check if this is a TS dataset (observations are simple values, not dictionaries)
        if (isinstance(data.get('observations', []), list) and
            data.get('observations') and
            not isinstance(data['observations'][0], dict) and
            isinstance(data.get('dimensions'), list)):

            logger.info("Dataset matches TS structure (simple observations array with dimensions)")
            is_ts_dataset = True

            # Create simple CSV with just observations
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['value'])
                for value in data['observations']:
                    writer.writerow([value])
            logger.info(f"Created simple CSV with {len(data['observations'])} observations at: {output_file}")

            # Create a better flat file with more information
            logger.info("Creating enhanced flat file with dimension information for TS dataset")
            transform_rm_dataset_to_flat(data, flat_file)

            logger.info(f"TS Dataset processing complete. Created both simple CSV and enhanced flat file.")
            return

        # Try to detect RM dataset structure
        if (isinstance(data.get('observations', []), list) and
            isinstance(data.get('dimensions'), list)):

            logger.info("Dataset appears to be an RM dataset")
            is_rm_dataset = True

            # Process simple observations (array of values)
            if data['observations'] and not isinstance(data['observations'][0], dict):
                logger.info("RM dataset has simple value observations")

                # Check for headers
                headers = data.get('headers', [])
                if headers:
                    logger.info(f"Found headers: {headers}")
                    # Create CSV with headers
                    with open(output_file, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(headers)

                        observations = data['observations']
                        # Write observations in rows based on header length
                        if len(headers) > 0:
                            for i in range(0, len(observations), len(headers)):
                                if i + len(headers) <= len(observations):
                                    row = observations[i:i+len(headers)]
                                    writer.writerow(row)
                else:
                    # No headers, write simple value column
                    logger.info("No headers found, writing simple value column")
                    with open(output_file, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['value'])
                        for value in data['observations']:
                            writer.writerow([value])

                # Create enhanced flat file with dimension information
                logger.info("Creating enhanced flat file for RM dataset")
                transform_rm_dataset_to_flat(data, flat_file)
                return

            # Process complex observations (array of dictionaries)
            elif data['observations'] and isinstance(data['observations'][0], dict):
                logger.info("RM dataset has complex dictionary observations")
                # Process observations as dictionaries
                # [existing complex observation processing code]

        # Fallback for other dataset structures
        logger.info("Using fallback dataset processing")

        # Check for observations
        if not data.get('observations'):
            logger.warning("No observations in dataset response")
            return

        observations = data['observations']
        logger.info(f"Processing {len(observations)} observations in fallback mode")

        # Handle simple value observations
        if observations and not isinstance(observations[0], dict):
            logger.info(f"Observations are simple values, type: {type(observations[0]).__name__}")

            headers = data.get('headers', [])
            if headers:
                logger.info(f"Found headers: {headers}")
                # Write CSV with headers and values
                with open(output_file, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(headers)
                    # Write observations according to header length
                    for i in range(0, len(observations), len(headers)):
                        if i + len(headers) <= len(observations):
                            row = observations[i:i+len(headers)]
                            writer.writerow(row)
            else:
                logger.warning("No headers found, writing simple value column")
                with open(output_file, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['value'])
                    for value in observations:
                        writer.writerow([value])

            # Create enhanced flat file with dimension information
            if data.get('dimensions'):
                logger.info("Creating enhanced flat file in fallback mode")
                transform_rm_dataset_to_flat(data, flat_file)
            return

        # Handle complex observations (dictionaries)
        logger.warning("Encountered complex observations, using basic fallback")
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['raw_observation'])
            for obs in observations:
                writer.writerow([json.dumps(obs) if isinstance(obs, dict) else str(obs)])

        # Verify file was created successfully
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            logger.info(f"Dataset CSV created: {output_file}, size: {file_size} bytes")
            if file_size == 0:
                logger.warning("WARNING: Dataset CSV file is empty!")

        logger.info("=== END HANDLE_RM_DATASET_RESPONSE ===")

    except Exception as e:
        logger.error(f"Error processing dataset response: {e}")
        import traceback
        logger.error(traceback.format_exc())


def transform_rm_dataset_to_batch_csv(data: Dict[str, Any], output_file: str) -> None:
    """
    Transform a batch of RM dataset to CSV for appending to the main output file.
    This is a variant of transform_rm_dataset_to_flat optimized for batch processing.

    Args:
        data (Dict[str, Any]): The dataset response data
        output_file (str): Path to save the batch CSV file
    """
    try:
        if not data.get('observations'):
            logger.warning("No observations to transform in batch")
            return

        if not data.get('dimensions'):
            logger.warning("No dimensions found in batch data")
            return

        observations = data['observations']
        dimensions = data['dimensions']

        # Extract dimension information
        dim_info = []
        for dim in dimensions:
            if isinstance(dim, dict):
                # Look for dimension name field - support multiple possible keys
                dim_name = None
                if 'dimension_name' in dim:
                    dim_name = dim['dimension_name']
                elif 'name' in dim:
                    dim_name = dim['name']

                # Look for options field
                options = []
                if 'options' in dim and dim_name:
                    options = dim['options']
                    dim_info.append({
                        'name': dim_name,
                        'options': options
                    })

        if not dim_info:
            logger.warning("Could not extract dimension information for batch")
            return

        # Calculate the shape of the multidimensional data
        shape = [len(dim['options']) for dim in dim_info]

        # Generate all dimension combinations
        indices = [0] * len(dim_info)
        rows = []

        # Calculate the total number of combinations
        total_combinations = 1
        for s in shape:
            total_combinations *= s

        # For each observation, create a row with all dimension values
        obs_index = 0
        while obs_index < len(observations):
            # Stop if we've processed all expected combinations
            if obs_index >= total_combinations:
                break

            row = {}

            # Add dimension values and codes
            for i, dim in enumerate(dim_info):
                opt_index = indices[i]
                if opt_index < len(dim['options']):
                    option = dim['options'][opt_index]
                    dim_name = dim['name']

                    # Handle different option structures
                    label = ""
                    code = ""

                    if isinstance(option, dict):
                        # Look for label field with fallbacks
                        if 'label' in option:
                            label = option['label']
                        elif 'name' in option:
                            label = option['name']

                        # Look for ID field with fallbacks
                        if 'id' in option:
                            code = option['id']
                        elif 'option' in option:
                            code = option['option']
                    else:
                        # If option is not a dict, use the raw value
                        label = str(option)
                        code = str(option)

                    row[dim_name] = label
                    row[f"{dim_name}_code"] = code

            # Add the observation value
            row['observation'] = observations[obs_index]
            rows.append(row)

            # Increment indices (like counting with carry)
            for i in range(len(indices) - 1, -1, -1):
                indices[i] += 1
                if indices[i] < shape[i]:
                    break
                indices[i] = 0

            obs_index += 1

        # Generate field names preserving order
        fieldnames = []
        for dim in dim_info:
            fieldnames.append(dim['name'])
            fieldnames.append(f"{dim['name']}_code")
        fieldnames.append('observation')

        # Write to CSV
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    except Exception as e:
        logger.error(f"Error transforming batch data: {e}")
        import traceback
        logger.error(traceback.format_exc())


def append_batch_to_output(batch_file: str, output_file: str, header_written: bool) -> None:
    """
    Append a batch CSV file to the main output file.

    Args:
        batch_file (str): Path to the batch CSV file
        output_file (str): Path to the main output file
        header_written (bool): Whether the header has already been written to the output file
    """
    try:
        # If output file doesn't exist yet, just copy the batch file
        if not os.path.exists(output_file):
            import shutil
            shutil.copy(batch_file, output_file)
            logger.info(f"Created new output file from batch: {output_file}")
            return

        # Otherwise, append the batch data (skipping header)
        with open(batch_file, 'r', newline='') as batch_csv:
            reader = csv.reader(batch_csv)
            header = next(reader)  # Skip header row from batch

            with open(output_file, 'a', newline='') as out_csv:
                writer = csv.writer(out_csv)
                for row in reader:
                    writer.writerow(row)

        logger.info(f"Appended {batch_file} to {output_file}")

    except Exception as e:
        logger.error(f"Error appending batch to output: {e}")
        import traceback
        logger.error(traceback.format_exc())


# Example usage
if __name__ == "__main__":
    # Demonstration of the different ways to get data
    # Comment/uncomment examples as needed

    # Example 1: Get data for a single area (England)
    # data = get_fast_census_data(
    #     dataset_id="TS008",
    #     geo_level="ctry",
    #     area_codes=["E92000001"],  # England
    # )

    # Example 2: Download data for ALL regions in one go
    # output_file = download_all_areas_for_level(
    #     dataset_id="TS008",
    #     geo_level="rgn",
    #     batch_size=10  # Process 10 regions at a time
    # )
    # logger.info(f"All regions data saved to: {output_file}")

    # Example 3: Download ALL data for ALL geographic levels at once
    # all_levels_data = download_all_geographic_levels(
    #     dataset_id="TS008",
    #     batch_size=10  # Process 10 areas at a time for each level
    # )

    # Example 4: Get data for specific geographic levels
    # results = download_multiple_levels(
    #     dataset_id="TS008",
    #     area_sets={
    #         "ctry": ["E92000001"],        # England
    #         "rgn": ["E12000001", "E12000002"]  # North East, North West
    #     }
    # )

    pass
