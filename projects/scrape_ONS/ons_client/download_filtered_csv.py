import time
import requests
import json
from typing import Optional, List, Dict

import csv
import io
import os

from ons_client import ONSApiClient
from models import FilterRequest, DatasetIdentifier, DimensionFilter, Area
from filter_processor import FilterProcessor
from utils import chunk_list, ensure_dir

BASE_URL = "https://api.beta.ons.gov.uk/v1"


def download_filtered_csv(
        dataset_id: str,
        population_type: str,
        area_type: str,
        area_code: Optional[str] = None,
        edition: str = "2021",
        version: int = 1,
        output_file: Optional[str] = None,
        batch_size: int = 500
    ) -> List[str]:
    """
    Creates a filter for a given dataset with the specified population type and area type.
    If area_code is provided, downloads the CSV filter output for that specific area code.
    If area_code is None, retrieves all areas for the given population_type and area_type,
    and downloads the CSV outputs in batches, combining them into a single file.

    Parameters:
        dataset_id (str): The dataset ID (e.g., "TS008").
        population_type (str): The population type (e.g., "UR").
        area_type (str): The dimension identifier for the geography filter (e.g., "rgn" or "ctry").
        area_code (Optional[str]): The area code to filter on (e.g., "E12000001"). If None, downloads for all areas.
        edition (str, optional): The dataset edition. Defaults to "2021".
        version (int, optional): The dataset version. Defaults to 1.
        output_file (Optional[str]): The file path to save the CSV file for a single area.
        If None, a default path will be used based on the dataset and area type.
        batch_size (int, optional): Number of area codes to process in each batch. Defaults to 500.

    Returns:
        List[str]: List of paths to the downloaded CSV files.
    """
    # Create directory structure for the dataset
    dataset_dir = os.path.join("data", dataset_id)
    ensure_dir("data")
    ensure_dir(dataset_dir)

    # Set up the filter processor
    filter_processor = FilterProcessor()

    # If output_file is None, generate a default path in the dataset directory
    if output_file is None:
        if area_code is None:
            output_file = os.path.join(dataset_dir, f"{area_type}_all.csv")
        else:
            output_file = os.path.join(dataset_dir, f"{area_type}_{area_code}.csv")

    # If no area_code is provided, get all areas and process in batches
    if area_code is None:
        client = ONSApiClient()
        areas = client.get_cached_areas(population_type, area_type)
        if not areas:
            raise Exception(f"No areas found for population type '{population_type}' and area type '{area_type}'.")

        # Get all area codes for this area type
        all_area_codes = [area.id for area in areas]
        print(f"Found {len(all_area_codes)} area codes for area type '{area_type}'")

        # If we have fewer areas than batch_size, process them all at once
        if len(all_area_codes) <= batch_size:
            print(f"Processing all {len(all_area_codes)} area codes in a single batch")
            filter_request = filter_processor.create_filter_request(
                dataset_id=dataset_id,
                population_type=population_type,
                area_type=area_type,
                area_codes=all_area_codes,
                edition=edition,
                version=version
            )
            downloaded_file = filter_processor.process_filter(
                filter_request, dataset_id, area_type, "all", output_file
            )
            return [downloaded_file]
        else:
            # Split area codes into chunks and process each chunk
            area_code_chunks = chunk_list(all_area_codes, batch_size)
            print(f"Splitting {len(all_area_codes)} area codes into {len(area_code_chunks)} batches of {batch_size}")

            # Initialize variables to track file creation and headers
            output_file_created = False
            csv_headers = None

            # Use a temporary directory for batch files
            temp_dir = os.path.join(dataset_dir, "temp")
            ensure_dir(temp_dir)

            # Process each batch and append to the output file
            for i, chunk in enumerate(area_code_chunks):
                print(f"Processing batch {i+1}/{len(area_code_chunks)} with {len(chunk)} area codes")
                # Use a temporary file name for each batch download
                temp_output_file = os.path.join(temp_dir, f"{area_type}_batch{i+1}.csv")

                filter_request = filter_processor.create_filter_request(
                    dataset_id=dataset_id,
                    population_type=population_type,
                    area_type=area_type,
                    area_codes=chunk,
                    edition=edition,
                    version=version
                )

                # Process the batch and get the temporary file
                temp_file = filter_processor.process_filter(
                    filter_request, dataset_id, area_type, f"batch{i+1}", temp_output_file
                )

                # Read the temporary file and append to output file
                try:
                    with open(temp_file, 'r', newline='', encoding='utf-8') as temp_csv:
                        reader = csv.reader(temp_csv)

                        # Get headers from first batch
                        if not output_file_created:
                            csv_headers = next(reader)
                            # Create the output file with headers
                            with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                                writer = csv.writer(outfile)
                                writer.writerow(csv_headers)
                            output_file_created = True
                        else:
                            # Skip header for subsequent batches
                            next(reader)

                        # Append data rows to output file
                        with open(output_file, 'a', newline='', encoding='utf-8') as outfile:
                            writer = csv.writer(outfile)
                            for row in reader:
                                writer.writerow(row)

                    # Clean up temporary file
                    os.remove(temp_file)
                    print(f"Appended batch {i+1} data to {output_file} and removed temporary file")

                except Exception as e:
                    print(f"Error processing batch {i+1}: {str(e)}")

                # Add a delay between batches to respect rate limits
                if i < len(area_code_chunks) - 1:  # Don't delay after the last batch
                    delay = 5  # 5 seconds between batches
                    print(f"Pausing for {delay} seconds before processing next batch to avoid rate limits...")
                    time.sleep(delay)

            # Try to remove the temp directory if it's empty
            try:
                os.rmdir(temp_dir)
                print(f"Removed temporary directory: {temp_dir}")
            except:
                pass  # Directory might not be empty or might have been removed

            if output_file_created:
                print(f"Successfully created and populated {output_file} with all batched data")
                return [output_file]
            else:
                print("Failed to create output file from batches")
                return []

    else:
        # Process for single provided area_code
        filter_request = filter_processor.create_filter_request(
            dataset_id=dataset_id,
            population_type=population_type,
            area_type=area_type,
            area_codes=[area_code],
            edition=edition,
            version=version
        )
        downloaded_file = filter_processor.process_filter(
            filter_request, dataset_id, area_type, area_code, output_file
        )
        return [downloaded_file]


def download_all_area_types(dataset_id: str, population_type: str, edition: str = "2021", version: int = 1) -> Dict[str, List[str]]:
    """
    For the specified dataset and population type, this function retrieves all available area types
    and downloads the CSV outputs for each area within each area type using the download_filtered_csv function.

    Parameters:
        dataset_id (str): The dataset ID (e.g., "TS008"). Must be a valid ID.
        population_type (str): The population type (e.g., "UR").
        edition (str, optional): The dataset edition. Defaults to "2021".
        version (int, optional): The dataset version. Defaults to 1.

    Returns:
        dict: A dictionary mapping each area type id to a list of downloaded CSV file paths.
    """
    # Validate dataset_id is not empty
    if not dataset_id:
        raise ValueError("Dataset ID cannot be empty. Please provide a valid dataset ID (e.g., 'TS008').")

    # Create dataset directory
    dataset_dir = os.path.join("data", dataset_id)
    ensure_dir("data")
    ensure_dir(dataset_dir)

    client = ONSApiClient()
    area_types = client.get_area_types(population_type)
    if not area_types:
        raise Exception(f"No area types found for population type '{population_type}'.")

    results = {}
    for area_type in area_types:
        try:
            # For each area type, retrieve the list of areas
            areas = client.get_cached_areas(population_type, area_type.id)
            if areas:
                # Generate output file path in the dataset directory
                output_file = os.path.join(dataset_dir, f"{area_type.id}.csv")

                print(f"Downloading CSV for area type '{area_type.name}' into file '{output_file}'")
                csv_files = download_filtered_csv(
                    dataset_id, population_type, area_type.id, None, edition, version, output_file, batch_size=2000
                )
                results[area_type.id] = csv_files
            else:
                print(f"No areas found for area type '{area_type.name}' (id: {area_type.id}), skipping CSV download.")
        except Exception as e:
            print(f"Error processing area type '{area_type.name}' (id: {area_type.id}): {str(e)}. Skipping CSV download.")

    return results


if __name__ == '__main__':
    # Example usage:
    # For dataset TS008 and population type "UR", first check all available area types
    # and then download the CSV outputs for each area within each area type.
    all_csv_files = download_all_area_types("TS003", "UR")  # Using a valid dataset ID
    print("Downloaded CSV files for all area types:")
    print(all_csv_files)
