#!/usr/bin/env python3
"""
Test script for fast census data retrieval.
This script demonstrates using the direct census-observations endpoint
and other fast methods to access ONS data without filter jobs.
Tests all geographic levels from country to output area.
"""

import json
import csv
import os
import sys
import time
from ons_client import ONSApiClient
from ons_client.fast_census_data import get_fast_census_data, get_areas_for_level


def main():
    """Test the fast census data retrieval methods."""
    client = ONSApiClient()

    # Create directory for output
    os.makedirs('data/fast_retrieval', exist_ok=True)

    # Test census observations approach with all geographic levels
    test_all_geographic_levels(client)

    # Try direct dataset access as an alternative
    try_direct_dataset_access(client)


def test_all_geographic_levels(client):
    """Test retrieving data at all geographic levels using census-observations."""
    print("\n===== TESTING ALL GEOGRAPHIC LEVELS =====")

    # Dictionary of geographic levels from largest to smallest
    # With sample area codes for testing
    geographic_levels = {
        "ctry": ["E92000001"],                  # Country (England)
        "rgn": ["E12000001"],                   # Region (North East)
        "cty": ["E10000002"],                   # County (Buckinghamshire)
        "lad": ["E06000022"],                   # Local Authority (Bath and North East Somerset)
        "msoa": ["E02000001"],                  # Middle Super Output Area
        "lsoa": ["E01000001"],                  # Lower Super Output Area
        "oa": ["E00000001"]                     # Output Area
    }

    # Dimensions that are known to work with census-observations
    dimensions = ["health_in_general", "highest_qualification"]

    # Test each geographic level
    for level, area_codes in geographic_levels.items():
        print(f"\n--- Testing {level.upper()} level ---")
        try:
            # Get data for this level
            observations = client.get_fast_census_data(
                population_type="UR",
                area_type=level,
                area_codes=area_codes,
                dimensions=dimensions
            )

            # Check if we got observations
            if 'observations' in observations and observations['observations']:
                count = len(observations['observations'])
                print(f"Success! Retrieved {count} observations for {level}")

                # Save and convert results
                save_and_convert(observations, f'level_{level}')

                # Show a sample observation
                if count > 0:
                    print("\nSample observation:")
                    print(json.dumps(observations['observations'][0], indent=2))
            else:
                print(f"No observations found for {level}")

        except Exception as e:
            print(f"Error retrieving {level} data: {e}")
            print(f"The {level} level may not work with direct census-observations.")

        # Brief pause to avoid rate limiting
        time.sleep(1)


def try_direct_dataset_access(client):
    """Try using the dataset direct access endpoint."""
    print("\n===== TESTING DIRECT DATASET ACCESS =====")

    # Try a few dataset IDs
    datasets_to_try = ['TS003', 'TS008', 'TS021']

    for dataset_id in datasets_to_try:
        try:
            print(f"\nTrying direct access for dataset {dataset_id}:")

            # First get dimensions to understand the dataset
            dimensions_response = client.get_dataset_dimensions(dataset_id, '2021', '1')
            print(f"Available dimensions for dataset {dataset_id}:")
            for dim in dimensions_response.items:
                if dim.id.lower() != 'ltla':
                    print(f"  - {dim.id}: {dim.label}")

            # Try to get data directly using the dataset json endpoint
            data = client.get_dataset_observations(dataset_id, '2021', '1')

            # Save and display results
            print(f"Success! Retrieved data for {dataset_id}")

            # Save the raw response
            with open(f'data/fast_retrieval/dataset_{dataset_id}.json', 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved raw data to data/fast_retrieval/dataset_{dataset_id}.json")

            # Display structure of the response
            print("\nResponse structure:")
            for key in data.keys():
                if isinstance(data[key], list):
                    print(f"  - {key}: list with {len(data[key])} items")
                else:
                    print(f"  - {key}: {type(data[key]).__name__}")

            # If the response contains observations, convert to CSV
            if 'observations' in data and data['observations']:
                output_file = f'data/fast_retrieval/dataset_{dataset_id}.csv'
                convert_dataset_to_csv(data, output_file)

            # Break out of loop on first success
            break

        except Exception as e:
            print(f"Error with dataset {dataset_id}: {e}")
            print("Trying next dataset...")

    else:  # This runs if the loop completes without a break
        print("\nCould not access any of the test datasets directly.")
        print("You may need to try the filter API for more complex requirements.")


def convert_dataset_to_csv(data, output_file):
    """Convert dataset observations to CSV, handling different formats."""
    try:
        if 'observations' not in data:
            print("No observations found to convert to CSV")
            return

        observations = data['observations']
        if not observations:
            print("Empty observations list")
            return

        # Format depends on the structure of the observations
        if isinstance(observations[0], dict):
            # Dictionary format - extract headers from keys
            with open(output_file, 'w', newline='') as csvfile:
                all_keys = set()
                for obs in observations:
                    all_keys.update(obs.keys())

                writer = csv.DictWriter(csvfile, fieldnames=sorted(all_keys))
                writer.writeheader()
                writer.writerows(observations)
        else:
            # Simple value format - extract dimension info from 'dimensions'
            if 'dimensions' not in data:
                print("Cannot convert to CSV: no dimensions information available")
                return

            dimensions = data['dimensions']

            # Create a header row based on dimensions plus 'observation'
            headers = [f"{dim.get('name', dim.get('dimension_id', f'dim{i}'))}"
                      for i, dim in enumerate(dimensions)]
            headers.append('observation')

            # Create CSV with dimension values and observation
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)

                # Add a row for each observation value
                for i, obs in enumerate(observations):
                    # For each observation, get the corresponding dimension values
                    row = []
                    for dim_index, dim in enumerate(dimensions):
                        if 'options' in dim and i < len(dim['options']):
                            row.append(dim['options'][i].get('option', ''))
                        else:
                            row.append('')
                    row.append(obs)
                    writer.writerow(row)

        print(f"Converted to CSV: {output_file}")
    except Exception as e:
        print(f"Error converting to CSV: {e}")
        print(f"Data structure: {type(observations[0])}")


def save_and_convert(observations_data, name_suffix):
    """Save observations as JSON and CSV."""
    # Save to JSON
    json_file = f'data/fast_retrieval/observations_{name_suffix}.json'
    with open(json_file, 'w') as f:
        json.dump(observations_data, f, indent=2)
    print(f"Saved to JSON: {json_file}")

    # Convert to CSV if there are observations
    if 'observations' in observations_data and observations_data['observations']:
        csv_file = f'data/fast_retrieval/observations_{name_suffix}.csv'
        try:
            with open(csv_file, 'w', newline='') as csvfile:
                # Check if observations are dictionaries
                observations = observations_data['observations']
                if observations and isinstance(observations[0], dict):
                    all_keys = set()
                    for obs in observations:
                        all_keys.update(obs.keys())

                    writer = csv.DictWriter(csvfile, fieldnames=sorted(all_keys))
                    writer.writeheader()
                    writer.writerows(observations)
                else:
                    # If observations are not dictionaries, create a simple format
                    writer = csv.writer(csvfile)
                    writer.writerow(['observation'])
                    for obs in observations:
                        writer.writerow([obs])

            print(f"Converted to CSV: {csv_file}")
        except Exception as e:
            print(f"Error converting to CSV: {e}")


if __name__ == '__main__':
    main()
