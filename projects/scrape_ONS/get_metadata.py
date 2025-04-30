#!/usr/bin/env python3
"""
Script to get metadata from a specific dataset to analyze its structure
"""

import json
import os
from ons_client.ons_client import ONSApiClient

# Configure the dataset you want to investigate
DATASET_ID = "TS030"  # Change to the dataset ID you want to explore
EDITION = "2021"
VERSION = "1"

# Initialize client
client = ONSApiClient()

# Create output directory
output_dir = f"data/{DATASET_ID}_metadata"
os.makedirs(output_dir, exist_ok=True)

# Get metadata
print(f"Getting metadata for {DATASET_ID}...")
metadata = client.get_dataset_metadata(DATASET_ID, EDITION, VERSION)

# Save metadata to file
metadata_file = os.path.join(output_dir, f"{DATASET_ID}_metadata.json")
with open(metadata_file, 'w') as f:
    # Use Pydantic model_dump method to convert to dict
    json.dump(metadata.model_dump(by_alias=True), f, indent=2)
print(f"Saved metadata to {metadata_file}")

# Get dimensions
print(f"Getting dimensions for {DATASET_ID}...")
try:
    dimensions = client.get_dataset_dimensions(DATASET_ID, EDITION, VERSION)

    # Save dimensions to file
    dimensions_file = os.path.join(output_dir, f"{DATASET_ID}_dimensions.json")
    with open(dimensions_file, 'w') as f:
        # Use Pydantic model_dump method to convert to dict
        dimensions_data = [dim.model_dump(by_alias=True) for dim in dimensions.items]
        json.dump(dimensions_data, f, indent=2)
    print(f"Saved dimensions to {dimensions_file}")

    # For each dimension, get its options
    for dim in dimensions.items:
        dim_id = dim.id or dim.name  # Use id or name as fallback
        if not dim_id:
            print(f"Skipping dimension with no ID or name: {dim}")
            continue

        print(f"Getting options for dimension {dim_id}...")
        try:
            options = client.get_dimension_options(DATASET_ID, EDITION, VERSION, dim_id)

            # Save options to file
            options_file = os.path.join(output_dir, f"{DATASET_ID}_dimension_{dim_id}_options.json")
            with open(options_file, 'w') as f:
                # Use Pydantic model_dump method to convert to dict
                options_data = [opt.model_dump(by_alias=True) for opt in options.items]
                json.dump(options_data, f, indent=2)
            print(f"Saved {len(options.items)} options for dimension {dim_id}")
        except Exception as e:
            print(f"Error getting options for dimension {dim_id}: {e}")
except Exception as e:
    print(f"Error getting dimensions: {e}")

# Get a sample observation (if needed)
print(f"Getting sample observations for {DATASET_ID}...")
try:
    # Get observations directly using the client method
    observations = client.get_dataset_observations_by_area_type(
        dataset_id=DATASET_ID,
        edition=EDITION,
        version=VERSION,
        area_type="ctry",
        area_codes=["E92000001"]  # England
    )

    # Save observations to file
    observations_file = os.path.join(output_dir, f"{DATASET_ID}_sample_observations.json")
    with open(observations_file, 'w') as f:
        json.dump(observations, f, indent=2)
    print(f"Saved sample observations to {observations_file}")

    # Print key info about the observations
    print(f"Observation keys: {list(observations.keys() if isinstance(observations, dict) else [])}")
    if isinstance(observations, dict) and 'observations' in observations:
        print(f"Number of observations: {len(observations['observations'])}")
        if observations['observations']:
            print(f"Sample observation type: {type(observations['observations'][0]).__name__}")
            if isinstance(observations['observations'][0], dict):
                print(f"Sample observation keys: {list(observations['observations'][0].keys())}")
except Exception as e:
    print(f"Error getting sample observations: {e}")

print("Done!")
