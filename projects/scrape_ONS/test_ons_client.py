#!/usr/bin/env python3
"""
Test script for ONSApiClient.
This script retrieves and prints the dimensions available in dataset TS001.
"""

from ons_client.ons_client import ONSApiClient


def main() -> None:
    # Instantiate the client
    client = ONSApiClient()
    try:
        # Using dataset id 'TS001', edition '2021', version '1'
        dimensions_response = client.get_dataset_dimensions('TS001', '2021', '3')
        metadata_response = client.get_dataset_metadata('TS001', '2021', '3')
        print('Dimensions for dataset TS001:')
        for dimension in dimensions_response.items:
            # print(f"ID: {dimension.id}, Label: {dimension.label}")
            print(f"ID: {dimension.id}")
        # print('Metadata for dataset TS001:')
        # print(metadata_response)

        # Get dimension options for the first dimension (if available)
        if dimensions_response.items:
            print(f"Dimensions: {dimensions_response.items}")
            first_dimension = dimensions_response.items[0]
            options_response = client.get_dimension_options('TS001', '2021', '3', first_dimension.id)
            print(f"Dimension options for dimension {first_dimension.id}:")
            for option in options_response.items:
                print(f"ID: {option.id}, Label: {option.label}")
        else:
            print("No dimensions available to fetch dimension options.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
