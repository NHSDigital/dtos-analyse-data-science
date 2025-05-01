import json
import csv
import logging
import pandas as pd
import traceback
from typing import Dict, List, Any, Optional, Tuple
import os

from .base import BaseProcessor
from ..models.ts_models import TSResponse, TSFlattenedRow

logger = logging.getLogger(__name__)


class TSProcessor(BaseProcessor):
    """
    Processor for Time Series (TS) datasets.

    Handles the specific processing requirements for TS datasets:
    - Simple structure with a 'value' column
    - Dimensions stored in metadata
    - Requires reconstructing dimension combinations
    """

    def process_response(self, response: Dict[str, Any], output_file: str) -> str:
        """
        Process TS dataset response and save to CSV.

        Args:
            response: Raw API response data
            output_file: Path to save the processed data

        Returns:
            str: Path to the saved file
        """
        try:
            # Save debug info for potential later use
            debug_file = f"{output_file}.debug.json"
            with open(debug_file, "w") as f:
                json.dump(response, f, indent=2)

            logger.debug(f"Response keys: {list(response.keys())}")

            # Log response structure for debugging
            if "dimensions" in response:
                logger.debug(f"Dimensions in response: {len(response['dimensions'])}")
                for i, dim in enumerate(response["dimensions"]):
                    if isinstance(dim, dict):
                        logger.debug(
                            f"Dimension {i}: {dim.get('dimension_name', 'unknown')}"
                        )

            # For TS datasets, the response should have observations array
            observations = response.get("observations", [])

            if not observations:
                logger.warning(f"No observations found in response")
                return ""

            # For TS datasets, we simply write the values to CSV
            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["value"])
                for value in observations:
                    writer.writerow([value])

            logger.info(f"Saved {len(observations)} observations to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error processing TS response: {str(e)}")
            logger.error(traceback.format_exc())
            return ""

    def flatten_data(self, input_file: str, output_file: Optional[str] = None) -> str:
        """
        Flatten TS dataset by reconstructing dimension combinations.

        This method:
        1. Reads the simple CSV with just a 'value' column
        2. Uses the metadata in the debug JSON to reconstruct dimensions
        3. Creates a flat structure with all dimension combinations

        Args:
            input_file: Path to the raw data CSV
            output_file: Path to save the flattened data (optional)

        Returns:
            str: Path to the flattened data file
        """
        if not output_file:
            output_file = self.get_default_output_file(input_file)

        if not self.validate_file_exists(input_file):
            return ""

        try:
            # Read the input CSV values
            df = pd.read_csv(input_file)
            values = df["value"].tolist() if "value" in df.columns else []

            if not values:
                logger.warning(f"No values found in {input_file}")
                return ""

            # Look for debug file with metadata
            debug_file = f"{input_file}.debug.json"

            # Check if debug file exists with raw response
            if os.path.exists(debug_file):
                try:
                    with open(debug_file, "r") as f:
                        metadata = json.load(f)
                    logger.info(f"Found debug file with metadata: {debug_file}")
                    logger.debug(f"Metadata keys: {list(metadata.keys())}")

                    # Inspect metadata structure
                    if "dimensions" in metadata:
                        dimensions = metadata["dimensions"]
                        logger.debug(f"Dimensions type: {type(dimensions).__name__}")
                        if isinstance(dimensions, list) and dimensions:
                            logger.debug(
                                f"First dimension keys: {list(dimensions[0].keys()) if isinstance(dimensions[0], dict) else 'not a dict'}"
                            )

                    # Process the metadata to create dimension combinations
                    flat_rows, fieldnames = self._create_dimension_combinations(
                        metadata, values
                    )

                    if flat_rows:
                        # Write the flat CSV with dimension info
                        with open(output_file, "w", newline="") as outfile:
                            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(flat_rows)

                        logger.info(
                            f"Created flattened file with {len(flat_rows)} rows at {output_file}"
                        )
                        return output_file

                except Exception as e:
                    logger.error(f"Error processing metadata: {str(e)}")
                    logger.error(traceback.format_exc())

            # If no metadata or processing failed, just copy the original file
            logger.info(
                f"No metadata available or processing failed, creating simple copy"
            )
            df.to_csv(output_file, index=False)
            return output_file

        except Exception as e:
            logger.error(f"Error flattening TS data: {str(e)}")
            logger.error(traceback.format_exc())
            return ""

    def _create_dimension_combinations(
        self, metadata: Dict[str, Any], values: List[str]
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        """
        Create all possible dimension combinations based on metadata.

        Args:
            metadata: Dataset metadata containing dimensions
            values: List of observation values

        Returns:
            Tuple containing:
            - List of dictionaries, each representing a row
            - List of fieldnames for the CSV
        """
        # Extract dimension information
        dimension_data = metadata.get("dimensions", [])

        # Get all possible dimension values
        dimensions = []
        dimension_values = {}

        # TS dimension structures can vary, so try different approaches
        try:
            # First approach: dimension_name and options format
            if isinstance(dimension_data, list):
                for dim in dimension_data:
                    if isinstance(dim, dict):
                        # Try different key names for dimension name
                        dim_name = None
                        for key in ["dimension_name", "name", "id"]:
                            if key in dim:
                                dim_name = dim.get(key)
                                break

                        if dim_name:
                            dimensions.append(dim_name)
                            # Try different key names for options
                            for key in ["options", "items", "values"]:
                                if key in dim:
                                    dimension_values[dim_name] = dim[key]
                                    break

            # If no dimensions found, try other metadata structures
            if not dimensions and "dimension_metadata" in metadata:
                dim_metadata = metadata["dimension_metadata"]
                if isinstance(dim_metadata, list):
                    for dim in dim_metadata:
                        if isinstance(dim, dict) and "name" in dim:
                            dimensions.append(dim["name"])
                            if "values" in dim:
                                dimension_values[dim["name"]] = dim["values"]

        except Exception as e:
            logger.error(f"Error extracting dimensions: {str(e)}")
            logger.error(traceback.format_exc())

        if not dimensions:
            logger.warning("No dimensions found in metadata")
            # As a fallback, check if we have a complete pre-formatted dataset
            if (
                "ctry" in metadata
                and "religion_tb" in metadata
                and "observation" in metadata
            ):
                logger.info("Found pre-formatted dataset with dimensions")
                # Create a simple fieldnames list and rows from the pre-formatted data
                fieldnames = [
                    "ctry",
                    "ctry_code",
                    "religion_tb",
                    "religion_tb_code",
                    "observation",
                ]
                try:
                    # Attempt to construct rows from the pre-formatted data
                    ctry_values = metadata.get("ctry", [])
                    ctry_codes = metadata.get("ctry_code", [])
                    religion_values = metadata.get("religion_tb", [])
                    religion_codes = metadata.get("religion_tb_code", [])
                    observation_values = metadata.get("observation", values)

                    flat_rows = []
                    for i in range(
                        min(
                            len(ctry_values),
                            len(religion_values),
                            len(observation_values),
                        )
                    ):
                        row = {
                            "ctry": ctry_values[i] if i < len(ctry_values) else "",
                            "ctry_code": ctry_codes[i] if i < len(ctry_codes) else "",
                            "religion_tb": (
                                religion_values[i] if i < len(religion_values) else ""
                            ),
                            "religion_tb_code": (
                                religion_codes[i] if i < len(religion_codes) else ""
                            ),
                            "observation": (
                                observation_values[i]
                                if i < len(observation_values)
                                else ""
                            ),
                        }
                        flat_rows.append(row)
                    return flat_rows, fieldnames
                except Exception as e:
                    logger.error(
                        f"Error creating rows from pre-formatted data: {str(e)}"
                    )

            return [], []

        logger.info(f"Found dimensions in metadata: {dimensions}")

        # Create fieldnames for all dimensions and their codes
        fieldnames = []
        for dim in dimensions:
            fieldnames.append(dim)
            fieldnames.append(f"{dim}_code")
        fieldnames.append("observation")

        # Calculate the total number of combinations for each dimension
        dim_counts = []
        for dim in dimensions:
            options = dimension_values.get(dim, [])
            count = len(options) if options else 1
            dim_counts.append(count)

        # Calculate product of all dimension counts
        total_combinations = 1
        for count in dim_counts:
            total_combinations *= count

        logger.info(
            f"Total combinations: {total_combinations}, Observations: {len(values)}"
        )

        # Generate all dimension combinations and map to observations
        flat_rows = []

        # Initialize indices for each dimension
        indices = [0] * len(dimensions)
        obs_index = 0

        # Generate all dimension combinations
        # This matches ONS API's ordering logic
        while obs_index < len(values) and obs_index < total_combinations:
            # Create a new row
            row = {}

            # Add dimension values and codes for this combination
            for i, dim in enumerate(dimensions):
                options = dimension_values.get(dim, [])
                if options and indices[i] < len(options):
                    opt = options[indices[i]]
                    # Handle different option formats
                    if isinstance(opt, dict):
                        # Try different key names for labels and IDs
                        for label_key in ["label", "name", "value"]:
                            if label_key in opt:
                                row[dim] = opt[label_key]
                                break
                        for id_key in ["id", "code"]:
                            if id_key in opt:
                                row[f"{dim}_code"] = opt[id_key]
                                break
                    else:
                        # If option is just a string
                        row[dim] = str(opt)
                        row[f"{dim}_code"] = str(opt)

            # Add the observation value
            row["observation"] = values[obs_index] if obs_index < len(values) else ""
            flat_rows.append(row)

            # Increment indices (like counting with carry)
            # Last dimension changes fastest, first dimension changes slowest
            for i in range(len(indices) - 1, -1, -1):
                indices[i] += 1
                if indices[i] < dim_counts[i]:
                    break
                indices[i] = 0

            # Move to next observation
            obs_index += 1

        return flat_rows, fieldnames
