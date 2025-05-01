import json
import csv
import logging
import pandas as pd
import traceback
from typing import Dict, List, Any, Optional, Set, Tuple
import os

from .base import BaseProcessor
from ..models.rm_models import RMObservation, RMResponse

logger = logging.getLogger(__name__)


class RMProcessor(BaseProcessor):
    """
    Processor for Regular Matrix (RM) datasets.

    Handles the specific processing requirements for RM datasets:
    - Similar to TS datasets, but with different dimensions
    - Flat output format with one dimension per column
    """

    def process_response(self, response: Dict[str, Any], output_file: str) -> str:
        """
        Process RM dataset response and save to CSV.

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

            # RM datasets have the same response format as TS datasets
            observations = response.get("observations", [])

            if not observations:
                logger.warning(f"No observations found in response")
                return ""

            # For RM datasets (like TS), we simply write the values to CSV
            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["value"])
                for value in observations:
                    writer.writerow([value])

            logger.info(f"Saved {len(observations)} observations to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error processing RM response: {str(e)}")
            logger.error(traceback.format_exc())
            return ""

    def flatten_data(self, input_file: str, output_file: Optional[str] = None) -> str:
        """
        Flatten RM dataset by reconstructing dimension combinations.

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
            # Check if there's a debug JSON file with additional information
            debug_file = f"{input_file}.debug.json"
            metadata = None

            if os.path.exists(debug_file):
                try:
                    with open(debug_file, "r") as f:
                        metadata = json.load(f)
                    logger.info(f"Found debug file with metadata: {debug_file}")
                except Exception as e:
                    logger.warning(f"Could not read debug file: {str(e)}")

            if not metadata:
                logger.warning("No metadata found, cannot flatten file")
                return ""

            # Read the input CSV values
            df = pd.read_csv(input_file)
            values = df["value"].tolist() if "value" in df.columns else []

            if not values:
                logger.warning(f"No values found in {input_file}")
                return ""

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
            else:
                logger.warning("No rows generated to write to flattened CSV")
                return ""

        except Exception as e:
            logger.error(f"Error flattening RM data: {str(e)}")
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

        # Extract dimensions and options
        if isinstance(dimension_data, list):
            for dim in dimension_data:
                if isinstance(dim, dict) and "dimension_name" in dim:
                    dim_name = dim.get("dimension_name")
                    dimensions.append(dim_name)
                    # Store dimension values if available
                    if "options" in dim:
                        dimension_values[dim_name] = dim["options"]

        if not dimensions:
            logger.warning("No dimensions found in metadata")
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
                        row[dim] = opt.get("label", "")
                        row[f"{dim}_code"] = opt.get("id", "")
                    else:
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
