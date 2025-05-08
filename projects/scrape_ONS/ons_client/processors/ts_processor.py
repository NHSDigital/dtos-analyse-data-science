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

    def process_response(self, response: Dict[str, Any], output_file: str, area_metadata: Dict[str, Any] = None) -> str:
        """
        Process TS dataset response and save to CSV.

        Args:
            response: Raw API response data
            output_file: Path to save the processed data
            area_metadata: Additional metadata about areas (optional)

        Returns:
            str: Path to the saved file
        """
        try:
            # Save debug info for potential later use
            debug_file = f"{output_file}.debug.json"

            # If we have area metadata, add it to the debug info
            if area_metadata:
                response["_area_metadata"] = area_metadata

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

    def flatten_data(self, input_file: str, output_file: Optional[str] = None, area_metadata: Dict[str, Any] = None) -> str:
        """
        Flatten TS dataset by reconstructing dimension combinations.

        This method:
        1. Reads the simple CSV with just a 'value' column
        2. Uses the metadata in the debug JSON to reconstruct dimensions
        3. Creates a flat structure with all dimension combinations

        Args:
            input_file: Path to the raw data CSV
            output_file: Path to save the flattened data (optional)
            area_metadata: Additional metadata about areas (optional)

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

                    # Merge any additional area metadata if provided
                    if area_metadata and "_area_metadata" not in metadata:
                        metadata["_area_metadata"] = area_metadata

                    # If we already have area metadata in the debug file, use that
                    if "_area_metadata" not in metadata and area_metadata:
                        metadata["_area_metadata"] = area_metadata

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

        # Check if we have more observations than combinations
        # This happens when we have metadata for only a subset of the areas
        if len(values) > total_combinations:
            logger.warning(f"More observations ({len(values)}) than dimension combinations ({total_combinations})")
            logger.info("Attempting to reconstruct complete dimension data")

            # Get area metadata if available
            area_metadata = metadata.get("_area_metadata", {})
            area_codes = area_metadata.get("area_codes", [])
            geo_level = area_metadata.get("geo_level", "")

            # For census datasets, the first dimension is usually geographic area
            # and the second dimension is the category (like religion)
            if len(dimensions) == 2:
                # Assume standard Census structure (area dim and category dim)
                geo_dim = dimensions[0]  # First dimension (geographic)
                cat_dim = dimensions[1]  # Second dimension (category)

                # Get the number of categories from existing metadata
                cat_options = dimension_values.get(cat_dim, [])
                num_categories = len(cat_options)

                if num_categories > 0:
                    # Calculate how many geographic areas we should have
                    num_areas = len(values) // num_categories

                    logger.info(f"Reconstructing data for {num_areas} areas with {num_categories} categories each")

                    # Check if we have area codes in the metadata
                    has_area_codes = len(area_codes) > 0
                    if has_area_codes:
                        logger.info(f"Using {len(area_codes)} area codes from metadata")
                        if len(area_codes) < num_areas:
                            logger.warning(f"Not enough area codes in metadata ({len(area_codes)}), needed {num_areas}")

                    # Generate flat rows with area IDs
                    flat_rows = []
                    for area_idx in range(num_areas):
                        for cat_idx in range(num_categories):
                            # Calculate the observation index
                            obs_index = area_idx * num_categories + cat_idx
                            if obs_index < len(values):
                                # Get category data from existing metadata
                                cat_option = cat_options[cat_idx] if cat_idx < len(cat_options) else {}

                                # Create row with area ID
                                row = {}

                                # Add geographic dimension with actual area code if available
                                if has_area_codes and area_idx < len(area_codes):
                                    # Use actual area code from metadata
                                    area_code = area_codes[area_idx]
                                    row[geo_dim] = f"Area {area_idx+1}"  # We don't have area names
                                    row[f"{geo_dim}_code"] = area_code
                                else:
                                    # Use reconstructed area ID
                                    row[geo_dim] = f"Area {area_idx+1}"
                                    row[f"{geo_dim}_code"] = f"AREA{area_idx+1:06d}"

                                # Add category dimension from existing metadata
                                if isinstance(cat_option, dict):
                                    # Try different key names for labels and IDs
                                    for label_key in ["label", "name", "value"]:
                                        if label_key in cat_option:
                                            row[cat_dim] = cat_option[label_key]
                                            break
                                    for id_key in ["id", "code"]:
                                        if id_key in cat_option:
                                            row[f"{cat_dim}_code"] = cat_option[id_key]
                                            break
                                else:
                                    # If option is just a string
                                    row[cat_dim] = str(cat_option)
                                    row[f"{cat_dim}_code"] = str(cat_option)

                                # Add observation value
                                row["observation"] = values[obs_index]
                                flat_rows.append(row)

                    logger.info(f"Created {len(flat_rows)} reconstructed rows")
                    return flat_rows, fieldnames

            # If we can't reconstruct with categories but have area codes, create area-based rows
            if area_codes:
                logger.info(f"Creating area-based rows with {len(area_codes)} areas")
                flat_rows = []
                values_per_area = len(values) // len(area_codes)
                for area_idx, area_code in enumerate(area_codes):
                    for value_idx in range(values_per_area):
                        obs_index = area_idx * values_per_area + value_idx
                        if obs_index < len(values):
                            row = {
                                geo_level: f"Area {area_idx+1}",
                                f"{geo_level}_code": area_code,
                                "value_index": value_idx,
                                "observation": values[obs_index]
                            }
                            flat_rows.append(row)
                return flat_rows, [geo_level, f"{geo_level}_code", "value_index", "observation"]

            # If we can't reconstruct, just create simple rows with values
            logger.info("Using simple method for large dataset")
            flat_rows = []
            for i, value in enumerate(values):
                row = {"observation": value, "row_index": i}
                flat_rows.append(row)
            return flat_rows, ["row_index", "observation"]

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
