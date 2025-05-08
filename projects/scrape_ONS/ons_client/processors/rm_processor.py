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

    def process_response(self, response: Dict[str, Any], output_file: str, area_metadata: Dict[str, Any] = None) -> str:
        """
        Process RM dataset response and save to CSV.

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

    def flatten_data(self, input_file: str, output_file: Optional[str] = None, area_metadata: Dict[str, Any] = None) -> str:
        """
        Flatten RM dataset by reconstructing dimension combinations.

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
            # Check if there's a debug JSON file with additional information
            debug_file = f"{input_file}.debug.json"
            metadata = None

            if os.path.exists(debug_file):
                try:
                    with open(debug_file, "r") as f:
                        metadata = json.load(f)
                    logger.info(f"Found debug file with metadata: {debug_file}")

                    # Merge any additional area metadata if provided
                    if area_metadata and "_area_metadata" not in metadata:
                        metadata["_area_metadata"] = area_metadata

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
            # and second dimension is usually the category
            if len(dimensions) >= 2:
                # Assume standard Census structure
                geo_dim = dimensions[0]  # First dimension (geographic)

                # Get category dimensions (everything else)
                category_dims = dimensions[1:]

                # Get category options for all dimensions
                category_options = []
                for dim in category_dims:
                    options = dimension_values.get(dim, [])
                    if options:
                        category_options.append((dim, options))

                if category_options:
                    # Calculate number of categories per area
                    num_categories_per_area = 1
                    for _, options in category_options:
                        num_categories_per_area *= len(options)

                    # Calculate how many geographic areas we should have
                    num_areas = len(values) // num_categories_per_area

                    logger.info(f"Reconstructing data for {num_areas} areas with {num_categories_per_area} categories each")

                    # Check if we have area codes in the metadata
                    has_area_codes = len(area_codes) > 0
                    if has_area_codes:
                        logger.info(f"Using {len(area_codes)} area codes from metadata")
                        if len(area_codes) < num_areas:
                            logger.warning(f"Not enough area codes in metadata ({len(area_codes)}), needed {num_areas}")

                    # Generate flat rows with area IDs and reconstruct all combinations
                    flat_rows = []

                    # Define a recursive function to generate all category combinations
                    def generate_combinations(dim_idx, current_row, area_idx, start_obs_idx):
                        if dim_idx >= len(category_options):
                            # Add observation value
                            current_row["observation"] = values[start_obs_idx] if start_obs_idx < len(values) else ""
                            flat_rows.append(current_row.copy())
                            return 1

                        dim_name, options = category_options[dim_idx]
                        total_added = 0

                        for i, opt in enumerate(options):
                            if isinstance(opt, dict):
                                # If option is a dictionary with label and id
                                for label_key in ["label", "name", "value"]:
                                    if label_key in opt:
                                        current_row[dim_name] = opt[label_key]
                                        break
                                for id_key in ["id", "code"]:
                                    if id_key in opt:
                                        current_row[f"{dim_name}_code"] = opt[id_key]
                                        break
                            else:
                                # If option is a string or other value
                                current_row[dim_name] = str(opt)
                                current_row[f"{dim_name}_code"] = str(opt)

                            # Calculate observation index
                            obs_index = start_obs_idx + total_added

                            # Recursively add next dimension
                            added = generate_combinations(dim_idx + 1, current_row, area_idx, obs_index)
                            total_added += added

                        return total_added

                    # Generate combinations for each area
                    obs_index = 0
                    for area_idx in range(min(num_areas, len(values) // num_categories_per_area)):
                        # Create row template with area info
                        base_row = {}

                        # Add geographic dimension with actual area code if available
                        if has_area_codes and area_idx < len(area_codes):
                            # Use actual area code from metadata
                            area_code = area_codes[area_idx]
                            base_row[geo_dim] = f"Area {area_idx+1}"  # We don't have area names
                            base_row[f"{geo_dim}_code"] = area_code
                        else:
                            # Use reconstructed area ID
                            base_row[geo_dim] = f"Area {area_idx+1}"
                            base_row[f"{geo_dim}_code"] = f"AREA{area_idx+1:06d}"

                        # Generate all category combinations for this area
                        generate_combinations(0, base_row, area_idx, area_idx * num_categories_per_area)

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
