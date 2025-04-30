import logging
from typing import Dict, List, Any, Optional
import requests
import json
from tqdm import tqdm

from .client import ONSApiClient, with_retry
from ..models.ts_models import TSResponse

logger = logging.getLogger(__name__)

class TSApiClient(ONSApiClient):
    """
    Client for interacting with Time Series (TS) datasets in the ONS API.

    Handles TS-specific API calls and data formatting.
    """

    @with_retry(max_retries=3)
    def get_dataset_data(self,
                         dataset_id: str,
                         area_codes: List[str],
                         geo_level: str = "ctry",
                         population_type: str = "UR") -> Dict[str, Any]:
        """
        Retrieve data for a TS dataset for specific areas.

        Args:
            dataset_id: The Time Series dataset ID
            area_codes: List of area codes to retrieve data for
            geo_level: Geographic level (default: "ctry")
            population_type: The population type (default: "UR")

        Returns:
            Dict containing the dataset response
        """
        if not dataset_id or not dataset_id.startswith("TS"):
            raise ValueError(f"Invalid TS dataset ID: {dataset_id}")

        if not area_codes:
            raise ValueError("No area codes provided")

        logger.debug(f"Fetching TS dataset {dataset_id} for {len(area_codes)} areas")

        # TS datasets have a specific format: area-type=geo_level,area1,area2,...
        # Format according to ONS API docs: /datasets/TS008/editions/2021/versions/1/json?area-type=ctry,W92000004
        area_param = f"{geo_level},{','.join(area_codes)}"
        url = f"{self.base_url}/datasets/{dataset_id}/editions/2021/versions/1/json"
        url += f"?area-type={area_param}"

        logger.debug(f"Request URL: {url}")
        response = requests.get(url)
        response.raise_for_status()

        # Log the full response for debugging
        data = response.json()
        logger.debug(f"Response data keys: {data.keys()}")

        # Check if we have observations
        observations_count = len(data.get('observations', []))
        logger.debug(f"Found {observations_count} observations")

        return data

    def batch_get_dataset_data(self,
                               dataset_id: str,
                               area_codes: List[str],
                               geo_level: str = "ctry",
                               batch_size: int = 100,
                               population_type: str = "UR") -> Dict[str, Any]:
        """
        Retrieve data for a TS dataset in batches to avoid URL length limitations.

        Args:
            dataset_id: The Time Series dataset ID
            area_codes: List of area codes to retrieve data for
            geo_level: Geographic level (default: "ctry")
            batch_size: Maximum number of areas per request
            population_type: The population type (default: "UR")

        Returns:
            Dict containing the combined dataset response
        """
        if not area_codes:
            return {}

        logger.info(f"Fetching TS dataset {dataset_id} for {geo_level} level in batches")

        # For TS datasets, we need to combine observations from multiple batches
        all_observations = []
        dimensions = None
        headers = None

        # Calculate total batches
        num_batches = (len(area_codes) - 1) // batch_size + 1
        total_areas = len(area_codes)

        # Create progress bar
        progress_desc = f"Dataset {dataset_id} | Level {geo_level}"
        with tqdm(total=total_areas, desc=progress_desc, unit="areas") as pbar:
            # Process areas in batches
            for i in range(0, len(area_codes), batch_size):
                batch = area_codes[i:i+batch_size]
                batch_num = i // batch_size + 1

                # Update progress bar description with batch info
                pbar.set_description(f"{progress_desc} | Batch {batch_num}/{num_batches}")

                try:
                    data = self.get_dataset_data(dataset_id, batch, geo_level, population_type)

                    # Extract dimensions from first batch
                    if dimensions is None and 'dimensions' in data:
                        dimensions = data['dimensions']

                    # Extract headers from first batch
                    if headers is None and 'headers' in data:
                        headers = data['headers']

                    # Append observations to the combined list
                    observations = data.get('observations', [])
                    if observations:
                        all_observations.extend(observations)
                        # Use tqdm.write to avoid breaking the progress bar
                        tqdm.write(f"Added {len(observations)} observations from batch {batch_num}")
                    else:
                        tqdm.write(f"No observations found in batch {batch_num}")

                except Exception as e:
                    # Use tqdm.write to avoid breaking the progress bar
                    tqdm.write(f"Error processing batch {batch_num}: {str(e)}")

                # Update progress bar
                pbar.update(len(batch))

        # Combine results into a single response
        combined_data = {
            'dimensions': dimensions or [],
            'observations': all_observations,
            'headers': headers
        }

        logger.info(f"Retrieved {len(all_observations)} total observations for dataset {dataset_id}")
        return combined_data
