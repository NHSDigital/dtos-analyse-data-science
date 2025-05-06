import requests
import logging
import time
import random
from typing import Dict, List, Any, Optional, Union, Callable
from functools import wraps
import json
import os
import pandas as pd
import tempfile

from ..models.common import (
    Dataset,
    Dimension,
    DimensionOption,
    DimensionWithOptions,
    DatasetAvailability,
)

logger = logging.getLogger(__name__)


def with_retry(max_retries=3, initial_delay=2.0, backoff_factor=2.0):
    """
    Decorator for adding retry logic with exponential backoff to API methods.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay between retries

    Returns:
        Decorated function with retry logic
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = initial_delay

            while attempt <= max_retries:
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    status_code = (
                        e.response.status_code if hasattr(e, "response") else None
                    )

                    # Handle different HTTP errors
                    if status_code == 429:  # Too Many Requests
                        logger.warning(f"Rate limit exceeded (429): {str(e)}")
                        # Always retry with longer delay for rate limiting
                        retry = True
                        delay = max(
                            delay, 60.0
                        )  # At least 60 seconds for rate limit errors
                    elif status_code in [
                        502,
                        503,
                        504,
                        520,
                        521,
                        522,
                        523,
                        524,
                    ]:  # Server errors
                        logger.warning(f"Server error {status_code}: {str(e)}")
                        retry = attempt < max_retries
                    else:
                        # For other HTTP errors, don't retry after first attempt
                        logger.error(f"HTTP error {status_code}: {str(e)}")
                        retry = False

                    if not retry or attempt >= max_retries:
                        logger.error(f"Giving up after {attempt+1} attempts")
                        raise

                except Exception as e:
                    logger.error(f"Error: {str(e)}")
                    if attempt >= max_retries:
                        logger.error(f"Giving up after {attempt+1} attempts")
                        raise

                # Calculate delay with jitter
                jitter = random.uniform(0.8, 1.2)
                sleep_time = delay * jitter
                logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)

                # Increase delay for next attempt
                delay *= backoff_factor
                attempt += 1

        return wrapper

    return decorator


class ONSApiClient:
    """
    Base client for interacting with the ONS API.

    Handles common functionality like making HTTP requests,
    retrieving datasets, and fetching dimensions.
    """

    BASE_URL = "https://api.beta.ons.gov.uk/v1"

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the ONS API client.

        Args:
            base_url: Override the default base URL for the API
        """
        self.base_url = base_url or self.BASE_URL
        self.session = requests.Session()

    @with_retry(max_retries=3)
    def get_datasets(self) -> List[Dataset]:
        """
        Get all available datasets from the ONS API, including Census datasets.
        Uses pagination and proper discovery instead of hard-coded lists.

        Returns:
            List of Dataset objects
        """
        logger.info("Fetching datasets from the ONS API")
        datasets = []

        # Step 1: Fetch standard datasets from the API endpoint with pagination
        base_url = f"{self.base_url}/datasets"
        page_count = 0
        total_api_datasets = 0

        # Set pagination parameters
        limit = 100  # Get more items per page to reduce number of requests
        offset = 0
        total_count = None  # Will be set from first response

        # Continue fetching until we've got all datasets
        while True:
            page_count += 1
            # Construct URL with pagination parameters
            next_url = f"{base_url}?limit={limit}&offset={offset}"
            # logger.info(f"Fetching datasets page {page_count} from {next_url}")

            try:
                response = requests.get(next_url)
                response.raise_for_status()
                data = response.json()

                # Get total count from first response
                if total_count is None and "total_count" in data:
                    total_count = data["total_count"]
                    # logger.info(f"API reports {total_count} total datasets available")

                # Process current page items
                items_count = 0
                for item in data.get("items", []):
                    dataset = Dataset(
                        id=item.get("id", ""),
                        title=item.get("title", ""),
                        description=item.get("description", ""),
                    )
                    datasets.append(dataset)
                    total_api_datasets += 1
                    items_count += 1

                # logger.info(f"Retrieved {items_count} datasets from page {page_count}")

                # Update offset for next page
                offset += items_count

                # Check if we've retrieved all datasets
                if total_count is not None and total_api_datasets >= total_count:
                    logger.info(f"Retrieved all {total_api_datasets} datasets")
                    break

                # If we got fewer items than the limit, we've reached the end
                if items_count < limit:
                    logger.info(
                        f"No more datasets to fetch (received {items_count} < limit {limit})"
                    )
                    break

                # Add a small delay to avoid hitting rate limits
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error fetching datasets page {page_count}: {str(e)}")
                break

        logger.info(
            f"Retrieved {total_api_datasets} standard datasets from {page_count} pages"
        )

        # Step 2: Discover Census datasets through population-types API
        try:
            logger.info("Discovering Census datasets through population-types API")

            # Get population types (e.g., "UR" for usual residents)
            pop_url = f"{self.base_url}/population-types"
            logger.info(f"Fetching population types from {pop_url}")
            pop_response = requests.get(pop_url)
            pop_response.raise_for_status()
            pop_data = pop_response.json()

            # Process each population type
            pop_types = []
            for item in pop_data.get("items", []):
                pop_id = item.get("id")
                if pop_id:
                    pop_types.append(pop_id)

            logger.info(
                f"Found {len(pop_types)} population types: {', '.join(pop_types)}"
            )

            # For each population type, find dimensions which can indicate Census datasets
            census_datasets = []
            for pop_type in pop_types:
                logger.info(f"Checking dimensions for population type: {pop_type}")

                # Get dimensions for this population type
                dim_url = f"{self.base_url}/population-types/{pop_type}/dimensions"
                dim_response = requests.get(dim_url)

                if dim_response.status_code == 200:
                    dim_data = dim_response.json()
                    for dim in dim_data.get("items", []):
                        # Find any dimension link that includes a dataset ID
                        for link_type, link_data in dim.get("links", {}).items():
                            href = link_data.get("href", "")

                            # Extract dataset ID if it matches Census format (TS or RM)
                            # Example: "/datasets/TS008/editions/2021/versions/1/..."
                            if "/datasets/" in href:
                                parts = href.split("/")
                                idx = (
                                    parts.index("datasets")
                                    if "datasets" in parts
                                    else -1
                                )

                                if idx >= 0 and idx + 1 < len(parts):
                                    dataset_id = parts[idx + 1]
                                    if (
                                        dataset_id.startswith("TS")
                                        or dataset_id.startswith("RM")
                                    ) and dataset_id not in census_datasets:
                                        census_datasets.append(dataset_id)

                # Small delay to avoid hitting rate limits
                time.sleep(0.5)

            # For discovered Census dataset IDs, get metadata
            logger.info(f"Found {len(census_datasets)} Census dataset IDs")

            for dataset_id in census_datasets:
                # Check if this dataset ID is already in our list
                if not any(ds.id == dataset_id for ds in datasets):
                    # Try to get metadata for this dataset
                    try:
                        meta_url = f"{self.base_url}/datasets/{dataset_id}/editions/2021/versions/1"
                        meta_response = requests.get(meta_url)

                        if meta_response.status_code == 200:
                            meta_data = meta_response.json()

                            # Create dataset object with metadata
                            dataset = Dataset(
                                id=dataset_id,
                                title=meta_data.get(
                                    "title", f"Census 2021 - {dataset_id}"
                                ),
                                description=meta_data.get("description", ""),
                            )
                            datasets.append(dataset)
                            logger.debug(f"Added Census dataset: {dataset_id}")
                        else:
                            # If metadata fails, add with basic info
                            dataset = Dataset(
                                id=dataset_id,
                                title=f"Census 2021 - {dataset_id}",
                                description="Census 2021 dataset",
                            )
                            datasets.append(dataset)
                            logger.debug(
                                f"Added Census dataset with basic info: {dataset_id}"
                            )

                    except Exception as e:
                        logger.error(
                            f"Error fetching metadata for {dataset_id}: {str(e)}"
                        )
                        # Still add with basic info
                        dataset = Dataset(
                            id=dataset_id,
                            title=f"Census 2021 - {dataset_id}",
                            description="Census 2021 dataset",
                        )
                        datasets.append(dataset)

                # Small delay to avoid hitting rate limits
                time.sleep(0.2)

            logger.info(f"Added {len(census_datasets)} Census datasets")

            # Step 3: Also check for Census datasets using census-observations endpoint
            # This is another route to discover Census datasets
            try:
                logger.info("Checking census-observations for additional datasets")

                # Check UR population type for census observations
                obs_url = (
                    f"{self.base_url}/population-types/UR/census-observations?limit=10"
                )
                obs_response = requests.get(obs_url)

                if obs_response.status_code == 200:
                    obs_data = obs_response.json()

                    # Look for dataset references in the response
                    for dataset_link in obs_data.get("dataset_links", []):
                        href = dataset_link.get("href", "")

                        # Extract dataset ID from link
                        if "/datasets/" in href:
                            parts = href.split("/")
                            idx = parts.index("datasets") if "datasets" in parts else -1

                            if idx >= 0 and idx + 1 < len(parts):
                                dataset_id = parts[idx + 1]

                                # Only add if not already in our list
                                if (
                                    dataset_id.startswith("TS")
                                    or dataset_id.startswith("RM")
                                ) and not any(ds.id == dataset_id for ds in datasets):
                                    dataset = Dataset(
                                        id=dataset_id,
                                        title=f"Census 2021 - {dataset_id}",
                                        description="Census 2021 dataset discovered via census-observations",
                                    )
                                    datasets.append(dataset)
                                    logger.debug(
                                        f"Added Census dataset from observations: {dataset_id}"
                                    )

            except Exception as e:
                logger.error(f"Error checking census-observations: {str(e)}")

        except Exception as e:
            logger.error(f"Error discovering Census datasets: {str(e)}")
            logger.debug(f"Full error: {str(e)}")

        # Sort datasets by ID for consistent output
        datasets.sort(key=lambda x: x.id)

        logger.info(f"Total datasets: {len(datasets)}")
        return datasets

    @with_retry(max_retries=3)
    def get_dimensions(self, population_type: str = "UR") -> List[DimensionWithOptions]:
        """
        Get dimensions for the specified population type.

        Args:
            population_type: The population type (default: "UR")

        Returns:
            List of Dimension objects with their options
        """
        url = f"{self.base_url}/population-types/{population_type}/dimensions"
        # logger.info(f"Fetching dimensions for population type {population_type}")

        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        dimensions = []

        for item in data.get("items", []):
            # Fetch options for this dimension
            dimension_id = item.get("id", "")
            options = self.get_dimension_options(population_type, dimension_id)

            dimension = DimensionWithOptions(
                id=dimension_id, label=item.get("label", ""), options=options
            )
            dimensions.append(dimension)

        # logger.info(f"Retrieved {len(dimensions)} dimensions")
        return dimensions

    @with_retry(max_retries=3)
    def get_dimension_options(
        self, population_type: str, dimension_id: str
    ) -> List[DimensionOption]:
        """
        Get options for a specific dimension.

        Args:
            population_type: The population type
            dimension_id: The dimension ID

        Returns:
            List of DimensionOption objects
        """
        url = f"{self.base_url}/population-types/{population_type}/dimensions/{dimension_id}/options"
        # logger.debug(f"Fetching options for dimension {dimension_id}")

        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        options = []

        for item in data.get("items", []):
            option = DimensionOption(id=item.get("id", ""), label=item.get("label", ""))
            options.append(option)

        # logger.debug(f"Retrieved {len(options)} options for dimension {dimension_id}")
        return options

    @with_retry(max_retries=3)
    def get_areas_for_level(
        self, geo_level: str, population_type: str = "UR"
    ) -> List[Dict[str, str]]:
        """
        Get all areas for a specific geographic level.

        Args:
            geo_level: Geographic level (e.g., ctry, rgn, la, msoa, lsoa, oa)
            population_type: Population type (default: "UR")

        Returns:
            List of area dictionaries with id and label
        """
        # Use the correct URL format based on original implementation
        url = f"{self.base_url}/population-types/{population_type}/area-types/{geo_level}/areas"
        logger.info(f"Fetching areas for level {geo_level}")

        # First check if we have a cached version
        cache_dir = "cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_filename = os.path.join(
            cache_dir, f"areas_{population_type}_{geo_level}.json"
        )

        # If cache exists, use it
        if os.path.exists(cache_filename):
            logger.info(f"Using cached areas from {cache_filename}")
            try:
                with open(cache_filename, "r", encoding="utf-8") as f:
                    data = json.load(f)
                areas = []
                for item in data:
                    areas.append(
                        {"id": item.get("id", ""), "label": item.get("label", "")}
                    )
                logger.info(
                    f"Retrieved {len(areas)} areas for level {geo_level} from cache"
                )
                return areas
            except Exception as e:
                logger.warning(f"Error reading cache, will fetch from API: {e}")

        # If no cache or error, fetch from API
        areas = []

        # Handle pagination using _paginate_get method
        next_page = url
        while next_page:
            logger.debug(f"Fetching page: {next_page}")
            response = requests.get(next_page)
            response.raise_for_status()

            data = response.json()
            items = data.get("items", [])

            for item in items:
                area = {"id": item.get("id", ""), "label": item.get("label", "")}
                areas.append(area)

            # Check for next page
            next_page = data.get("links", {}).get("next", {}).get("href", "")

            # Add a small delay between pages to avoid rate limiting
            if next_page:
                time.sleep(0.5)

        logger.info(f"Retrieved {len(areas)} areas for level {geo_level}")

        # Save to cache for future use
        try:
            with open(cache_filename, "w", encoding="utf-8") as f:
                json.dump(areas, f, indent=2)
            logger.info(f"Saved areas to cache: {cache_filename}")
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")

        return areas

    def get_dataset_using_filter(
        self,
        dataset_id: str,
        geo_level: str,
        output_dir: str,
        area_codes: List[str] = None,
        population_type: str = "UR",
        max_poll_attempts: int = 30,
        poll_interval: int = 5,
    ) -> Optional[str]:
        """
        Retrieve dataset using the filter API for large datasets.

        Args:
            dataset_id: The dataset ID (e.g., TS030)
            geo_level: Geographic level (e.g., lsoa, msoa)
            output_dir: Directory to save the output file
            area_codes: List of area codes or None for all areas
            population_type: Population type (default: "UR")
            max_poll_attempts: Maximum number of attempts to poll for filter completion
            poll_interval: Time in seconds between poll attempts

        Returns:
            Path to the downloaded file or None if failed
        """
        from .filter_client import ONSFilterClient
        import os
        import time
        import pandas as pd
        import tempfile
        import json

        logger.info(f"Using filter API to get {dataset_id} for level {geo_level}")

        # Create filter client
        filter_client = ONSFilterClient(self.base_url)

        # If no area codes provided, get all areas for this level
        if not area_codes:
            areas = self.get_areas_for_level(geo_level, population_type)
            area_codes = [area["id"] for area in areas]
            logger.info(f"Retrieved {len(area_codes)} areas for level {geo_level}")

        # Define the maximum batch size for area codes
        # The ONS API seems to have issues with payloads larger than ~10,000 areas
        max_batch_size = 5000

        # Final output file path
        final_output_file = os.path.join(output_dir, f"{dataset_id}_{geo_level}.csv")

        # If the area list is too large, split into batches
        if len(area_codes) > max_batch_size:
            logger.info(f"Large area list detected ({len(area_codes)} areas), splitting into batches of {max_batch_size}")

            # Split area codes into batches
            batches = [area_codes[i:i + max_batch_size] for i in range(0, len(area_codes), max_batch_size)]
            logger.info(f"Split into {len(batches)} batches")

            # Create a list to store temp files with batch results
            batch_files = []
            batch_debug_files = []

            # Process each batch
            for batch_num, batch_areas in enumerate(batches, 1):
                logger.info(f"Processing batch {batch_num}/{len(batches)} with {len(batch_areas)} areas")

                try:
                    # Create a temporary file for this batch
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                        batch_output_file = temp_file.name

                    # Process this batch
                    batch_result = self._process_filter_batch(
                        filter_client,
                        dataset_id,
                        geo_level,
                        batch_areas,
                        batch_output_file,
                        population_type,
                        max_poll_attempts,
                        poll_interval
                    )

                    if batch_result:
                        batch_files.append(batch_result)
                        # Add debug file to list if it exists
                        debug_file = f"{batch_result}.debug.json"
                        if os.path.exists(debug_file):
                            batch_debug_files.append(debug_file)
                    else:
                        logger.error(f"Failed to process batch {batch_num}")
                except Exception as e:
                    logger.error(f"Error processing batch {batch_num}: {str(e)}")
                    import traceback
                    logger.debug(traceback.format_exc())

            # Combine all batch results if we have any
            if batch_files:
                logger.info(f"Combining {len(batch_files)} batch results")

                # Use pandas to read and combine all CSV files
                dfs = []
                for file in batch_files:
                    try:
                        df = pd.read_csv(file)
                        dfs.append(df)
                    except Exception as e:
                        logger.warning(f"Error reading batch file {file}: {str(e)}")

                if dfs:
                    # Combine all dataframes
                    combined_df = pd.concat(dfs, ignore_index=True)

                    # Write combined results to the final output file
                    combined_df.to_csv(final_output_file, index=False)
                    logger.info(f"Saved combined results to {final_output_file}")

                    # Combine debug files for dimension information
                    combined_debug = {
                        "dataset_id": dataset_id,
                        "geo_level": geo_level,
                        "population_type": population_type,
                        "_area_metadata": {
                            "geo_level": geo_level,
                            "area_codes": area_codes
                        }
                    }

                    # Find a debug file with dimensions
                    dimension_data = None
                    for debug_file in batch_debug_files:
                        try:
                            with open(debug_file, 'r', encoding='utf-8') as f:
                                debug_data = json.load(f)
                                if debug_data.get("dimensions") and (dimension_data is None or debug_data.get("sample_response", False)):
                                    dimension_data = debug_data
                                    logger.info(f"Using dimension data from {debug_file}")
                                    if debug_data.get("sample_response", False):
                                        # If we found a sample response, no need to check other files
                                        break
                        except Exception as e:
                            logger.warning(f"Error reading debug file {debug_file}: {str(e)}")

                    # If we found dimension data, add it to the combined debug
                    if dimension_data:
                        combined_debug["dimensions"] = dimension_data.get("dimensions", [])
                        combined_debug["sample_response"] = dimension_data.get("sample_response", False)

                    # Save combined debug information
                    combined_debug_file = f"{final_output_file}.debug.json"
                    try:
                        with open(combined_debug_file, 'w', encoding='utf-8') as f:
                            json.dump(combined_debug, f, indent=2)
                        logger.info(f"Saved combined debug information to {combined_debug_file}")
                    except Exception as e:
                        logger.warning(f"Error saving combined debug file: {str(e)}")

                    # Clean up temporary files
                    for file in batch_files:
                        try:
                            os.remove(file)
                            # Also remove debug file if it exists
                            debug_file = f"{file}.debug.json"
                            if os.path.exists(debug_file):
                                os.remove(debug_file)
                        except Exception as e:
                            logger.warning(f"Failed to remove temp file {file}: {str(e)}")

                    return final_output_file
                else:
                    logger.error("No valid batch results to combine")
                    return None
            else:
                logger.error("All batches failed to process")
                return None
        else:
            # For smaller area lists, process directly
            return self._process_filter_batch(
                filter_client,
                dataset_id,
                geo_level,
                area_codes,
                final_output_file,
                population_type,
                max_poll_attempts,
                poll_interval
            )

    def _process_filter_batch(
        self,
        filter_client,
        dataset_id: str,
        geo_level: str,
        area_codes: List[str],
        output_file: str,
        population_type: str = "UR",
        max_poll_attempts: int = 30,
        poll_interval: int = 5,
    ) -> Optional[str]:
        """
        Process a single batch of area codes using the filter API.

        Args:
            filter_client: Instance of ONSFilterClient
            dataset_id: The dataset ID (e.g., TS030)
            geo_level: Geographic level (e.g., lsoa, msoa)
            area_codes: List of area codes for this batch
            output_file: Path to save the output file
            population_type: Population type (default: "UR")
            max_poll_attempts: Maximum number of attempts to poll for filter completion
            poll_interval: Time in seconds between poll attempts

        Returns:
            Path to the downloaded file or None if failed
        """
        import time

        try:
            # Create filter
            filter_response = filter_client.create_filter(
                dataset_id=dataset_id,
                population_type=population_type,
                geo_level=geo_level,
                area_codes=area_codes
            )
            filter_id = filter_response["filter_id"]
            logger.info(f"Created filter with ID: {filter_id}")

            # Submit filter for processing
            submit_response = filter_client.submit_filter(filter_id)
            filter_output_id = submit_response["filter_output_id"]
            logger.info(f"Filter submitted, output ID: {filter_output_id}")

            # Poll for filter completion and get download URLs
            for attempt in range(max_poll_attempts):
                logger.info(f"Checking filter status (attempt {attempt+1}/{max_poll_attempts})")
                filter_output = filter_client.get_filter_output(filter_output_id)

                # Check if CSV download is available
                downloads = filter_output.get("downloads", {})
                csv_info = downloads.get("csv", {})
                csv_url = csv_info.get("href") if csv_info else None

                if csv_url:
                    logger.info("CSV download available, proceeding with download")
                    filter_client.download_filter_output(csv_url, output_file)
                    logger.info(f"Downloaded data to {output_file}")

                    # Save dimension information in a debug file for later processing
                    debug_file = f"{output_file}.debug.json"
                    debug_data = {
                        "dataset_id": dataset_id,
                        "geo_level": geo_level,
                        "population_type": population_type,
                        "dimensions": filter_output.get("dimensions", []),
                        "_area_metadata": {
                            "geo_level": geo_level,
                            "area_codes": area_codes
                        }
                    }

                    # Fetch a small sample of the data to get dimension information
                    # This is needed because the filter API doesn't include dimension breakdown
                    try:
                        if dataset_id.startswith("TS"):
                            from ..api.ts_client import TSApiClient
                            sample_client = TSApiClient(self.base_url)
                            # Get sample for first 5 areas or fewer
                            sample_areas = area_codes[:5]
                            sample_response = sample_client.get_dataset_data_for_areas(
                                dataset_id, geo_level, sample_areas, population_type
                            )
                            if sample_response:
                                # Add dimensions from sample response
                                debug_data["dimensions"] = sample_response.get("dimensions", [])
                                debug_data["sample_response"] = True
                        elif dataset_id.startswith("RM"):
                            from ..api.rm_client import RMApiClient
                            sample_client = RMApiClient(self.base_url)
                            # Get sample for first 5 areas or fewer
                            sample_areas = area_codes[:5]
                            sample_response = sample_client.get_dataset_data_for_areas(
                                dataset_id, geo_level, sample_areas, population_type
                            )
                            if sample_response:
                                # Add dimensions from sample response
                                debug_data["dimensions"] = sample_response.get("dimensions", [])
                                debug_data["sample_response"] = True
                    except Exception as e:
                        logger.warning(f"Failed to get dimension sample: {str(e)}")

                    # Save the debug information
                    import json
                    with open(debug_file, "w", encoding="utf-8") as f:
                        json.dump(debug_data, f, indent=2)

                    return output_file

                # If not ready, wait and try again
                logger.info(f"Filter still processing, waiting {poll_interval} seconds")
                time.sleep(poll_interval)

            logger.error("Filter did not complete within the expected time")
            return None

        except Exception as e:
            logger.error(f"Error using filter API: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _make_request(
        self, endpoint: str, params: Dict = None, max_retries: int = 3
    ) -> Dict:
        """Make a request to the ONS API with retry logic.

        Args:
            endpoint: API endpoint to call.
            params: Query parameters.
            max_retries: Maximum number of retries.

        Returns:
            Response JSON as dictionary.
        """
        url = f"{self.base_url}{endpoint}"
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = self.session.get(url, params=params)

                if response.status_code == 429:  # Too Many Requests
                    # Implement exponential backoff
                    wait_time = 2**retry_count
                    logger.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue

                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(1)  # Simple delay between retries
                else:
                    raise

        raise Exception(f"Failed to get data after {max_retries} retries")

    def get_area_types(self, population_type: str = "UR") -> List[Dict]:
        """Get available area types for a population type.

        Args:
            population_type: Population type.

        Returns:
            List of area types.
        """
        endpoint = f"/population-types/{population_type}/area-types"
        response = self._make_request(endpoint)
        return response.get("items", [])

    def get_areas(self, population_type: str, area_type: str) -> List[Dict]:
        """Get areas for a specific area type.

        Args:
            population_type: Population type.
            area_type: Area type.

        Returns:
            List of areas.
        """
        endpoint = f"/population-types/{population_type}/area-types/{area_type}/areas"
        response = self._make_request(endpoint)
        return response.get("items", [])

    def check_dataset_availability(
        self, dataset_id: str, geo_level: str, population_type: str = "UR"
    ) -> DatasetAvailability:
        """Check if a dataset is available at a specific geographic level.

        Args:
            dataset_id: Dataset ID to check.
            geo_level: Geographic level to check.
            population_type: Population type.

        Returns:
            DatasetAvailability object with availability information.
        """
        result = DatasetAvailability(
            dataset_id=dataset_id,
            geo_level=geo_level,
            population_type=population_type,
            is_available=False,
        )

        try:
            # Get a single area of the specified geographic level
            areas = self.get_areas(population_type, geo_level)
            if not areas:
                result.error_message = (
                    f"No areas found for geographic level: {geo_level}"
                )
                logger.warning(result.error_message)
                return result

            # Use the first area to test availability
            test_area = areas[0]
            area_code = test_area.get("id")
            area_label = test_area.get("label", "Unknown")
            logger.info(
                f"Testing dataset {dataset_id} availability at {geo_level} level using area: {area_label} ({area_code})"
            )

            # Make a test request to check if data is available
            endpoint = f"/datasets/{dataset_id}/editions/2021/versions/1/json"
            params = {"area-type": f"{geo_level},{area_code}"}

            try:
                response = self._make_request(endpoint, params)

                # Check if the response contains observations with data
                if response and "observations" in response and response["observations"]:
                    result.is_available = True
                    logger.info(
                        f"Dataset {dataset_id} is available at {geo_level} level"
                    )
                else:
                    result.error_message = f"Dataset {dataset_id} doesn't contain observations at {geo_level} level"
                    logger.warning(result.error_message)
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if hasattr(e, "response") else None
                if status_code == 404:
                    result.error_message = (
                        f"Dataset {dataset_id} not found at {geo_level} level (404)"
                    )
                else:
                    result.error_message = f"HTTP error {status_code} checking dataset {dataset_id} at {geo_level} level"
                logger.warning(result.error_message)
            except Exception as e:
                result.error_message = f"Error requesting dataset: {str(e)}"
                logger.warning(result.error_message)
        except Exception as e:
            result.error_message = f"Error checking dataset availability: {str(e)}"
            logger.warning(result.error_message)

        return result
