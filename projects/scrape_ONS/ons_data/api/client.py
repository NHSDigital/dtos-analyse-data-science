import requests
import logging
import time
import random
from typing import Dict, List, Any, Optional, Union, Callable
from functools import wraps
import json
import os

from ..models.common import Dataset, Dimension, DimensionOption, DimensionWithOptions

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
                    status_code = e.response.status_code if hasattr(e, 'response') else None

                    # Handle different HTTP errors
                    if status_code == 429:  # Too Many Requests
                        logger.warning(f"Rate limit exceeded (429): {str(e)}")
                        # Always retry with longer delay for rate limiting
                        retry = True
                        delay = max(delay, 60.0)  # At least 60 seconds for rate limit errors
                    elif status_code in [502, 503, 504, 520, 521, 522, 523, 524]:  # Server errors
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

    @with_retry(max_retries=3)
    def get_datasets(self) -> List[Dataset]:
        """
        Get all available datasets from the ONS API.

        Returns:
            List of Dataset objects
        """
        url = f"{self.base_url}/datasets"
        logger.info(f"Fetching datasets from {url}")

        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        datasets = []

        for item in data.get("items", []):
            dataset = Dataset(
                id=item.get("id", ""),
                title=item.get("title", ""),
                description=item.get("description", "")
            )
            datasets.append(dataset)

        logger.info(f"Retrieved {len(datasets)} datasets")
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
        logger.info(f"Fetching dimensions for population type {population_type}")

        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        dimensions = []

        for item in data.get("items", []):
            # Fetch options for this dimension
            dimension_id = item.get("id", "")
            options = self.get_dimension_options(population_type, dimension_id)

            dimension = DimensionWithOptions(
                id=dimension_id,
                label=item.get("label", ""),
                options=options
            )
            dimensions.append(dimension)

        logger.info(f"Retrieved {len(dimensions)} dimensions")
        return dimensions

    @with_retry(max_retries=3)
    def get_dimension_options(self, population_type: str, dimension_id: str) -> List[DimensionOption]:
        """
        Get options for a specific dimension.

        Args:
            population_type: The population type
            dimension_id: The dimension ID

        Returns:
            List of DimensionOption objects
        """
        url = f"{self.base_url}/population-types/{population_type}/dimensions/{dimension_id}/options"
        logger.debug(f"Fetching options for dimension {dimension_id}")

        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        options = []

        for item in data.get("items", []):
            option = DimensionOption(
                id=item.get("id", ""),
                label=item.get("label", "")
            )
            options.append(option)

        logger.debug(f"Retrieved {len(options)} options for dimension {dimension_id}")
        return options

    @with_retry(max_retries=3)
    def get_areas_for_level(self, geo_level: str, population_type: str = "UR") -> List[Dict[str, str]]:
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
        cache_filename = os.path.join(cache_dir, f"areas_{population_type}_{geo_level}.json")

        # If cache exists, use it
        if os.path.exists(cache_filename):
            logger.info(f"Using cached areas from {cache_filename}")
            try:
                with open(cache_filename, "r", encoding='utf-8') as f:
                    data = json.load(f)
                areas = []
                for item in data:
                    areas.append({
                        "id": item.get("id", ""),
                        "label": item.get("label", "")
                    })
                logger.info(f"Retrieved {len(areas)} areas for level {geo_level} from cache")
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
                area = {
                    "id": item.get("id", ""),
                    "label": item.get("label", "")
                }
                areas.append(area)

            # Check for next page
            next_page = data.get("links", {}).get("next", {}).get("href", "")

            # Add a small delay between pages to avoid rate limiting
            if next_page:
                time.sleep(0.5)

        logger.info(f"Retrieved {len(areas)} areas for level {geo_level}")

        # Save to cache for future use
        try:
            with open(cache_filename, "w", encoding='utf-8') as f:
                json.dump(areas, f, indent=2)
            logger.info(f"Saved areas to cache: {cache_filename}")
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")

        return areas
