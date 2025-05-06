import logging
import time
import os
import requests
from typing import Dict, List, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)


def with_retry_and_backoff(max_retries=5, initial_delay=2.0, backoff_factor=2.0):
    """Decorator for retry with exponential backoff, with handling for 429 and 413 status codes."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    last_exception = e
                    status_code = e.response.status_code

                    # Handle rate limiting (429)
                    if status_code == 429:
                        retry_after = int(e.response.headers.get('Retry-After', delay))
                        logger.warning(f"Rate limited (429). Waiting {retry_after} seconds.")
                        time.sleep(retry_after)
                    # Handle payload too large (413)
                    elif status_code == 413:
                        logger.warning(f"Request failed: 413 Payload Too Large. The area list is too large for the API.")
                        if attempt < max_retries - 1:
                            logger.warning(f"Retrying in {delay:.1f}s. Consider reducing the batch size.")
                            time.sleep(delay)
                            delay *= backoff_factor
                        else:
                            logger.error("Maximum retries reached for payload too large error.")
                            logger.error("Recommendation: Split the area list into smaller batches of 5000 or fewer areas.")
                            raise RuntimeError("Payload too large. Split into smaller batches.") from e
                    else:
                        # For other errors, use exponential backoff
                        logger.warning(f"Request failed: {str(e)}. Retrying in {delay:.1f}s")
                        time.sleep(delay)
                        delay *= backoff_factor
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Error: {str(e)}. Retrying in {delay:.1f}s")
                    time.sleep(delay)
                    delay *= backoff_factor

            # If we've exhausted retries, raise the last exception
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


class ONSFilterClient:
    """Client for creating and managing filters for ONS data downloads."""

    def __init__(self, base_url="https://api.beta.ons.gov.uk/v1"):
        self.base_url = base_url
        self.session = requests.Session()

    @with_retry_and_backoff(max_retries=5)
    def create_filter(self, dataset_id, edition="2021", version=1,
                      population_type="UR", geo_level=None, area_codes=None):
        """
        Create a new filter for a dataset.

        Args:
            dataset_id: The dataset ID (e.g., TS030)
            edition: Dataset edition (default: "2021")
            version: Dataset version (default: 1)
            population_type: Population type (default: "UR")
            geo_level: Geographic level (e.g., lsoa, msoa)
            area_codes: List of area codes

        Returns:
            Dict containing the filter creation response
        """
        # Prepare filter request payload
        filter_payload = {
            "dataset": {
                "id": dataset_id,
                "edition": edition,
                "version": version
            },
            "population_type": population_type,
            "dimensions": []
        }

        # Add geographic dimension if specified
        if geo_level and area_codes:
            geo_dimension = {
                "name": geo_level,
                "is_area_type": True,
                "options": area_codes
            }
            filter_payload["dimensions"].append(geo_dimension)

        # Create the filter
        url = f"{self.base_url}/filters"
        logger.info(f"Creating filter for dataset {dataset_id} at level {geo_level}")
        logger.debug(f"Filter payload: {filter_payload}")

        response = self.session.post(url, json=filter_payload)
        response.raise_for_status()

        return response.json()

    @with_retry_and_backoff(max_retries=5)
    def submit_filter(self, filter_id):
        """
        Submit a filter for processing.

        Args:
            filter_id: ID of the filter to submit

        Returns:
            Dict containing the filter submission response
        """
        url = f"{self.base_url}/filters/{filter_id}/submit"
        logger.info(f"Submitting filter with ID: {filter_id}")

        response = self.session.post(url)
        response.raise_for_status()

        return response.json()

    @with_retry_and_backoff(max_retries=5)
    def get_filter_output(self, filter_output_id):
        """
        Get information about a filter output, including download URLs.

        Args:
            filter_output_id: ID of the filter output

        Returns:
            Dict containing the filter output information
        """
        url = f"{self.base_url}/filter-outputs/{filter_output_id}"
        logger.debug(f"Checking filter output with ID: {filter_output_id}")

        response = self.session.get(url)
        response.raise_for_status()

        return response.json()

    @with_retry_and_backoff(max_retries=5)
    def download_filter_output(self, download_url, output_file):
        """
        Download a filter output file.

        Args:
            download_url: URL to download the filter output
            output_file: Path to save the output file

        Returns:
            Path to the downloaded file
        """
        logger.info(f"Downloading filter output from: {download_url}")

        response = self.session.get(download_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 8192

        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)

        logger.info(f"Downloaded filter output to: {output_file}")
        return output_file
