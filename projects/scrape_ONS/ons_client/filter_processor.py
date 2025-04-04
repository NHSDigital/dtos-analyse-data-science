import requests
import time
import os
from typing import Optional, Dict, Any, List

from models import (
    FilterResponse,
    FilterSubmitResponse,
    FilterOutputResponse,
    FilterRequest,
    DatasetIdentifier,
    DimensionFilter,
)
from utils import ensure_dir

BASE_URL = "https://api.beta.ons.gov.uk/v1"


class FilterProcessor:
    """
    Class to handle creating, submitting, and processing filter requests to the ONS API.
    """

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}

    def create_filter(self, payload: Dict[str, Any]) -> FilterResponse:
        """
        Create a filter using the ONS API

        Args:
            payload: The filter request payload

        Returns:
            FilterResponse: The filter response containing the filter ID
        """
        filter_url = f"{self.base_url}/filters"
        response = requests.post(filter_url, headers=self.headers, json=payload)
        response.raise_for_status()
        filter_response_data = response.json()
        return FilterResponse(**filter_response_data)

    def submit_filter(self, filter_id: str) -> FilterSubmitResponse:
        """
        Submit a filter to trigger file generation

        Args:
            filter_id: The ID of the filter to submit

        Returns:
            FilterSubmitResponse: The response containing the filter output ID
        """
        submit_url = f"{self.base_url}/filters/{filter_id}/submit"
        submit_response = requests.post(submit_url, headers=self.headers, json={})
        submit_response.raise_for_status()
        submit_data = submit_response.json()
        return FilterSubmitResponse(**submit_data)

    def poll_filter_output(
        self, filter_output_id: str, max_wait: int = 300, wait_interval: int = 10
    ) -> Optional[str]:
        """
        Poll the filter output until the CSV download link is available

        Args:
            filter_output_id: The ID of the filter output to poll
            max_wait: Maximum wait time in seconds
            wait_interval: Seconds between polls

        Returns:
            Optional[str]: The CSV download URL if available, None otherwise
        """
        filter_output_url = f"{self.base_url}/filter-outputs/{filter_output_id}"
        print(f"Polling filter output URL: {filter_output_url}")

        elapsed = 0
        csv_href = None

        while (not csv_href) and (elapsed < max_wait):
            print(
                f"CSV download link not ready yet. Waiting for {wait_interval} seconds..."
            )
            time.sleep(wait_interval)
            elapsed += wait_interval

            poll_response = requests.get(filter_output_url)
            poll_response.raise_for_status()
            filter_response = poll_response.json()

            # Parse response into model
            filter_output = FilterOutputResponse(**filter_response)

            # Check if csv download link is available
            if (
                filter_output.downloads
                and filter_output.downloads.csv
                and filter_output.downloads.csv.href
            ):
                csv_href = filter_output.downloads.csv.href

        return csv_href

    def download_csv(self, csv_url: str, output_file: str) -> str:
        """
        Download the CSV file from the provided URL

        Args:
            csv_url: The URL to download the CSV from
            output_file: The file path to save the CSV to

        Returns:
            str: The path to the saved CSV file
        """
        print(f"Downloading CSV from: {csv_url}")
        csv_response = requests.get(csv_url)
        csv_response.raise_for_status()

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        with open(output_file, "wb") as f:
            f.write(csv_response.content)

        print(f"CSV saved to {output_file}")
        return output_file

    def process_filter(
        self,
        filter_request: FilterRequest,
        dataset_id: str,
        area_type: str,
        area_code: str,
        output_file: Optional[str] = None,
    ) -> str:
        """
        Process a filter request from start to finish (create, submit, poll, download)

        Args:
            filter_request: The filter request to process
            dataset_id: The dataset ID
            area_type: The area type
            area_code: The area code or identifier for this filter
            output_file: Optional path to save the CSV file, generated if None

        Returns:
            str: The path to the downloaded CSV file
        """
        # Convert Pydantic model to dict for API request
        payload = filter_request.model_dump()

        # Create the filter
        filter_response = self.create_filter(payload)
        filter_id = filter_response.filter_id

        # Submit the filter job
        submit_response = self.submit_filter(filter_id)
        filter_output_id = submit_response.filter_output_id

        # Poll for the CSV download link
        csv_href = self.poll_filter_output(filter_output_id)
        if not csv_href:
            raise Exception("Filter job did not complete within expected time.")

        # Generate output file name if not provided
        if output_file is None:
            output_file = f"{dataset_id}_{area_type}_{area_code}.csv"

        # Download the CSV file
        return self.download_csv(csv_href, output_file)

    def create_filter_request(
        self,
        dataset_id: str,
        population_type: str,
        area_type: str,
        area_codes: List[str],
        edition: str = "2021",
        version: int = 1,
    ) -> FilterRequest:
        """
        Create a filter request for the given parameters

        Args:
            dataset_id: The dataset ID
            population_type: The population type
            area_type: The area type
            area_codes: List of area codes to include in the filter
            edition: The dataset edition
            version: The dataset version

        Returns:
            FilterRequest: The constructed filter request
        """
        return FilterRequest(
            dataset=DatasetIdentifier(id=dataset_id, edition=edition, version=version),
            population_type=population_type,
            dimensions=[DimensionFilter(name=area_type, options=area_codes)],
        )
