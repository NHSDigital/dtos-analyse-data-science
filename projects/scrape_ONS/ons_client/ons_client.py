from typing import List, Optional, Dict, Any, TypeVar, Generic, Callable
import requests
import json
import os
import time
from pydantic import BaseModel

from .models import (
    Dataset, DatasetsResponse, PopulationType, AreaType,
    Area, Dimension, Categorisation, CensusObservationsResponse,
    DimensionsResponse, DimensionOptionsResponse, DimensionOption, DatasetMetadata
)
from .utils import retry_with_backoff

T = TypeVar('T')

class ONSApiClient:
    def __init__(self, base_url: str = "https://api.beta.ons.gov.uk/v1") -> None:
        """Initialize the API client with a base URL and a requests session."""
        self.base_url = base_url
        self.session = requests.Session()
        # Add a custom User-Agent header to identify the bot and improve rate limit handling
        self.session.headers.update({
            "User-Agent": "ONSDataBot/1.0.0 (NHS-Code-Project +https://github.com/yourusername/NHS_CODE)"
        })

    def _paginate_get(self, url: str, params: Optional[dict] = None, item_key: Optional[str] = None) -> List:
        """Retrieve all pages for a GET request with retry on 429 errors, respecting the Retry-After header."""
        results = []
        if params is None:
            params = {}
        if "offset" not in params:
            params["offset"] = 0
        max_retries = 5
        while True:
            retries = 0
            while retries < max_retries:
                response = self.session.get(url, params=params)
                if response.status_code == 429:
                    # Use the 'Retry-After' header if available
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        wait_time = int(retry_after)
                    else:
                        wait_time = 5 * (retries + 1)  # exponential backoff fallback
                    print(f"Received 429 Too Many Requests for URL: {url}. Waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    break
            if retries == max_retries:
                response.raise_for_status()
            response.raise_for_status()
            data = response.json()
            page_items = []
            if isinstance(data, dict):
                if item_key and item_key in data:
                    page_items = data[item_key]
                else:
                    page_items = []
            elif isinstance(data, list):
                page_items = data
            results.extend(page_items)
            if isinstance(data, dict) and "offset" in data and "limit" in data:
                offset = data["offset"]
                limit = data["limit"]
                total = data.get("total_count") or data.get("total_observations")
                if total is None:
                    if len(page_items) < limit:
                        break
                else:
                    if offset + limit >= total:
                        break
                params["offset"] = offset + limit
            else:
                break
        return results

    def get_datasets(self) -> DatasetsResponse:
        """Retrieve a list of datasets from the ONS Developer Hub.

        Returns:
            DatasetsResponse: A pydantic model containing a list of Dataset objects.
        """
        url = f"{self.base_url}/datasets"
        items = self._paginate_get(url, params={}, item_key="items")
        return DatasetsResponse(items=items)

    def get_population_types(self) -> List[PopulationType]:
        """Retrieve available population types.

        Returns:
            List[PopulationType]: A list of population types available.
        """
        url = f"{self.base_url}/population-types"
        items = self._paginate_get(url)
        return [PopulationType(**item) for item in items]

    def get_area_types(self, population_type_name: str) -> List[AreaType]:
        """Retrieve area types for a given population type.

        Args:
            population_type_name (str): The name of the population type.

        Returns:
            List[AreaType]: A list of area types for the specified population type.
        """
        url = f"{self.base_url}/population-types/{population_type_name}/area-types"
        items = self._paginate_get(url, params={}, item_key="items")
        return [AreaType(**item) for item in items]

    def get_areas(self, population_type_name: str, area_type_id: str, q: Optional[str] = None) -> List[Area]:
        """Retrieve areas for a given population type and area type. Optionally filter areas with a query parameter.

        Args:
            population_type_name (str): The name of the population type.
            area_type_id (str): The identifier of the area type.
            q (Optional[str], optional): An optional query parameter for filtering areas. Defaults to None.

        Returns:
            List[Area]: A list of areas matching the criteria.
        """
        url = f"{self.base_url}/population-types/{population_type_name}/area-types/{area_type_id}/areas"
        params = {}
        if q:
            params["q"] = q
        items = self._paginate_get(url, params=params, item_key="items")
        return [Area(**item) for item in items]

    def get_cached_areas(self, population_type_name: str, area_type_id: str, q: Optional[str] = None) -> List[Area]:
        """Retrieve areas with caching for better performance and reduced API calls.

        Args:
            population_type_name (str): The name of the population type.
            area_type_id (str): The identifier of the area type.
            q (Optional[str], optional): An optional query parameter for filtering areas. Defaults to None.

        Returns:
            List[Area]: A list of areas matching the criteria, either from cache or fresh from the API.
        """
        cache_dir = "cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_filename = os.path.join(cache_dir, f"areas_{population_type_name}_{area_type_id}.json")
        if os.path.exists(cache_filename):
            with open(cache_filename, "r", encoding='utf-8') as f:
                data = json.load(f)
            # Convert the list of dictionaries to list of Area objects
            return [Area(**item) for item in data]
        else:
            areas = self.get_areas(population_type_name, area_type_id, q)
            with open(cache_filename, "w", encoding='utf-8') as f:
                json.dump([area.model_dump() for area in areas], f, indent=2)
            return areas

    def get_dimensions(self, population_type_name: str, q: Optional[str] = None) -> List[Dimension]:
        """Retrieve dimensions for a given population type. Can optionally filter dimensions using a query.

        Args:
            population_type_name (str): The name of the population type.
            q (Optional[str], optional): An optional query to filter dimensions. Defaults to None.

        Returns:
            List[Dimension]: A list of dimensions available.
        """
        url = f"{self.base_url}/population-types/{population_type_name}/dimensions"
        params = {}
        if q:
            params["q"] = q
        items = self._paginate_get(url, params=params, item_key="items")
        print(items)
        return [Dimension(**item) for item in items]

    def get_categorisations(self, population_type_name: str, dimension_id: str) -> List[Categorisation]:
        """Retrieve the categorisations for a specific dimension for a given population type.

        Args:
            population_type_name (str): The name of the population type.
            dimension_id (str): The dimension identifier.

        Returns:
            List[Categorisation]: A list of categorisations for the dimension.
        """
        url = f"{self.base_url}/population-types/{population_type_name}/dimensions/{dimension_id}/categorisations"
        items = self._paginate_get(url)
        return [Categorisation(**item) for item in items]

    def get_census_observations(self, population_type_name: str, area_type: str, areas: str, dimensions: str) -> CensusObservationsResponse:
        """Retrieve observations for a custom Census dataset query.

        For example, the endpoint:
        /population-types/UR/census-observations?area-type=ctry,E92000001&dimensions=health_in_general,highest_qualification

        Args:
            population_type_name (str): The population type name (e.g., 'UR').
            area_type (str): The area type code (e.g., 'ctry').
            areas (str): Comma-separated area identifiers (e.g., 'E92000001').
            dimensions (str): Comma-separated dimension identifiers (e.g., 'health_in_general,highest_qualification').

        Returns:
            CensusObservationsResponse: A pydantic model containing observation data aggregated across all pages.
        """
        url = f"{self.base_url}/population-types/{population_type_name}/census-observations"
        params = {
            "area-type": f"{area_type},{areas}",
        }

        # Only add dimensions parameter if dimensions are provided
        if dimensions:
            params["dimensions"] = dimensions

        observations = self._paginate_get(url, params=params, item_key="observations")
        return CensusObservationsResponse(observations=observations)

    def get_dataset_dimensions(self, dataset_id: str, edition: str, version: str, limit: Optional[int] = None, offset: Optional[int] = None) -> DimensionsResponse:
        """Retrieve all dimensions for a specific dataset version.

        Args:
            dataset_id (str): The dataset identifier.
            edition (str): The edition of the dataset.
            version (str): The version of the dataset.
            limit (Optional[int]): Maximum number of items to retrieve.
            offset (Optional[int]): Starting index for the items.

        Returns:
            DimensionsResponse: A pydantic model containing the dimensions data.
        """
        url = f"{self.base_url}/datasets/{dataset_id}/editions/{edition}/versions/{version}/dimensions"
        params = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        items = self._paginate_get(url, params=params, item_key="items")
        return DimensionsResponse(items=items)

    def get_dimension_options(self, dataset_id: str, edition: str, version: str, dimension: str, option_ids: Optional[List[str]] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> DimensionOptionsResponse:
        """Retrieve options for a specific dimension of a dataset.

        Args:
            dataset_id (str): The dataset identifier.
            edition (str): The dataset edition.
            version (str): The dataset version.
            dimension (str): The dimension identifier.
            option_ids (Optional[List[str]]): List of option IDs to filter; if provided, overrides limit and offset.
            limit (Optional[int]): Maximum number of options to retrieve.
            offset (Optional[int]): Starting index for options.

        Returns:
            DimensionOptionsResponse: A pydantic model containing the dimension options data.
        """
        url = f"{self.base_url}/datasets/{dataset_id}/editions/{edition}/versions/{version}/dimensions/{dimension}/options"
        params = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if option_ids is not None:
            params["id"] = ",".join(option_ids)
        items = self._paginate_get(url, params=params, item_key="items")
        return DimensionOptionsResponse(items=items)

    def get_dataset_metadata(self, dataset_id: str, edition: str, version: str) -> DatasetMetadata:
        """Retrieve metadata for a specific dataset version.

        Args:
            dataset_id (str): The dataset identifier.
            edition (str): The dataset edition.
            version (str): The dataset version.

        Returns:
            DatasetMetadata: A pydantic model containing the metadata.
        """
        url = f"{self.base_url}/datasets/{dataset_id}/editions/{edition}/versions/{version}/metadata"
        response = self.session.get(url)
        response.raise_for_status()
        data = response.json()
        return DatasetMetadata(**data)

    def get_fast_census_data(self, population_type: str, area_type: str, area_codes: List[str],
                            dimensions: List[str], max_retries: int = 3) -> Dict[str, Any]:
        """Retrieve census observations directly without creating a filter job.

        This method uses the census-observations endpoint, which returns data immediately
        without the need for a filter job. This is more efficient for most use cases.

        Args:
            population_type (str): The population type (e.g., 'UR').
            area_type (str): The area type code (e.g., 'ctry', 'rgn').
            area_codes (List[str]): List of area codes to include.
            dimensions (List[str]): List of dimension IDs to include.
            max_retries (int, optional): Maximum number of retries for rate limiting. Defaults to 3.

        Returns:
            Dict[str, Any]: The JSON response containing the observations.
        """
        # Construct the area-type parameter (e.g., "ctry,E92000001,W92000004")
        area_param = f"{area_type},{','.join(area_codes)}"

        # Build the URL
        url = f"{self.base_url}/population-types/{population_type}/census-observations"
        params = {
            "area-type": area_param,
        }

        # Only add dimensions parameter if dimensions are provided
        if dimensions:
            # Construct the dimensions parameter (e.g., "sex,age")
            dimensions_param = ",".join(dimensions)
            params["dimensions"] = dimensions_param

        # Make the request with retry for rate limiting
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params)

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "5")
                    wait_time = int(retry_after) if retry_after.isdigit() else 5 * (attempt + 1)
                    print(f"Rate limited. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.HTTPError as e:
                # Print the full URL with parameters to help debug
                print(f"Error: {e} for url: {response.url}")

                if attempt < max_retries - 1:
                    print(f"Retrying... ({attempt+1}/{max_retries})")
                    time.sleep(2 * (attempt + 1))  # Exponential backoff
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
                    time.sleep(2 * (attempt + 1))  # Exponential backoff
                else:
                    raise

        raise Exception(f"Failed to get census data after {max_retries} attempts")

    def get_dataset_observations(self, dataset_id: str, edition: str, version: str,
                               area_type: Optional[str] = None, area_code: Optional[str] = None) -> Dict[str, Any]:
        """Get observations directly from a dataset without creating a filter job.

        Args:
            dataset_id (str): The dataset ID.
            edition (str): The dataset edition.
            version (str): The dataset version.
            area_type (Optional[str], optional): The area type to filter by. Defaults to None.
            area_code (Optional[str], optional): The area code to filter by. Defaults to None.

        Returns:
            Dict[str, Any]: The JSON response containing observations.
        """
        url = f"{self.base_url}/datasets/{dataset_id}/editions/{edition}/versions/{version}/json"

        # Add area filtering if provided
        params = {}
        if area_type and area_code:
            params["area-type"] = f"{area_type},{area_code}"

        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_dataset_observations_by_area_type(self, dataset_id: str, edition: str, version: str,
                                 area_type: str, area_codes: List[str],
                                 max_retries: int = 3) -> Dict[str, Any]:
        """Get observations for a dataset using a specific area type and list of area codes.

        This is the preferred way to access both TS and RM datasets for Census 2021 data.
        It works for both Topic Summaries (TS) and Multivariate Datasets (RM) and all geographic levels.

        Args:
            dataset_id (str): The dataset ID (e.g., "TS008" or "RM097").
            edition (str): The dataset edition (typically "2021").
            version (str): The dataset version (typically "1").
            area_type (str): The geographic level code (e.g., "ctry", "rgn", "la", "msoa", "lsoa", "oa").
            area_codes (List[str]): List of area codes to include in the query.
            max_retries (int, optional): Maximum number of retries for rate limiting. Defaults to 3.

        Returns:
            Dict[str, Any]: The JSON response containing the observations.
        """
        # Construct the area parameter (e.g., "ctry,E92000001,W92000004")
        area_param = f"{area_type},{','.join(area_codes)}"

        # Build the URL for the dataset-specific endpoint
        url = f"{self.base_url}/datasets/{dataset_id}/editions/{edition}/versions/{version}/json"
        params = {
            "area-type": area_param
        }

        # Make the request with retry for rate limiting
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params)

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "5")
                    wait_time = int(retry_after) if retry_after.isdigit() else 5 * (attempt + 1)
                    print(f"Rate limited. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.HTTPError as e:
                # Print the full URL with parameters to help debug
                print(f"Error: {e} for url: {response.url}")

                if attempt < max_retries - 1:
                    print(f"Retrying... ({attempt+1}/{max_retries})")
                    time.sleep(2 * (attempt + 1))  # Exponential backoff
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
                    time.sleep(2 * (attempt + 1))  # Exponential backoff
                else:
                    raise

        raise Exception(f"Failed to get dataset observations after {max_retries} attempts")

    # Additional methods (e.g., search endpoints) can be added below following the same pattern


def save_area_types(dataset: Dataset, area_types: List[AreaType]) -> None:
    """Save list of area types to a JSON file named 'area_types_<dataset.id>.json'."""
    # Convert area types from pydantic model to dict using model_dump method (which preserves aliases)
    area_types_data = [area.model_dump(by_alias=True) for area in area_types]
    filename = f"area_types_{dataset.id}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(area_types_data, f, indent=2)
    print(f"Saved area types for dataset '{dataset.title}' (ID: {dataset.id}) to {filename}")


if __name__ == '__main__':
    client = ONSApiClient()
    # Get all datasets
    datasets_response = client.get_datasets()
    datasets = datasets_response.datasets

    # Print information about each dataset
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset.title} (ID: {dataset.id})")

    # # For each dataset, download its area types if possible
    # for dataset in datasets:
    #     if dataset.is_based_on and dataset.is_based_on.id:
    #         population_type = dataset.is_based_on.id
    #         print(f"Downloading area types for dataset '{dataset.title}' with population type '{population_type}'")
    #         try:
    #             area_types = client.get_area_types(population_type)
    #             save_area_types(dataset, area_types)
    #         except Exception as e:
    #             print(f"Failed to download area types for dataset '{dataset.title}': {str(e)}")
    #     else:
    #         print(f"Skipping dataset '{dataset.title}' (ID: {dataset.id}) as no population type information is available.")

    # print("Done saving area types for all datasets.")


# # Example usage:
# client = ONSApiClient()
# datasets = client.get_datasets()

# print(datasets)
# print("--------------------------------")
# print(datasets.datasets[0])
