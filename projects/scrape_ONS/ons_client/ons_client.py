from typing import List, Optional, Dict, Any, TypeVar, Generic, Callable
import requests
import json
import os
import time
from pydantic import BaseModel

from models import (
    Dataset, DatasetsResponse, PopulationType, AreaType,
    Area, Dimension, Categorisation, CensusObservationsResponse
)
from utils import retry_with_backoff

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
        items = self._paginate_get(url, params=params)
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
            "dimensions": dimensions
        }
        observations = self._paginate_get(url, params=params, item_key="observations")
        return CensusObservationsResponse(observations=observations)

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
    print(f"Found {len(datasets)} datasets.")
    print(datasets[-1])

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
