# ONS Client

## Overview

The ONS Client package provides an interface to the Office for National Statistics (ONS) Developer Hub API. It allows you to programmatically access, filter, and download datasets from ONS using HTTP requests. The package uses pydantic models for robust data validation and type safety, ensuring that API interactions are reliable and maintainable.

## Features

- **API Integration:** Connects to the ONS Developer Hub API at `https://api.beta.ons.gov.uk/v1` for retrieving and filtering datasets.
- **Filter Creation:** Utilizes pydantic models (e.g., FilterRequest, DatasetIdentifier, DimensionFilter, Area) to create and validate filter requests.
- **CSV Downloads:** Supports downloading filtered dataset outputs as CSV files. Handles both single-area queries and batch processing for large sets of area codes.
- **Batch Processing & Rate Limiting:** Implements batch processing with fixed delays to adhere to rate limits (e.g., 15 requests per 10 seconds for high-demand assets). This ensures compliance with ONS rate limiting guidelines. Future improvements could include dynamic adjustment based on the `Retry-After` header.
- **File Management:** Manages temporary files during batch processing, merging multiple CSV outputs into a consolidated file while ensuring proper header handling.
- **Utility Functions:** Includes helper functions for directory creation and list chunking to support data processing.

## Package Structure

- **models.py:** Contains pydantic models representing various entities such as filter requests, dataset identifiers, dimensions, and area information.
- **ons_client.py:** Implements the core ONS API client. This module handles API requests, caches area data, and provides methods to retrieve area types and other metadata.
- **filter_processor.py:** Processes filter requests by constructing the required payloads, submitting them to the API, and handling the response (including file downloads).
- **download_filtered_csv.py:** A script that leverages the client and filter processor to download CSV outputs. It demonstrates both single-area and batch processing workflows.
- **utils.py:** Provides utility functions (e.g., `chunk_list`, `ensure_dir`) to support directory management and data handling.

## How It Works

1. **Initialization and Caching:** The client (`ONSApiClient`) is instantiated to communicate with the ONS API. It also implements a caching mechanism for area codes. The method `get_cached_areas` retrieves cached area data for a given population type and area type. If the data is already cached, it returns it immediately, reducing redundant API calls and improving performance. If not, it fetches the data from the API and stores it in the cache for future use.

2. **Filter Request Creation:** Using the `FilterProcessor`, a filter request is built using specified parameters (dataset ID, population type, area type, and optionally, area codes). Pydantic models ensure that the request data is valid.

3. **Data Download:** The filter request is processed to submit the job to the API, which generates CSV outputs. For large requests, the package splits the area codes into batches, processes each batch separately (with delays to respect rate limits), and then merges them into a final CSV file.

4. **Output:** The final CSV file(s) are saved in a structured directory (under a `data` folder) and can be used for further analysis or integration into other systems.

## Usage

### Downloading CSV for a Specific Area

Run the following command to download CSV data for a specific area:

```python
python download_filtered_csv.py
```

This script will use the provided dataset ID, population type, and area type (with an optional area code) to download the CSV file.

### Downloading CSV for All Area Types

Use the `download_all_area_types` function in `download_filtered_csv.py` to iterate through all available area types and download corresponding CSV outputs, merging batch results into a single file per area type.

## Requirements

- Python 3.7+
- requests
- pydantic

## Installation

Clone the repository and install the required dependencies:

```python
pip install -r requirements.txt
```

## Integration with ONS Developer Hub

This package is built to work seamlessly with the ONS Developer Hub API:

- **Endpoint:** Uses the base URL `https://api.beta.ons.gov.uk/v1`.
- **Rate Limiting:** Implements delays between batches to respect the APIs rate limits.
- **Filter Service:** Supports creating customizable filter requests as documented in the ONS Developer Hub guidelines.

For further details, refer to the [ONS Developer Hub](https://developer.ons.gov.uk/) documentation.

## Future Improvements

- Enhance rate limiting by dynamically handling the `Retry-After` header from API responses.
- Explore asynchronous processing for faster batch downloads.
- Improve error handling and logging for better troubleshooting and stability.

## License

[Specify Your License Here]
