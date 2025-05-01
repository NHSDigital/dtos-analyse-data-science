# ONS Census Data Tool

A Python tool for retrieving, processing, and flattening census data from the UK Office for National Statistics (ONS) API.

## Overview

This tool provides a clean, efficient way to download and process Census 2021 data from the ONS API. It supports both Time Series (TS) and Regular Matrix (RM) datasets, handling the specific requirements of each format while providing a consistent output structure.

Key features:

- Progress bar visualization for batch processing
- Efficient handling of large datasets through batched requests
- Automatic flattening of complex hierarchical data into tabular CSV format
- Support for all geographic levels (country, region, local authority, MSOA, LSOA, output area)
- Modular architecture with Pydantic models for robust type checking
- Implements ONS API rate limiting guidelines

## Installation

### Prerequisites

- Python 3.8 or higher
- Required Python packages:
  - `requests`
  - `pandas`
  - `tqdm` (for progress bars)
  - `pydantic`

### Setup

1. Clone this repository
2. Install dependencies:

```bash
pip install requests pandas tqdm pydantic
```

## Usage

Run the tool from the command line:

```bash
# Download a dataset
python run_census.py --dataset TS030 --geo-level ctry

# List all available datasets
python run_census.py --list-datasets
```

### Command-line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--dataset` | Dataset ID to retrieve | `TS030`, `RM097` |
| `--list-datasets` | List all available datasets and examples | |
| `--geo-level` | Geographic level to retrieve (if not specified, all levels will be processed) | `ctry`, `rgn`, `la`, `msoa`, `lsoa`, `oa` |
| `--output-dir` | Output directory for data files (defaults to `data/{dataset_id}/`) | `./my_data` |
| `--batch-size` | Override the default batch size for area processing | `100` |
| `--population-type` | Population type for census data (default: `UR`) | `UR` |
| `--debug` | Enable debug logging | |

### Examples

Download a Time Series dataset for all countries:

```bash
python run_census.py --dataset TS030 --geo-level ctry
```

Download a Regular Matrix dataset for all regions:

```bash
python run_census.py --dataset RM097 --geo-level rgn
```

Download a dataset for all geographic levels:

```bash
python run_census.py --dataset TS030
```

Use a specific output directory:

```bash
python run_census.py --dataset TS030 --geo-level ctry --output-dir ./my_census_data
```

Handle dataset availability at different geographic levels:

```bash
# This will work (Religion data is available at OA level)
python run_census.py --dataset TS030 --geo-level oa

# This will show an availability error (dataset not available at OA level)
python run_census.py --dataset TS009 --geo-level oa

# Try a different geographic level instead
python run_census.py --dataset TS009 --geo-level lsoa
```

## ONS API Integration

This tool connects to the Office for National Statistics (ONS) API to retrieve census data. The API is available at:

[https://api.beta.ons.gov.uk/v1](https://api.beta.ons.gov.uk/v1)

No API key is required as the ONS API is open and unrestricted.

### API Endpoints Used

The tool primarily uses the following ONS API endpoints as documented in the [ONS Developer Hub](https://developer.ons.gov.uk/):

1. **Population Types**:
   - `GET /population-types` - Lists available population types

2. **Area Types**:
   - `GET /population-types/{population-type}/area-types` - Lists available geographic levels
   - `GET /population-types/{population-type}/area-types/{area-type}/areas` - Lists areas for a specific geographic level

3. **Census Observations (for RM datasets)**:
   - `GET /population-types/{population-type}/census-observations` - Retrieves observation data

4. **Dataset Observations (for TS datasets)**:
   - `GET /datasets/{datasetId}/editions/{edition}/versions/{version}/json` - Retrieves time series data

### Rate Limiting

The tool implements best practices to respect ONS API rate limits:

- **Batch Processing**: Data is retrieved in batches to avoid exceeding API limits
- **Request Throttling**: Implements delays between requests as per ONS guidelines
- **Retry Logic**: Handles 429 responses with exponential backoff

According to ONS Developer Hub, the following rate limits are applied:

- 120 requests per 10 seconds for all site and API assets
- 200 requests per 1 minute for all site and API assets
- 15 requests per 10 seconds for high-demand assets

## Dataset Types

The tool supports two types of ONS datasets:

### Time Series (TS) Datasets

Time Series datasets (prefixed with `TS`) represent simpler data structures focused on a single variable across different dimensions. These are accessed using the `/datasets/{datasetId}/editions/{edition}/versions/{version}/json` endpoint.

Examples include:

- `TS030`: Religion by geographic area
- `TS008`: Sex by geographic area
- `TS021`: Age by geographic area

### Regular Matrix (RM) Datasets

Regular Matrix datasets (prefixed with `RM`) represent more complex data structures with multiple dimensions. The original implementation in ONS Developer Hub documentation suggests using the `/population-types/{population-type}/census-observations` endpoint, but our testing found that using the same endpoint as TS datasets works more reliably.

Examples include:

- `RM097`: Occupancy rating (bedrooms) by ethnic group
- `RM040`: Age by sex by long-term health condition
- `RM052`: Country of birth by sex

### Finding Dataset IDs

While this tool requires you to know the dataset ID you want to download, you can browse available datasets through the [ONS website](https://www.ons.gov.uk/census) or use the API directly:

### Example API Request

To list all available datasets, you can use the following API request:

```http
GET https://api.beta.ons.gov.uk/v1/datasets
```

This endpoint returns a list of datasets available in the ONS API, including metadata such as dataset IDs, titles, and descriptions. You can use this information to identify the dataset ID required for your analysis.

## Output Format

Data is saved in flattened CSV format with dimensions as columns:

``` csv
ctry,ctry_code,religion_tb,religion_tb_code,observation
England,E92000001,No religion,1,20715664
England,E92000001,Christian,2,26167900
...
```

Each row represents a unique combination of dimensions with the corresponding observation value.

## Project Structure

``` plaintext
ons_data/
├── __init__.py          # Package initialization
├── api/                 # API client components
│   ├── __init__.py
│   ├── client.py        # Base API client
│   ├── ts_client.py     # Time Series client
│   └── rm_client.py     # Regular Matrix client
├── models/              # Pydantic data models
│   ├── __init__.py
│   ├── common.py        # Shared models
│   ├── ts_models.py     # Time Series models
│   └── rm_models.py     # Regular Matrix models
├── processors/          # Data processors
│   ├── __init__.py
│   ├── base.py          # Base processor
│   ├── ts_processor.py  # Time Series processor
│   └── rm_processor.py  # Regular Matrix processor
└── cli.py               # Command-line interface
```

## Testing

The project includes a comprehensive test suite to ensure code quality and reliability. Tests are organized in a structure that mirrors the main codebase.

### Test Structure

``` planetext
tests/
├── unit/                # Unit tests
│   ├── api/             # API client tests
│   ├── models/          # Data model tests
│   └── processors/      # Processor tests
└── integration/         # Integration tests
```

### Running Tests

To run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run unit tests only
python -m pytest tests/unit/

# Run specific test modules
python -m pytest tests/unit/api/
python -m pytest tests/unit/models/
python -m pytest tests/unit/processors/

# Run with coverage report
python -m pytest tests/ --cov=ons_data
```

### Test Coverage

The project maintains high test coverage across critical components:

- **Models**: 100% coverage - ensuring robust data validation
- **API Clients**: ~90% coverage - verifying correct API interactions
- **Processors**: ~80% coverage - testing data transformation logic

The overall test coverage is maintained above 65%, with continuous improvements targeted in the CLI module.

### Test Types

1. **Unit Tests**:
   - **Model Tests**: Verify the data validation and conversion logic in Pydantic models
   - **API Client Tests**: Test API URL construction, response parsing, and error handling
   - **Processor Tests**: Verify data transformation and flattening functionality

2. **Test Fixtures**:
   - Sample API responses for both TS and RM datasets
   - Mock HTTP responses for predictable testing

3. **Mocking Strategy**:
   - External API calls are mocked to prevent actual network requests
   - File operations are mocked for testing I/O functionality

4. **Pydantic Models in Testing**:
   - Pydantic models provide automatic validation during tests
   - Models catch type mismatches and schema violations early
   - Serialization/deserialization is thoroughly tested to ensure data consistency
   - The models serve as a contract between API responses and application code

### Contributing Tests

When adding new features, ensure test coverage by:

1. Adding unit tests for any new functions or classes
2. Creating mocks for external dependencies
3. Running the test suite with coverage to identify any gaps

## How It Works

1. **Configuration**: The tool parses command-line arguments and creates a configuration object.
2. **Area Retrieval**: It fetches all areas for the specified geographic level using `/population-types/{population_type}/area-types/{geo_level}/areas`.
3. **Batch Processing**: Areas are processed in batches to avoid API limits, with progress visualization.
4. **Data Processing**:
   - For TS datasets: Uses `/datasets/{datasetId}/editions/2021/versions/1/json?area-type={geo_level},{area_codes}`
   - For RM datasets: Uses the same endpoint format as TS datasets
   - Raw API responses are temporarily stored
   - The appropriate processor flattens the data by reconstructing dimension combinations
   - The final CSV is generated with dimensions as columns
5. **Cleanup**: Temporary files are removed, leaving only the flattened data

## Design Patterns

The tool implements several design patterns for flexibility and maintainability:

- **Factory Pattern**: For creating the appropriate client and processor based on dataset type
- **Strategy Pattern**: Different processing strategies for different dataset types
- **Decorator Pattern**: For retry logic in API requests
- **Repository Pattern**: For data access and caching

## Troubleshooting

If you encounter any issues:

1. Use the `--debug` flag to enable detailed logging
2. Check the `.debug.json` files in the output directory for raw API responses
3. Ensure you have a stable internet connection
4. Verify that the dataset ID is correct and available in the ONS API

Common issues:

- **404 Errors**: The ONS API sometimes returns 404 errors for valid URLs. The tool includes retry logic.
- **429 Too Many Requests**: If you hit rate limits, the tool will back off and retry automatically.
- **Timeout Errors**: For very large datasets, try using a smaller geographic level or increasing batch size.

## Limitations

- The ONS API sometimes returns 404 errors for valid URLs. The tool includes retry logic.
- Some datasets may have rate limiting. The tool implements appropriate backoff strategies.
- Very large geographic levels (like OA) may take significant time to process.
- The API is in Beta and may have breaking changes, as noted in the ONS Developer Hub.
- Not all datasets are available at all geographic levels. The tool now automatically checks if a dataset is available at a specified geographic level before attempting to process it, avoiding unnecessary processing.
  - For example, some datasets (like TS009) are not available at the Output Area (OA) level.
  - The tool will display a clear message when a dataset isn't available at a specific level.
  - When using the OA level and encountering unavailability, try using higher geographic levels like LSOA, MSOA, or Local Authority (LA).

## References

- [ONS Developer Hub](https://developer.ons.gov.uk/) - Official documentation for the ONS API
- [ONS Census 2021](https://www.ons.gov.uk/census) - Census data and documentation
- [ONS API Observations](https://developer.ons.gov.uk/observations/) - Guide to requesting specific observations
- [ONS API Rate Limiting](https://developer.ons.gov.uk/bots/) - Rate limiting guidelines

## License

This project is licensed under the MIT License - see the LICENSE file for details.
