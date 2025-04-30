from pydantic import BaseModel, Field, root_validator
from typing import Optional, List, Dict, Any, Union

# Models for API requests
class DatasetIdentifier(BaseModel):
    id: str
    edition: str = "2021"
    version: int = 1

class DimensionFilter(BaseModel):
    name: str
    is_area_type: bool = True
    options: List[str]

class FilterRequest(BaseModel):
    dataset: DatasetIdentifier
    population_type: str
    dimensions: List[DimensionFilter]

# Models for API responses
class IsBasedOn(BaseModel):
    id: Optional[str] = Field(None, alias="@id")
    type: Optional[str] = Field(None, alias="@type")

class Dataset(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    unit_of_measure: Optional[str] = None
    survey: Optional[str] = None
    uri: Optional[str] = None
    is_based_on: Optional[IsBasedOn] = None

class DatasetsResponse(BaseModel):
    datasets: List[Dataset] = Field(..., alias="items")

class PopulationType(BaseModel):
    id: str
    name: str

class AreaType(BaseModel):
    id: str
    name: str = Field(..., alias="label")
    total_count: int
    hierarchy_order: int

class Area(BaseModel):
    id: str
    label: str
    area_type: str

class Dimension(BaseModel):
    id: Optional[str] = None
    label: Optional[str] = Field(None, alias="label")
    name: Optional[str] = None
    description: Optional[str] = None

    class Config:
        extra = "allow"  # Allow additional fields in the response

class DimensionItem(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    options: Optional[List[Dict[str, Any]]] = None
    href: Optional[str] = None

    class Config:
        extra = "allow"  # Allow additional fields

class Categorisation(BaseModel):
    id: str
    label: str

class CensusObservationsResponse(BaseModel):
    observations: List[dict]  # This can be refined based on the actual response schema

# New models for filter responses
class CsvDownload(BaseModel):
    href: Optional[str] = None
    size: Optional[str] = None

class Downloads(BaseModel):
    csv: Optional[CsvDownload] = None

class FilterResponse(BaseModel):
    filter_id: str

class FilterSubmitResponse(BaseModel):
    filter_output_id: str

class FilterOutputResponse(BaseModel):
    downloads: Optional[Downloads] = None

## New models for dataset dimensions and metadata endpoints

class DimensionsResponse(BaseModel):
    items: List[DimensionItem] = Field(..., alias="items")

class DimensionOption(BaseModel):
    id: str = Field(..., alias="option")
    label: str

class DimensionOptionsResponse(BaseModel):
    items: List[DimensionOption] = Field(..., alias="items")

class DimensionValues(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    option_id: Optional[str] = None
    option: Optional[str] = None
    label: Optional[str] = None
    values: Optional[List[Dict[str, Any]]] = None

    class Config:
        extra = "allow"

class DatasetMetadata(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    headers: Optional[List[str]] = None
    is_based_on: Optional[Dict[str, Any]] = None
    type: Optional[str] = None
    dimensions: Optional[List[Dict[str, Any]]] = None
    dimension_groups: Optional[List[Dict[str, Any]]] = None

    class Config:
        extra = "allow"

# Model for TS (Topic Summary) dataset response structure
class TSDatasetResponse(BaseModel):
    observations: List[Union[str, int, float]]
    total_observations: Optional[int] = None
    dimensions: Optional[List[Dict[str, Any]]] = None
    headers: Optional[List[str]] = None
    metadata: Optional[DatasetMetadata] = None

    @root_validator(pre=True)
    def ensure_observations_list(cls, values):
        """Ensure observations is always a list, even if it's None in the input data"""
        if "observations" not in values or values["observations"] is None:
            values["observations"] = []
        return values

    class Config:
        extra = "allow"  # Allow extra fields in the response

# New models for RM dataset response structure
class RMDimensionItem(BaseModel):
    dimension_id: Optional[str] = None
    option: Optional[str] = None
    option_id: Optional[str] = None

    class Config:
        extra = "allow"  # Allow additional fields

class RMObservation(BaseModel):
    observation: Optional[Union[str, int, float]] = None
    dimensions: Optional[Union[Dict[str, str], List[RMDimensionItem], List[Dict[str, Any]]]] = None

    class Config:
        extra = "allow"  # Allow additional fields

class RMDimensionInfo(BaseModel):
    name: Optional[str] = None
    dimension_name: Optional[str] = None
    id: Optional[str] = None
    label: Optional[str] = None
    options: Optional[List[Dict[str, Any]]] = None

    class Config:
        extra = "allow"  # Allow additional fields

class RMDatasetResponse(BaseModel):
    observations: Optional[List[Union[str, int, float, Dict[str, Any], RMObservation]]] = []
    dimensions: Optional[Union[Dict[str, Any], List[RMDimensionInfo], List[Dict[str, Any]]]] = None
    headers: Optional[List[str]] = None
    total_observations: Optional[int] = None

    @root_validator(pre=True)
    def ensure_observations_list(cls, values):
        """Ensure observations is always a list, even if it's None in the input data"""
        if "observations" not in values or values["observations"] is None:
            values["observations"] = []
        return values

    class Config:
        extra = "allow"  # Allow extra fields in the response
