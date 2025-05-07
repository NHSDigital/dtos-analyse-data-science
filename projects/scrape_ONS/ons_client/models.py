from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

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
    id: str
    label: str

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
