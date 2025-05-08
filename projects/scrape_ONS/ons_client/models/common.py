from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum


class DatasetType(str, Enum):
    """Enum representing the types of datasets supported by the ONS API"""
    TOPIC_SUMMARY = "TS"
    REGULAR_MATRIX = "RM"
    OTHER = "OTHER"


class GeoLevel(BaseModel):
    """Geographic level representation with its batch size configuration"""
    id: str
    name: str
    batch_size: int = 200


class Dimension(BaseModel):
    """Base class for dataset dimensions"""
    id: str
    label: str


class DimensionOption(BaseModel):
    """An option within a dimension (e.g., a specific religion or area)"""
    id: str
    label: str


class DimensionWithOptions(Dimension):
    """Dimension that includes its available options"""
    options: List[DimensionOption] = []


class Dataset(BaseModel):
    """Base model for all ONS datasets"""
    id: str
    title: str
    description: Optional[str] = None

    @property
    def dataset_type(self) -> DatasetType:
        """Determine the dataset type based on the ID prefix"""
        if self.id.startswith("TS"):
            return DatasetType.TOPIC_SUMMARY
        elif self.id.startswith("RM"):
            return DatasetType.REGULAR_MATRIX
        else:
            return DatasetType.OTHER


class BatchSizeConfig(BaseModel):
    """Configuration for batch sizes by geographic level"""
    ctry: int = 200    # Countries
    rgn: int = 200     # Regions
    la: int = 200      # Local Authorities
    msoa: int = 200    # MSOAs
    lsoa: int = 200    # LSOAs
    oa: int = 200      # Output Areas

    def get_for_level(self, level: str) -> int:
        """Get the batch size for a specific geographic level"""
        if hasattr(self, level):
            return getattr(self, level)
        return 50  # Default batch size


class ONSConfig(BaseModel):
    """Configuration for ONS data retrieval"""
    dataset_id: str
    geo_levels: List[str] = []
    population_type: str = "UR"
    output_dir: str = "data"
    batch_sizes: BatchSizeConfig = Field(default_factory=BatchSizeConfig)
    use_filter: bool = False


class DatasetAvailability(BaseModel):
    """Dataset availability model for checking if a dataset exists at a specific geographic level."""
    dataset_id: str
    geo_level: str
    population_type: str
    is_available: bool
    error_message: Optional[str] = None
