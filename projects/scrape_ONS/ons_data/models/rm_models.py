from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from .common import Dataset, Dimension, DimensionWithOptions


class RMDimension(BaseModel):
    """Dimension in an RM dataset response"""
    dimension_id: str
    option: str
    option_id: Optional[str] = None


class RMObservation(BaseModel):
    """Observation value with dimensions for RM datasets"""
    observation: str
    dimensions: List[RMDimension]


class RMDataset(Dataset):
    """Regular Matrix dataset model"""
    dimensions: List[DimensionWithOptions] = []
    observations: List[RMObservation] = []


class RMResponse(BaseModel):
    """Response structure for RM dataset API calls"""
    observations: List[Dict[str, Any]] = []


class RMFlattenedRow(BaseModel):
    """Represents a single row in the flattened RM dataset output"""
    observation: str

    class Config:
        extra = "allow"  # Allow additional fields for dimensions
