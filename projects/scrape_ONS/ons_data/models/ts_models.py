from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from .common import Dataset, Dimension, DimensionWithOptions


class TSDataset(Dataset):
    """Time Series dataset model"""
    dimensions: List[DimensionWithOptions] = []
    observations: List[str] = []


class TSObservation(BaseModel):
    """Observation value for TS datasets"""
    value: str
    dimension_values: Dict[str, str] = {}


class TSResponse(BaseModel):
    """Response structure for TS dataset API calls"""
    dimensions: List[Dict[str, Any]] = []
    observations: List[str] = []
    headers: Optional[List[Dict[str, Any]]] = None


class TSFlattenedRow(BaseModel):
    """Represents a single row in the flattened TS dataset output"""
    observation: str

    class Config:
        extra = "allow"  # Allow additional fields for dimensions
