from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


class FilterDatasetRef(BaseModel):
    """Reference to a dataset within a filter request."""
    id: str
    edition: str = "2021"
    version: int = 1


class FilterDimension(BaseModel):
    """A dimension used in a filter request."""
    name: str
    is_area_type: bool = False
    options: Optional[List[str]] = None


class FilterCreate(BaseModel):
    """Request model for creating a new filter."""
    dataset: FilterDatasetRef
    population_type: str = "UR"
    dimensions: List[FilterDimension] = []


class FilterLink(BaseModel):
    """A link in an API response."""
    href: str
    id: Optional[str] = None


class FilterLinks(BaseModel):
    """Collection of links in a filter response."""
    version: Optional[FilterLink] = None
    self: Optional[FilterLink] = None
    dimensions: Optional[FilterLink] = None


class FilterResponse(BaseModel):
    """Response model for filter creation."""
    filter_id: str
    instance_id: str
    dataset: Optional[Dict[str, Any]] = None
    published: Optional[bool] = None
    type: Optional[str] = None
    population_type: Optional[str] = None
    links: Optional[FilterLinks] = None


class FilterOutputDownload(BaseModel):
    """Information about a downloadable file."""
    href: str
    size: Optional[str] = None
    public: Optional[str] = None


class FilterOutputDownloads(BaseModel):
    """Collection of downloadable files."""
    csv: Optional[FilterOutputDownload] = None
    xlsx: Optional[FilterOutputDownload] = None


class FilterOutput(BaseModel):
    """Response model for a filter output."""
    id: Optional[str] = None
    filter_id: Optional[str] = None
    filter_output_id: Optional[str] = None
    instance_id: Optional[str] = None
    state: Optional[str] = None
    downloads: Optional[FilterOutputDownloads] = None
    dataset: Optional[Dict[str, Any]] = None
    dimensions: Optional[List[Dict[str, Any]]] = None
    events: Optional[Dict[str, Any]] = None
    links: Optional[Dict[str, Any]] = None
    population_type: Optional[str] = None
