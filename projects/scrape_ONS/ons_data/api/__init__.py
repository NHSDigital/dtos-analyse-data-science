from typing import Optional
import logging

from .client import ONSApiClient
from .ts_client import TSApiClient
from .rm_client import RMApiClient
from ..models.common import DatasetType

logger = logging.getLogger(__name__)

class ApiClientFactory:
    """
    Factory for creating API clients based on dataset type.
    """

    @staticmethod
    def get_client(dataset_id: str) -> Optional[ONSApiClient]:
        """
        Get the appropriate API client for a dataset based on its ID.

        Args:
            dataset_id: The dataset ID (e.g., "TS030", "RM097")

        Returns:
            ONSApiClient: An instance of the appropriate client, or None if type unknown
        """
        if not dataset_id or len(dataset_id) < 2:
            logger.error(f"Invalid dataset ID: {dataset_id}")
            return None

        # Extract the dataset type from the ID prefix
        dataset_type_prefix = dataset_id[:2].upper()

        try:
            # Convert to enum value
            dataset_type = DatasetType(dataset_type_prefix)

            # Return the appropriate client
            if dataset_type == DatasetType.TIME_SERIES:
                logger.info(f"Creating TSApiClient for dataset {dataset_id}")
                return TSApiClient()
            elif dataset_type == DatasetType.REGULAR_MATRIX:
                logger.info(f"Creating RMApiClient for dataset {dataset_id}")
                return RMApiClient()

        except ValueError:
            logger.error(f"Unknown dataset type prefix: {dataset_type_prefix}")

        return None
