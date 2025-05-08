from typing import Optional
import logging

from .base import BaseProcessor
from .ts_processor import TSProcessor
from .rm_processor import RMProcessor
from ..models.common import DatasetType

logger = logging.getLogger(__name__)

class ProcessorFactory:
    """
    Factory for creating dataset processors based on dataset type.
    """

    @staticmethod
    def get_processor(dataset_id: str) -> Optional[BaseProcessor]:
        """
        Get the appropriate processor for a dataset based on its ID.

        Args:
            dataset_id: The dataset ID (e.g., "TS030", "RM097")

        Returns:
            BaseProcessor: An instance of the appropriate processor, or None if type unknown
        """
        if not dataset_id or len(dataset_id) < 2:
            logger.error(f"Invalid dataset ID: {dataset_id}")
            return None

        # Extract the dataset type from the ID prefix
        dataset_type_prefix = dataset_id[:2].upper()

        try:
            # Convert to enum value
            dataset_type = DatasetType(dataset_type_prefix)

            # Return the appropriate processor
            if dataset_type == DatasetType.TOPIC_SUMMARY:
                logger.info(f"Creating TSProcessor for dataset {dataset_id}")
                return TSProcessor()
            elif dataset_type == DatasetType.REGULAR_MATRIX:
                logger.info(f"Creating RMProcessor for dataset {dataset_id}")
                return RMProcessor()

        except ValueError:
            logger.error(f"Unknown dataset type prefix: {dataset_type_prefix}")

        return None
