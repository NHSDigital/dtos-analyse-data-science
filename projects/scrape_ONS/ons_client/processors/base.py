from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
import os
import logging


logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """
    Base abstract class for dataset processors.

    This class defines the interface that all dataset processors must implement,
    regardless of dataset type (TS or RM).
    """

    @abstractmethod
    def process_response(self, response: Dict[str, Any], output_file: str, area_metadata: Dict[str, Any] = None) -> str:
        """
        Process API response data and save it to a file.

        Args:
            response: Raw API response data
            output_file: Path to save the processed data
            area_metadata: Additional metadata about areas (optional)

        Returns:
            str: Path to the saved file
        """
        pass

    @abstractmethod
    def flatten_data(self, input_file: str, output_file: Optional[str] = None, area_metadata: Dict[str, Any] = None) -> str:
        """
        Flatten a dataset from its raw form into a standardized tabular format.

        Args:
            input_file: Path to the input file with raw data
            output_file: Path to save the flattened data (optional)
            area_metadata: Additional metadata about areas (optional)

        Returns:
            str: Path to the flattened file
        """
        pass

    def get_default_output_file(self, input_file: str, suffix: str = "_flat") -> str:
        """
        Generate a default output filename based on the input filename.

        Args:
            input_file: Original input filename
            suffix: Suffix to add to the output filename

        Returns:
            str: Generated output filename
        """
        base, ext = os.path.splitext(input_file)
        return f"{base}{suffix}{ext}"

    def validate_file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists and has content.

        Args:
            file_path: Path to check

        Returns:
            bool: True if file exists and has content, False otherwise
        """
        if not os.path.exists(file_path):
            logger.warning(f"File does not exist: {file_path}")
            return False

        if os.path.getsize(file_path) == 0:
            logger.warning(f"File is empty: {file_path}")
            return False

        return True
