#!/usr/bin/env python3
"""
Wrapper script to run the new ONS Census Data Tool with progress bars

This enhanced version includes:
- Progress bar visualization for batch processing
- Improved error handling and recovery
- Structure based on Pydantic models
"""

import sys
import os

# Check for required dependencies
try:
    import tqdm
except ImportError:
    print("Error: tqdm library is required for progress bars.")
    print("Please install it using: pip install tqdm")
    sys.exit(1)

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ons_data.cli import main

if __name__ == "__main__":
    main()
