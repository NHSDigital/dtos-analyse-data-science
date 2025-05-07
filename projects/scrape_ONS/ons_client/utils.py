import os
import time
from typing import List, TypeVar, Callable, Any

T = TypeVar('T')

def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """Split a list into smaller chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def retry_with_backoff(
        func: Callable,
        max_retries: int = 5,
        initial_backoff: int = 1,
        max_backoff: int = 60,
        backoff_factor: int = 2
    ) -> Any:
    """
    Retry a function with exponential backoff.

    Args:
        func: The function to retry
        max_retries: Maximum number of retries before giving up
        initial_backoff: Initial backoff time in seconds
        max_backoff: Maximum backoff time in seconds
        backoff_factor: Factor to multiply backoff time by after each retry

    Returns:
        The result of the function call

    Raises:
        The last exception raised by the function
    """
    retries = 0
    backoff = initial_backoff

    while retries < max_retries:
        try:
            return func()
        except Exception as e:
            retries += 1
            if retries == max_retries:
                raise e

            print(f"Retry {retries}/{max_retries} failed with error: {str(e)}")
            print(f"Retrying in {backoff} seconds...")

            time.sleep(backoff)
            backoff = min(backoff * backoff_factor, max_backoff)
