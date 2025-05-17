import logging
import os
from pathlib import Path

from dotenv import load_dotenv
import foundry_sdk
from foundry_sdk import FoundryClient
from pprint import pprint

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def create_dummy_json_lines_file(n_rows: int = 1_000, filename: str = "dummy", path=Path('/tmp/')) -> str:

    filename = path / f"{filename}--{n_rows}-lines.jsonl"

    with open(filename, "w") as f:
        for i in range(n_rows + 1):
            f.write(f'{{"id": {i}, "name": "Name {i}"}}\n')

    filesize_mb = os.path.getsize(filename) / (1024 * 1024)

    logger.debug(f"Created dummy file with {n_rows:,} JSON lines: {filename}, {filesize_mb:.2f} MB")

    return filename


if __name__ == "__main__":
    filename = create_dummy_json_lines_file()


