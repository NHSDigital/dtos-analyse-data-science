import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import List


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def create_list_of_events(n_events: int = 1_000, n_kv_pairs_per_event: int = 100) -> dict:
    events = []

    for i in range(n_events):
        payload = [{f"key_{j}": f"value_{j}"} for j in range(n_kv_pairs_per_event)]
        events.append({
            "id": i,
            "name": "foo",
            "timestamp": datetime.now().isoformat(),
            "payload": payload
            })

    logger.info(f"Created {n_events:,} events, each with {n_kv_pairs_per_event:,} key value pairs.")

    return events


def get_local_file_name(filename_description: str, n_events: int, n_kv_pairs_per_event: int) -> str:
    return f"{filename_description}--{n_events}-events-by-{n_kv_pairs_per_event}-kv-pairs.jsonl"


def create_json_lines_file(filename: str, events: List[dict], path=Path('/tmp/')) -> str:

    filepath = path / filename

    with open(filepath, "w") as f:
        f.write("\n".join(map(json.dumps, events)))

    filesize_mb = os.path.getsize(filepath) / (1024 * 1024)

    logger.info(f"Created local JSON lines file: {filepath}, {filesize_mb:.2f} MB")

    return str(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a local JSON lines file.")
    parser.add_argument("--n_events", type=int, default=100, help="Number of events to create")
    parser.add_argument("--n_kv_pairs_per_event", type=int, default=100, help="Number of key-value pairs per event")
    parser.add_argument("--filename_description", type=str, default="dummy", help="Description for the filename")
    args = parser.parse_args()

    n_events: int = args.n_events
    n_kv_pairs_per_event: int = args.n_kv_pairs_per_event
    filename_description: str = args.filename_description

    events = create_list_of_events(n_events, n_kv_pairs_per_event)
    filename_local = get_local_file_name(filename_description, n_events, n_kv_pairs_per_event)

    filepath_local = create_json_lines_file(filename_local, events)
