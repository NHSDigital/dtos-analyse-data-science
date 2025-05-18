import gc
import json
import logging
import os
from pathlib import Path
import time
from typing import List

from dotenv import load_dotenv
import foundry_sdk
from foundry_sdk import FoundryClient
from pprint import pprint

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def create_list_of_events(n_events: int = 1_000, n_kv_pairs_per_event: int = 100) -> dict:
    events = []

    for i in range(n_events):
        payload = [{f"key_{j}": f"value_{j}"} for j in range(n_kv_pairs_per_event)]
        events.append({"id": i, "name": "foo", "payload": payload})

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


def get_foundry_client(host: str, token: str) -> FoundryClient:
    return foundry_sdk.FoundryClient(
        auth=foundry_sdk.UserTokenAuth(token),
        hostname=host,
    )


def upload_file_to_foundry_dataset(client: FoundryClient, dataset_rid: str, filepath_local: str, filename_foundry: str) -> None:
    with open(filepath_local, "rb") as f:
        bytes_ = f.read()

    start_time = time.time()

    client.datasets.Dataset.File.upload(
        dataset_rid=dataset_rid,
        file_path=filename_foundry,
        body=bytes_
    )

    end_time = time.time()

    upload_time_seconds = end_time - start_time

    logger.info(f"File {filepath_local} uploaded to foundry as {filename_foundry} in {upload_time_seconds:.0f} seconds")


if __name__ == "__main__":

    logger.info("Start")

    token: str = os.environ["BEARER_TOKEN"]
    host: str = os.environ["HOSTNAME"]
    dataset_rid: str = os.environ["DATASET_RID"]

    n_events: int = 100
    n_kv_pairs_per_event: int = 100

    filename_description: str = 'dummy'

    events = create_list_of_events(n_events, n_kv_pairs_per_event)
    filename_local = get_local_file_name(filename_description, n_events, n_kv_pairs_per_event)

    filepath_local = create_json_lines_file(filename_local, events)
    client = get_foundry_client(host, token)
    upload_file_to_foundry_dataset(client, dataset_rid, filepath_local, filename_foundry=filename_local)

    gc.collect()

    logger.info("End")
