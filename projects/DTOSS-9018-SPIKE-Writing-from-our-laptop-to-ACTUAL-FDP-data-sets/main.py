import logging
import os
from pathlib import Path

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


def create_dummy_json_lines_file(filename: str = "dummy", n_rows: int = 1_000, path=Path('/tmp/')) -> str:

    filename_prefix, filename_suffix = filename.split(".")
    filepath = path / f"{filename_prefix}--{n_rows}-lines.{filename_suffix}"

    with open(filepath, "w") as f:
        for i in range(n_rows + 1):
            f.write(f'{{"id": {i}, "name": "Name {i}"}}\n')

    filesize_mb = os.path.getsize(filepath) / (1024 * 1024)

    logger.info(f"Created dummy file with {n_rows:,} JSON lines: {filepath}, {filesize_mb:.2f} MB")

    return str(filepath)


def get_foundry_client(host: str, token: str) -> FoundryClient:
    return foundry_sdk.FoundryClient(
        auth=foundry_sdk.UserTokenAuth(token),
        hostname=host,
    )


def upload_file_to_foundry_dataset(client: FoundryClient, dataset_rid: str, filepath_local: str, filename_foundry: str) -> None:
    with open(filepath_local, "rb") as f:
        client.datasets.Dataset.File.upload(
            dataset_rid=dataset_rid,
            file_path=filename_foundry,
            body=f.read()
        )
    logger.info(f"File {filepath_local} uploaded to foundry as {filename_foundry}.")


if __name__ == "__main__":
    token: str = os.environ["BEARER_TOKEN"]
    host: str = os.environ["HOSTNAME"]
    dataset_rid: str = os.environ["DATASET_RID"]

    filename_local: str = 'dummy.jsonl'
    filename_foundry: str = 'dummy_fdp.jsonl'

    filepath_local = create_dummy_json_lines_file(filename_local)
    client = get_foundry_client(host, token)
    upload_file_to_foundry_dataset(client, dataset_rid, filepath_local, filename_foundry)
