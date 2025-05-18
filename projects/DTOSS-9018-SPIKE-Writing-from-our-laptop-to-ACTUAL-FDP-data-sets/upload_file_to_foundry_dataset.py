import argparse
from datetime import datetime
import logging
import os
import time
import uuid

from dotenv import load_dotenv
import foundry_sdk
from foundry_sdk import FoundryClient
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def get_foundry_dataset_name() -> str:
    return f"raw-nsp-events--{datetime.now().strftime('%Y-%m-%d--%H-%M-%S')}--{uuid.uuid4()}"


def get_foundry_client(host: str, token: str) -> FoundryClient:
    return foundry_sdk.FoundryClient(
        auth=foundry_sdk.UserTokenAuth(token),
        hostname=host,
    )


def upload_file_to_foundry_dataset(client: FoundryClient, dataset_rid: str, filepath_local: str, foundry_filename: str) -> None:
    with open(filepath_local, "rb") as f:
        bytes_ = f.read()

    logger.info("Starting upload...")

    start_time = time.time()

    client.datasets.Dataset.File.upload(
        dataset_rid=dataset_rid,
        file_path=foundry_filename,
        body=bytes_
    )

    end_time = time.time()

    upload_time_seconds = end_time - start_time

    logger.info(f"File {filepath_local} uploaded to foundry as {foundry_filename} in {upload_time_seconds:.0f} seconds")


if __name__ == "__main__":
    token: str = os.environ["BEARER_TOKEN"]
    host: str = os.environ["HOSTNAME"]
    parent_folder_rid: str = os.environ["PARENT_FOLDER_RID"]

    parser = argparse.ArgumentParser(description="Upload a file to a Foundry dataset.")
    parser.add_argument("--filepath_local", type=str, default='/tmp/dummy.jsonl', help="Path to the local file to upload.")
    parser.add_argument("--foundry_dataset_name", type=str, default=get_foundry_dataset_name(), help="Name of the dataset in Foundry.")
    parser.add_argument("--foundry_filename", type=str, default=None, help="Foundry destination file path.")
    args = parser.parse_args()

    filepath_local: str = args.filepath_local
    filename_local: str = os.path.basename(filepath_local)

    foundry_dataset_name: str = args.foundry_dataset_name
    foundry_filename: str = args.foundry_filename or filename_local

    client = get_foundry_client(host, token)
    result = client.datasets.Dataset.create(name=foundry_dataset_name, parent_folder_rid=parent_folder_rid)
    upload_file_to_foundry_dataset(client, result.rid, filepath_local, foundry_filename=foundry_filename)
