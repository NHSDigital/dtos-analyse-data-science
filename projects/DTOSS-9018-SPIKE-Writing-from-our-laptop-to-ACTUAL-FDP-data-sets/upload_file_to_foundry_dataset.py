import argparse
import logging
import os
import time

from dotenv import load_dotenv
import foundry_sdk
from foundry_sdk import FoundryClient
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)



def get_foundry_client(host: str, token: str) -> FoundryClient:
    return foundry_sdk.FoundryClient(
        auth=foundry_sdk.UserTokenAuth(token),
        hostname=host,
    )


def upload_file_to_foundry_dataset(client: FoundryClient, dataset_rid: str, filepath_local: str, filename_foundry: str) -> None:
    with open(filepath_local, "rb") as f:
        bytes_ = f.read()

    logger.info("Starting upload...")

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
    token: str = os.environ["BEARER_TOKEN"]
    host: str = os.environ["HOSTNAME"]
    dataset_rid: str = os.environ["DATASET_RID"]

    parser = argparse.ArgumentParser(description="Upload a file to a Foundry dataset.")
    parser.add_argument("filepath_local", type=str, help="Path to the local file to upload.")
    parser.add_argument("--filename_foundry", type=str, default=None, help="Foundry destination file path.")
    args = parser.parse_args()

    filepath_local: str = args.filepath_local
    filename_local: str = os.path.basename(filepath_local)
    filename_foundry: str = args.filename_foundry or filename_local

    client = get_foundry_client(host, token)
    upload_file_to_foundry_dataset(client, dataset_rid, filepath_local, filename_foundry=filename_local)
