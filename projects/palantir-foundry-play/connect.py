import os

from dotenv import load_dotenv
from foundry import FoundryClient
import foundry
from pprint import pprint

load_dotenv()

PALANTIR_HOSTNAME = os.environ.get("PALANTIR_HOSTNAME")
PALANTIR_TOKEN = os.environ.get("PALANTIR_TOKEN")

client = FoundryClient(auth=foundry.UserTokenAuth(token=PALANTIR_TOKEN), hostname=PALANTIR_HOSTNAME)

dataset_rid = "FIXME"


from foundry.v2.datasets.errors import DatasetNotFound

try:
    response = client.datasets.Dataset.get(dataset_rid)

except DatasetNotFound as e:
    print("There was an error with the request", e.parameters)
