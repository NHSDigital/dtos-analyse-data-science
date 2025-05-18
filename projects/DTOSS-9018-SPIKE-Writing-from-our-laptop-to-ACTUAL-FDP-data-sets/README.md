# SPIKE - Writing from our laptop to ACTUAL FDP data sets

The ticket ([DTOSS-9018](https://nhsd-jira.digital.nhs.uk/browse/DTOSS-9018)) has more context and notes.

## Straw-man workflow

The `from-Service-Bus-to-Foundry` pipeline will trigger each time messages arrive on a Service Bus topic.
The Function App will:

1. Read as many messages as it can from the Service Bus topic (`maxMessageBatchSize` is [1,000](https://learn.microsoft.com/en-us/azure/azure-functions/functions-bindings-service-bus?tabs=isolated-process%2Cextensionv5%2Cextensionv3&pivots=programming-language-python)).
2. Create a new Foundry dataset with the name: `raw-nsp-events--<YYYY-MM-DD--HH-mm-ss>--<UUID>`
3. Write the messages as JSON lines to the dataset and commit the dataset transaction.
4. Tell the Service Bus to mark the messages as 'consumed'.
5. Repeat from step 1 until there are no more messages to process.

This process should cover BAU (low volume) and back-filling (large volume) use cases.

We note that:

1. If the Function App has scaled by n, horizontally then there will be n Functions writing to n Foundry datasets ***at the same time***, each with a unique timestamp-UUID combination.
2. We might need to do downstream processing to union or split raw data as necessary to get optimal file sizes for hadoop.
   1. See Palantir optimisation advice [here](https://www.palantir.com/docs/foundry/contour/performance-optimize#partitioning).

## Timings

FIXME - update with actual volumes from ticket.

### Results

| Description | n events total | n events per file | n key value pairs per event | File size (MB) | Upload time (s) |
| ----------- | -------: | -------: | --------------------------: | --------: | ----------: |
| One file to one dataset   | 100,000 | 100,000 | 100 | 235 | 35 |
| Two files to ***two separate*** datasets, concurrently   | 100,000 | 50,000 | 100 | 118 | 33 |
| Ten files to ***ten separate*** datasets, concurrently   | 100,000 | 10,000 | 100 | 24 | 30 |
| 1,000 files to ***1,000 separate*** datasets, concurrently   | 1,000 | 1 | 100 | ~0 | 70 |

'Two files to ***the same*** dataset' is not in the table because it didn't work.
We get [the error](https://www.palantir.com/docs/foundry/api/v2/general/overview/errors/?productId=foundry&slug=general&slug=overview&slug=errors): `OpenTransactionAlreadyExists`:

> A transaction is already open on this dataset and branch. A branch of a dataset can only have one open transaction at a time.

So we can't write multiple files to the same dataset at the same time.

### Commands

```bash
# Set up
cd projects/DTOSS-9018-SPIKE-Writing-from-our-laptop-to-ACTUAL-FDP-data-sets
python create_local_json_lines_file.py --n_events 1000 --n_kv_pairs_per_event 100

# One file to one dataset
python upload_file_to_foundry_dataset.py --filepath_local /tmp/dummy--100000-events-by-100-kv-pairs.jsonl

# Two files to ***the same*** dataset, concurrently - THIS FAILED - see note above
# NOTE that DATASET_RID must be set in .env for this to work
python upload_file_to_foundry_dataset.py --filepath_local /tmp/dummy--50000-events-by-100-kv-pairs.jsonl &
python upload_file_to_foundry_dataset.py --filepath_local /tmp/dummy--50000-events-by-100-kv-pairs.jsonl &

# Two files to ***two separate*** datasets, concurrently
python upload_file_to_foundry_dataset.py --filepath_local /tmp/dummy--50000-events-by-100-kv-pairs.jsonl &
python upload_file_to_foundry_dataset.py --filepath_local /tmp/dummy--50000-events-by-100-kv-pairs.jsonl &

# Ten files to ***ten separate*** datasets, concurrently
TEST_RESULT_FOLDER=/tmp/test_results_ten_files
rm -rf $TEST_RESULT_FOLDER
mkdir $TEST_RESULT_FOLDER
for i in {1..10}; do
  printf -v num "%02d" "$i"
  python upload_file_to_foundry_dataset.py --filepath_local /tmp/dummy--10000-events-by-100-kv-pairs.jsonl > $TEST_RESULT_FOLDER/upload_log_${num}.log 2>&1 &
done
wait

# Sense check the earliest and latest timestamps
grep 'uploaded to foundry' ${TEST_RESULT_FOLDER}/* | wc -l | awk '{print "N successful uploads: " $1}'
cat $TEST_RESULT_FOLDER/upload_log_* | grep ^2025 | sort | head -n 1 | awk -F' - ' '{print "Earliest timestamp: "$1}'
cat $TEST_RESULT_FOLDER/upload_log_* | grep ^2025 | sort | tail -n 1 | awk -F' - ' '{print "Latest timestamp: "$1}'

# 1,000 files to ***1,000 separate*** datasets, concurrently
TEST_RESULT_FOLDER=/tmp/test_results_1000_files
rm -rf $TEST_RESULT_FOLDER
mkdir $TEST_RESULT_FOLDER
for i in {1..1000}; do
  printf -v num "%04d" "$i"
  python upload_file_to_foundry_dataset.py --filepath_local /tmp/dummy--1-events-by-100-kv-pairs.jsonl > $TEST_RESULT_FOLDER/upload_log_${num}.log 2>&1 &
done

# Sense check the earliest and latest timestamps
grep 'uploaded to foundry' ${TEST_RESULT_FOLDER}/* | wc -l | awk '{print "N successful uploads: " $1}'
cat $TEST_RESULT_FOLDER/upload_log_* | grep ^2025 | sort | head -n 1 | awk -F' - ' '{print "Earliest timestamp: "$1}'
cat $TEST_RESULT_FOLDER/upload_log_* | grep ^2025 | sort | tail -n 1 | awk -F' - ' '{print "Latest timestamp: "$1}'

```

## Troubleshooting

- If the 'Apply Schema' button isn't working in Foundry.
  - Make sure the data uploaded is in bytes format.

- If Foundry's Preview doesn't show enough rows.
  - Open a Jupyter notebook and run:

```Python
from foundry.transforms import Dataset

dataset = Dataset.get("my_dataset")
files = dataset.files().download()
filename = next(iter(files.values()))
pd.read_json(filename, lines=True)
```
