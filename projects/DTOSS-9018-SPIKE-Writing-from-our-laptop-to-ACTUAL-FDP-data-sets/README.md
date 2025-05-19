# SPIKE - Writing from our laptop to ACTUAL FDP data sets

## TLDR

1. Uploading batches of realistic-looking events to Foundry was reasonably quick: 200k events (or 450MB) per minute.
2. The upload speed of the network was _probably_ the bottleneck.
3. Lots of files, each with one event is bad because you have upload overhead per file and datasets that are [sub-optimal for analysis](https://www.palantir.com/docs/foundry/contour/performance-optimize#partitioning) in Foundry.
4. Concurrent uploads can't write to the same dataset; each must write to its own dataset object.

## Overview

Here, we test how long it takes to upload data to Foundry from a laptop.
We construct arbitrary, but realistic-looking datasets of 100,000 events (or messages) using [create_local_json_lines_file.py](./create_local_json_lines_file.py).
We then upload to Foundry using [upload_file_to_foundry_dataset.py](./upload_file_to_foundry_dataset.py).
We send either one dataset at a time or many datasets concurrently (up to the RAM and CPU resources of the laptop).
Each file is written to _it's own Foundry dataset object_ because a dataset (strictly a dataset branch) can only have one live transaction at a time.
It takes around 30 seconds to upload 235MB whether we use one, two, or ten concurrent processes.
My internet upload speed is ~60Mb/s or 7.5MB/s; and 235/7.5 = 31s.
So, the bottleneck is probably my internet upload speed.

We wanted to test small file uploads too, by sending 100,000 files, each with one event.
This was too slow and resource intensive, so we test using only 1,000 files instead, which still takes 70s
from the first process starting to the last one finishing.

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

1. If the Function App has scaled by n, horizontally then there will be n Functions writing to n Foundry datasets **_at the same time_**, each with a unique timestamp-UUID combination.
2. We might need to do downstream processing to union or split raw data as necessary to get optimal file sizes for hadoop.
   1. See Palantir optimisation advice [here](https://www.palantir.com/docs/foundry/contour/performance-optimize#partitioning).

## Timings

### Results

| Description | n events total | n events per file | n key value pairs per event | File size (MB) | Upload time (s) |
| ----------- | -------: | -------: | --------------------------: | --------: | ----------: |
| One file to one dataset   | 100,000 | 100,000 | 100 | 235 | 35 |
| Two files to **_two separate_** datasets, concurrently   | 100,000 | 50,000 | 100 | 118 | 33 |
| Ten files to **_ten separate_** datasets, concurrently   | 100,000 | 10,000 | 100 | 24 | 30 |
| 1,000 files to **_1,000 separate_** datasets, concurrently   | 1,000 | 1 | 100 | ~0 | 70 |

'Two files to **_the same_** dataset' is not in the table because it didn't work.
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

# Two files to **_the same_** dataset, concurrently - THIS FAILED - see note above
# NOTE that DATASET_RID must be set in .env for this to work
python upload_file_to_foundry_dataset.py --filepath_local /tmp/dummy--50000-events-by-100-kv-pairs.jsonl &
python upload_file_to_foundry_dataset.py --filepath_local /tmp/dummy--50000-events-by-100-kv-pairs.jsonl &

# Two files to **_two separate_** datasets, concurrently
python upload_file_to_foundry_dataset.py --filepath_local /tmp/dummy--50000-events-by-100-kv-pairs.jsonl &
python upload_file_to_foundry_dataset.py --filepath_local /tmp/dummy--50000-events-by-100-kv-pairs.jsonl &

# Ten files to **_ten separate_** datasets, concurrently
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

# 1,000 files to **_1,000 separate_** datasets, concurrently
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
