# SPIKE - Writing from our laptop to ACTUAL FDP data sets

The ticket ([DTOSS-9018](https://nhsd-jira.digital.nhs.uk/browse/DTOSS-9018)) has more context and notes.

TODO - add detail

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
