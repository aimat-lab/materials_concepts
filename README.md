# Dataset Creation

## Create data/ folder

Create `data/` top-level folder.

## Data Fetching

Fetch all related sources: (`--folder data/` can be omitted)

`$ python downloader/download_sources.py 'materials science' --folder data/`

> This will create a `data/materials-science.sources.csv` file with all the sources.

Fetch works from single source:

`$ python downloader/download_works.py S82336448 --folder data/`

> This will create a `data/S82336448.csv` file with all the works belonging to that source.

Fetch works from all sources:

`$ python downloader/download_works.py data/materials-science.sources.csv --folder data/`

> This will create a `{source}.csv` file for each source in `data/materials-science_sources/` listing all the works which belong to that source.

# My handy tools

## Abstract checking

Retrieve abstract for work given work ID: `$ abstract W2159161622`

## Identifying Sourcce

Retrieve source for work given work ID: `$ getsource W2159161622`
