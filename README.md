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

> During fetching, this will create a `{source}.csv` file for each source in `{folder}/` listing all the works which belong to that source. After downloading, these are merged automatically into a single file `materials-science.works.csv` in the specified folder.
> If the download gets interrupted, the downloaded files serve as a cache. Re-run the script, it will automatically skip sources for which the data was already fetched.

## Data Filtering

Filter the data to improve its quality:

`$ python filtering/filter_works.py materials-science.works.csv --folder data/`

> This will output a file `materials-science.filtered.works.csv` in the specified folder containing all works which sufficed the conditions.

# TODO General

## Process

- [x] Create README with instructions to recreate progress
- [x] Implement cursor fetching
- [x] Work cleaning: Filter out works (no title, no abstract, retracted, paratext, english lang)
- [x] Abstract cleaning: Clean chemical elements
- [ ] Generate cleaned 'list' of all concepts
- [ ] Build graph with histogram edges
- [ ] Implement top performing model from kaggle challange
- [ ] Store model and graph
- [ ] Build API to query prediction service

## Optimization

- [ ] Where to store the data?
- [ ] Data storing for works: How to store concepts (fetched and generated)
- [ ] How to speed up text processing in pandas? Pandas 2.0 or other option to achieve pyarrow backend
- [ ] Dockerize what comes after data fetching

# My handy tools

## Abstract checking

Retrieve abstract for work given work ID: `$ abstract W2159161622`

## Identifying Sourcce

Retrieve source for work given work ID: `$ getsource W2159161622`
