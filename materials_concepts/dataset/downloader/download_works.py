import os

import click
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from materials_concepts.dataset.downloader.Downloader import Converter, OADownloader
from materials_concepts.utils.utils import Timer

WORKS_URL = "https://api.openalex.org/works"

FIELDS = [
    "id",
    "doi",
    "display_name",
    "publication_date",
    "is_retracted",
    "is_paratext",
    "abstract_inverted_index",
    "concepts",
]


def extract_concepts(concepts):
    return [(c["display_name"], c["level"], c["score"]) for c in concepts]


def inverted_abstract_to_abstract(inverted_abstract):
    if not inverted_abstract:
        return ""

    ab_len = -1
    for _, value in inverted_abstract.items():
        for index in value:
            ab_len = max(ab_len, index)

    abstract = [" "] * (ab_len + 1)
    for key, value in inverted_abstract.items():
        for i in value:
            abstract[i] = key
    return " ".join(abstract)


def invert_abstract_and_clean(abstract_inverted_index):
    abstract = inverted_abstract_to_abstract(abstract_inverted_index)
    return abstract.replace("\n", " ").replace("\r", " ").replace("\r\n", " ")


def create_file(filename: Path):
    with open(filename, "w") as _:
        pass  # create file


def append_to_file(filename: Path, df: pd.DataFrame):
    """Append a DataFrame to a file. If the file is empty, write the header."""
    with open(filename, "a") as f:
        file_empty = f.tell() == 0
        df.to_csv(f, header=file_empty, index=False)


converter = Converter(
    {
        "id": lambda x: x.rsplit("/", 1)[1],
        # invert abstract and remove newlines
        "abstract_inverted_index": invert_abstract_and_clean,
        "concepts": extract_concepts,
    }
)


def merge_files(source_file: Path, merged_file: Path, cache_folder: Path):
    create_file(merged_file)

    df = pd.read_csv(source_file)
    for source_id in tqdm(df["id"]):
        # load
        source_filename = cache_folder / f"{source_id}.csv"
        # append to merge file
        append_to_file(filename=merged_file, df=pd.read_csv(source_filename))
        # clean up source file
        os.remove(source_filename)


def get_line_count(filepath: Path):
    with open(filepath) as file:
        line_count = sum(1 for _ in file) - 1  # subtract 1 for the header
    return line_count


def file_cached(source: str, cache_folder: Path, works_count: int):
    # This won't always work because the amount of sources available is
    # not always the same as the 'works_count'.
    # Therefore, accept if the file is within a certain tolerance of the works_count.
    _tolerance = 30

    cache_file = cache_folder / f"{source}.csv"

    return (
        cache_file.exists() and get_line_count(cache_file) >= works_count - _tolerance
    )


@click.group()
def cli():
    pass


@cli.command()
@click.option("--source", default="S26125866")
@click.option("--out", default="data/table/S26125866.works.csv")
@click.option("--fetch-limit", default=None, type=int)
def fetchsingle(source: str, out: Path, fetch_limit: int | None, handler=None):

    downloader = OADownloader(
        url=WORKS_URL,
        fields=FIELDS,
        handler=handler,
        fetch_limit=fetch_limit,
        filter=f"primary_location.source.id:{source}",
    )

    with Timer("Download time:"):
        df = downloader.get().to_df(converter=converter)  # file handler stores data
        df["source_id"] = source
        df = df.rename(
            columns={"abstract_inverted_index": "abstract"}
        )  # after generating the abstract from the inverted index, the column name should be changed
        df.to_csv(out, index=False)


@cli.command()
@click.option("--sources", default="data/table/materials-science.sources.csv")
@click.option("--out", default="data/table/materials-science.works.csv")
@click.option("--fetch-limit", default=None, type=int)
@click.option("--cache", default="/tmp/materials_concepts/.cache/sources/")
def fetchall(
    sources: str,
    out: str = "data/table/materials-science.works.csv",
    fetch_limit: int | None = None,
    cache: str = "/tmp/materials_concepts/.cache/sources/",
):
    cache_folder = Path(cache)
    cache_folder.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Cache folder: {cache_folder}")

    df = pd.read_csv(sources)
    sources_count = len(df)
    logger.info("Fetching works for each source...")
    for index, (source_id, display_name, works_count) in enumerate(
        zip(df["id"], df["display_name"], df["works_count"], strict=False)
    ):
        if file_cached(source_id, cache_folder, works_count):
            logger.info(
                f"({index+1}/{sources_count}) Skipping: {works_count} works already cached for {display_name} ({source_id})..."
            )
            continue

        logger.info(
            f"({index+1}/{sources_count}) Fetching {works_count} works for {display_name} ({source_id})..."
        )
        fetchsingle.callback(
            source=source_id,
            out=cache_folder / f"{source_id}.csv",
            fetch_limit=(fetch_limit or works_count),
        )

    logger.info("Creating merged file...")
    merge_files(Path(sources), Path(out), cache_folder)


if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
