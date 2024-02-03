# custom imports
import os

import pandas as pd
from tqdm import tqdm

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


def create_file(filename):
    with open(filename, "w") as _:
        pass  # create file


def append_to_file(filename, df):
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


def fetch_single(source, fetch_limit, folder, handler=None):
    filename = os.path.join(folder, f"{source}.csv")

    downloader = OADownloader(
        url=WORKS_URL,
        fields=FIELDS,
        handler=handler,
        fetch_limit=fetch_limit,
        filter=f"host_venue.id:{source}",
    )

    with Timer("Download time:"):
        df = downloader.get().to_df(converter=converter)  # file handler stores data
        df["source_id"] = source
        df = df.rename(
            columns={"abstract_inverted_index": "abstract"}
        )  # after generating the abstract from the inverted index, the column name should be changed
        df.to_csv(filename, index=False)


def merge_files(csv_file, folder):
    merged_file = os.path.join(
        folder,
        # materials-science.sources.csv -> materials-science.works.csv
        os.path.basename(csv_file).split(".")[0] + ".works.csv",
    )

    create_file(merged_file)

    df = pd.read_csv(csv_file)
    for source_id in tqdm(df["id"]):
        # load
        source_filename = os.path.join(folder, f"{source_id}.csv")
        # append to merge file
        append_to_file(filename=merged_file, df=pd.read_csv(source_filename))
        # clean up source file
        os.remove(source_filename)


def get_line_count(filepath):
    with open(filepath) as file:
        line_count = sum(1 for _ in file) - 1  # subtract 1 for the header
    return line_count


def file_cached(source, folder, works_count):
    # This won't always work because the amount of sources available is
    # not always the same as the 'works_count'.
    # Therefore, accept if the file is within a certain tolerance of the works_count.

    filename = os.path.join(folder, f"{source}.csv")

    TOLERANCE = 30
    return (
        os.path.exists(filename) and get_line_count(filename) >= works_count - TOLERANCE
    )


def fetch_multiple(csv_file, fetch_limit=None, folder="data/"):
    df = pd.read_csv(csv_file)
    sources_count = len(df)
    for index, (source_id, display_name, works_count) in enumerate(
        zip(df["id"], df["display_name"], df["works_count"], strict=False)
    ):
        if file_cached(source_id, folder, works_count):
            print(
                f"({index+1}/{sources_count}) Skipping: {works_count} works already cached for {display_name} ({source_id})..."
            )
            continue

        print(
            f"({index+1}/{sources_count}) Fetching {works_count} works for {display_name} ({source_id})..."
        )
        fetch_single(
            source=source_id, fetch_limit=(fetch_limit or works_count), folder=folder
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script to retrieve works for given source (enter ID) \
            or for a list of sources (enter path to .csv file) from OpenAlex."
    )
    parser.add_argument(
        "source",
        help="The source ID to filter by or a csv file containing the column 'id', 'display_name', and 'works_count'",
    )
    parser.add_argument(
        "--fetch_limit",
        type=int,
        default=None,
        help="The number of works to fetch. Defaults to all works.",
    )
    parser.add_argument(
        "--folder",
        default="data/",
        help="The output file (default: 'data/'))",
    )
    args = parser.parse_args()

    try:
        if args.source.endswith(".csv"):
            print("Fetching works for each source...")
            fetch_multiple(
                args.source, fetch_limit=args.fetch_limit, folder=args.folder
            )
            print("Creating merged file...")
            merge_files(args.source, args.folder)

        else:
            fetch_single(
                source=args.source, fetch_limit=args.fetch_limit, folder=args.folder
            )
    except KeyboardInterrupt:
        print("Interrupted by user")
