# custom imports
from Downloader import OADownloader, FileHandler, Converter
import utils.utils as utils

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


def invert_abstract_and_clean(abstract_inverted_index):
    abstract = utils.inverted_abstract_to_abstract(abstract_inverted_index)
    return abstract.replace("\n", " ").replace("\r", " ").replace("\r\n", " ")


def fetch_single(host_venue, fetch_limit, output):
    converter = Converter(
        {
            "id": lambda x: x.rsplit("/", 1)[1],
            "abstract_inverted_index": invert_abstract_and_clean,
            "concepts": utils.extract_concepts,
        }
    )

    handler = FileHandler(
        output, fields=FIELDS, converter=converter, flush_threshold=5000
    )

    downloader = OADownloader(
        url=WORKS_URL,
        fields=FIELDS,
        per_page=200,  # 200 is max
        handler=handler,
        fetch_limit=fetch_limit,
        filter=f"host_venue.id:{host_venue}",
    )

    with utils.Timer("Downloading works"):
        downloader.get()  # file handler stores data


def fetch_multiple(csv_file):
    import pandas as pd
    import os

    subfolder = utils.make_valid_filename(os.path.basename(csv_file[:-4]))
    print("Creating folder:", subfolder)
    os.makedirs(f"data/{subfolder}/", exist_ok=True)

    df = pd.read_csv(csv_file)

    df_subset = zip(df["id"], df["display_name"], df["works_count"])

    venue_count = len(df)
    for index, (host_venue_id, display_name, works_count) in enumerate(df_subset):
        title = f"{host_venue_id}"
        file_name = f"data/{subfolder}/{title}.csv"

        print(
            f"{index+1}/{venue_count} Fetching {works_count} works for {display_name} ({host_venue_id})..."
        )
        fetch_single(host_venue_id, works_count, file_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "host_venue",
        help="The host venue ID to filter by or a csv file containing the column 'id', 'display_name', and 'works_count'",
    )
    parser.add_argument(
        "--fetch_limit",
        type=int,
        default=0,
        help="The number of works to fetch",
    )
    parser.add_argument(
        "--output",
        default="data/works.csv",
        help="The output file (default: 'data/works.csv'))",
    )
    args = parser.parse_args()

    try:
        if args.host_venue.endswith(".csv"):
            fetch_multiple(args.host_venue)
        else:
            fetch_single(args.host_venue, args.fetch_limit, args.output)
    except KeyboardInterrupt:
        print("Interrupted by user")
