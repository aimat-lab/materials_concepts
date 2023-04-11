# custom imports
from Downloader import OADownloader
from utils.utils import inverted_abstract_to_abstract, extract_concepts, Timer

WORKS_URL = "https://api.openalex.org/works"


def main(host_venue, fetch_limit, output):
    downloader = OADownloader(
        url=WORKS_URL,
        fields=[
            "id",
            "doi",
            "display_name",
            "publication_date",
            "is_retracted",
            "is_paratext",
            "abstract_inverted_index",
            "concepts",
        ],
        per_page=200,  # 200 is max
        fetch_limit=fetch_limit,  # limit is 10.000, implement cursor fetching for more
        filter=f"host_venue.id:{host_venue}",
    )

    with Timer("Downloading works"):
        df = downloader.get().to_df(
            converters={
                "id": lambda x: x.rsplit("/", 1)[1],
                "abstract_inverted_index": inverted_abstract_to_abstract,
                "concepts": extract_concepts,
            }
        )

    df = df.rename(columns={"abstract_inverted_index": "abstract"})

    df.to_csv(output, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "host_venue",
        help="The host venue ID to filter by",
    )
    parser.add_argument(
        "--fetch_limit",
        type=int,
        default=1000,
        help="The number of works to fetch",
    )
    parser.add_argument(
        "--output",
        default="data/works.csv",
        help="The output file (default: 'data/works.csv'))",
    )
    args = parser.parse_args()

    main(args.host_venue, args.fetch_limit, args.output)
