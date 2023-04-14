from Downloader import OADownloader, InMemoryHandler, Converter
from helper import make_valid_filename

converter = Converter(
    {
        # keep first if there is a non-empty list
        "alternate_titles": lambda x: x[0] if x else "",
        # remove the URL prefix
        "id": lambda x: x.rsplit("/", 1)[1],
    }
)


def main(topic, folder):
    import os

    SOURCES_URL = f"https://api.openalex.org/sources?search={topic}"

    downloader = OADownloader(
        url=SOURCES_URL,
        fields=["id", "display_name", "works_count", "type", "alternate_titles"],
    )

    cleaned_topic = make_valid_filename(topic)
    filename = f"{cleaned_topic}_sources.csv"

    downloader.get().to_csv(
        filename=os.path.join(folder, filename),
        converter=converter,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process some topic and folder arguments."
    )
    parser.add_argument("topic", help="the topic to process")
    parser.add_argument("--folder", help="the folder to process", default="data/")
    args = parser.parse_args()

    main(topic=args.topic, folder=args.folder)
