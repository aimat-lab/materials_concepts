import click

from materials_concepts.dataset.downloader.Downloader import Converter, OADownloader

converter = Converter(
    {
        "alternate_titles": lambda x: (
            x[0] if x else ""
        ),  # keep first if there is a non-empty list
        "id": lambda x: x.rsplit("/", 1)[1],  # remove the URL prefix
    }
)


@click.command(
    "Script to retrieve all sources (host venues) for a given topic from OpenAlex."
)
@click.option(
    "--query",
    default="materials science",
    help="The string to search for in the OpenAlex API when listing sources.",
)
@click.option(
    "--out",
    default="data/table/materials-science.sources.csv",
    help="The output file to save the sources to.",
)
def download_sources(
    query="materials science", out="data/table/materials-science.sources.csv"
):
    sources_url = f"https://api.openalex.org/sources?search={query}"

    downloader = OADownloader(
        url=sources_url,
        fields=["id", "display_name", "works_count", "type", "alternate_titles"],
    )

    downloader.get().to_csv(
        filename=out,
        converter=converter,
    )


if __name__ == "__main__":
    download_sources()
