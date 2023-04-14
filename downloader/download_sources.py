from Downloader import OADownloader, InMemoryHandler, Converter
from helper import make_valid_filename

TOPIC = "materials science"
SOURCES_URL = f"https://api.openalex.org/sources?search={TOPIC}"

downloader = OADownloader(
    url=SOURCES_URL,
    fields=["id", "display_name", "works_count", "type", "alternate_titles"],
    per_page=200,  # 200 is the maximum
    handler=InMemoryHandler(),
)


cleaned_topic = make_valid_filename(TOPIC)
filename = f"{cleaned_topic}_sources.csv"

converter = Converter(
    {
        "alternate_titles": lambda x: x[0]
        if x
        else "",  # keep first if there is a non-empty list
        "id": lambda x: x.rsplit("/", 1)[1],  # remove the URL prefix
    }
)

downloader.get().to_csv(
    filename=filename,
    converter=converter,
)
