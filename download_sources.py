from Downloader import OADownloader
from utils.utils import make_valid_filename

TOPIC = "material science"
SOURCES_URL = f"https://api.openalex.org/sources?search={TOPIC}"

downloader = OADownloader(
    url=SOURCES_URL,
    fields=["id", "display_name", "works_count", "type", "alternate_titles"],
    per_page=200,  # 200 is the maximum
)

cleaned_topic = make_valid_filename(TOPIC).replace(" ", "-")
filename = f"{cleaned_topic}_sources.csv"

downloader.get().to_csv(
    filename=filename,
    converters={
        "alternate_titles": lambda x: x[0]
        if x
        else "",  # keep first if there is a non-empty list
        "id": lambda x: x.rsplit("/", 1)[1],  # remove the URL prefix
    },
)
