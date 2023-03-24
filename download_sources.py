from Downloader import OADownloader

TOPIC = "material"
SOURCES_URL = f"https://api.openalex.org/sources?search={TOPIC}"

downloader = OADownloader(
    url=SOURCES_URL,
    fields=["id", "display_name", "works_count", "type", "alternate_titles"],
    per_page=200,  # 200 is the maximum
)

filename = f"data/{TOPIC.replace(' ', '-')}_sources.csv"
downloader.get().to_csv(
    filename=filename,
    converters={
        "alternate_titles": lambda x: x[0] if x else "",
        "id": lambda x: x.rsplit("/", 1)[1],
    },
)
