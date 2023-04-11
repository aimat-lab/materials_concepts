# custom imports
from Downloader import OADownloader
from utils.utils import inverted_abstract_to_abstract, extract_concepts

WORKS_URL = "https://api.openalex.org/works"

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
    per_page=200,  # 200
    fetch_limit=1000,  # limit is 10.000, implement cursor fetching for more
    filter="host_venue.id:S4210205488",
)

df = downloader.get().to_df(
    converters={
        "id": lambda x: x.rsplit("/", 1)[1],
        "abstract_inverted_index": inverted_abstract_to_abstract,
        "concepts": extract_concepts,
    }
)

df = df.rename(columns={"abstract_inverted_index": "abstract"})

df.to_csv("../data/works.csv", index=False)
