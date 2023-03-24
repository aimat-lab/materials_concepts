from rake_nltk import Rake

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
        "ngrams_url",
    ],
    per_page=200,  # 200
    fetch_limit=1000,  # limit is 10.000, implement cursor fetching for more
    filter="host_venue.id:S26125866",
)

df = downloader.get().to_df(
    converters={
        "id": lambda x: x.rsplit("/", 1)[1],
        "abstract_inverted_index": inverted_abstract_to_abstract,
        "concepts": extract_concepts,
    }
)

df.to_csv("data/works.csv", index=False)

# r = Rake(min_length=2, max_length=5, language="english")
# r.extract_keywords_from_text(df.loc[1]["abstract_inverted_index"])
# keywords = r.get_ranked_phrases()
