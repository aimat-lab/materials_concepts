import pandas as pd
from langdetect import detect
from langdetect import DetectorFactory
from tqdm import tqdm

tqdm.pandas()

DetectorFactory.seed = 0  # deterministic results: https://pypi.org/project/langdetect/

import datetime


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = datetime.datetime.now()
        return self

    def __exit__(self, *args):
        print(
            f"{self.name} time elapsed: {(datetime.datetime.now() - self.start).seconds} seconds..."
        )


def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "unknown"


def detect_other_language(text):
    """This can be used to detect (heuristically) if there are any non-english paragraphs in the text."""
    LANGUAGE = {
        "resumo": "pt",
        "autores": "pt",
        "auteurs": "fr",
        "autoren": "de",
        " und ": "de",
        " le ": "fr",
        " les ": "fr",
    }

    # not a complete list of all occuring languages
    # but the ones that were not filtered out by language detection
    # already will be filtered manually
    for word, lang in LANGUAGE.items():
        if word in text.lower():
            return lang

    return "en"


def add_primary_language(df):
    df["lang"] = df["abstract"].progress_apply(detect_language)
    return df


def add_secondary_language(df):
    # overwrite lang attribute, as after filtering out non-english abstracts before
    df["lang"] = df["abstract"].progress_apply(detect_other_language)
    return df


def filter(df, func, name=None):
    if name is not None:
        print("Applying filter: ", name)
    return df[func(df)]


has_abstract = lambda df: df["abstract"].notnull()
has_title = lambda df: df["display_name"].notnull()
not_paratext_or_retracted = lambda df: ~df["is_paratext"] & ~df["is_retracted"]
is_english = lambda df: df["lang"] == "en"


def filter_df(df):
    df = filter(df, has_title, name="Has title")
    df = filter(df, has_abstract, name="Has abstract")
    df = filter(df, not_paratext_or_retracted, name="Is not paratext or retracted")

    print("Detect primary language:")
    df = add_primary_language(df)
    df = filter(df, is_english, name="Primary language is english")

    print("Detect secondary language:")
    df = add_secondary_language(df)  # if any other language parts are still present
    df = filter(df, is_english, name="No other languages present")

    df = df.drop(columns=["lang"])
    return df


def main(works_file, folder):
    import os

    input_file = os.path.join(folder, works_file)
    df = pd.read_csv(input_file)
    df = filter_df(df)

    topic = works_file.split(".")[0]
    output_file = os.path.join(folder, f"{topic}.filtered.works.csv")
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script to filter works listed in a .csv file."
    )
    parser.add_argument(
        "works_file",
        help="The .csv file containing the works which should be filtered.",
    )
    parser.add_argument(
        "--folder",
        help="Where input file is located and where output file will be created. Defaults to 'data/'",
        default="data/",
    )
    args = parser.parse_args()

    with Timer(name="Filtering works"):
        main(works_file=args.works_file, folder=args.folder)
