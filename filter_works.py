import pandas as pd
import argparse
from langdetect import detect
from langdetect import DetectorFactory

from utils.utils import Timer

DetectorFactory.seed = 0  # deterministic results: https://pypi.org/project/langdetect/

CSV_FILE_IN = "data/works.csv"
CSV_FILE_OUT = "data/works_filtered.csv"


def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "unknown"


def add_language(df):
    df["lang"] = df["abstract"].apply(detect_language)
    return df


def filter(df, func):
    return df[func(df)]


has_abstract = lambda df: df["abstract"].notnull()
has_title = lambda df: df["display_name"].notnull()
not_paratext_or_retracted = lambda df: ~df["is_paratext"] & ~df["is_retracted"]
is_english = lambda df: df["lang"] == "en"


def main(file_in, file_out):
    df = pd.read_csv(file_in)
    with Timer(name="Filtering works"):
        df = filter(df, has_title)
        df = filter(df, has_abstract)
        df = filter(df, not_paratext_or_retracted)
        df = add_language(df)
        df = filter(df, is_english)
        del df["lang"]

    df.to_csv(file_out, index=False)


if __name__ == "__main__":
    # argparse here
    parser = argparse.ArgumentParser(
        description="Filter works from a CSV file, and save the result to another CSV file."
    )
    parser.add_argument("file_in", help="Input file name e.g. 'works.csv'")
    parser.add_argument(
        "file_out",
        nargs="?",
        default="filtered_works.csv",
        help="Output file name (default: 'filtered_works.csv')",
    )

    args = parser.parse_args()

    main(args.file_in, args.file_out)
