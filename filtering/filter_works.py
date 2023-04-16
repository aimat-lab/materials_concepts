import pandas as pd
import os
from langdetect import detect
from langdetect import DetectorFactory

DetectorFactory.seed = 0  # deterministic results: https://pypi.org/project/langdetect/

INPUT_DIR = "data/materials-science_sources"
CSV_FILE_OUT = "data/works_filtered.csv"
INPUT_FILES = [file for file in os.listdir(INPUT_DIR) if file.endswith(".csv")][:3]


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


def filter_df(df):
    df = df.rename(columns={"abstract_inverted_index": "abstract"})
    df = filter(df, has_title)
    df = filter(df, has_abstract)
    df = filter(df, not_paratext_or_retracted)
    df = add_language(df)
    df = filter(df, is_english)
    df = df.drop(columns=["lang"])
    return df


def process_df(f):
    print(f"Processing {f}:", end=" ")
    df = pd.read_csv(os.path.join(INPUT_DIR, f))

    print(len(df), end=" -> ")
    df = filter_df(df)
    print(len(df))
    return df


dfs = [process_df(f) for f in INPUT_FILES]
result = pd.concat(dfs, ignore_index=True)
result = result.reset_index(drop=True)
result.to_csv(CSV_FILE_OUT, index=False)
