from ast import literal_eval
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from langdetect import DetectorFactory, detect
from tqdm import tqdm

from typing import Callable
import click

from loguru import logger

tqdm.pandas()

DetectorFactory.seed = 0  # deterministic results: https://pypi.org/project/langdetect/


def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "unknown"


def add_primary_language(df):
    df["lang"] = df["abstract"].progress_apply(detect_language)
    return df


states = []


def filter_(df: pd.DataFrame, func: Callable, name=None) -> pd.DataFrame:
    before = len(df)
    if name is not None:
        logger.info(f"Applying filter: {name}")

    filtered = df[func(df)]

    states.append((name, df[~func(df)]))

    after = len(filtered)
    logger.info(f"{before} -> {after} ({before - after} filtered)")
    return filtered


def has_abstract(df: pd.DataFrame) -> pd.Series:
    return df["abstract"].notnull()


def has_title(df: pd.DataFrame) -> pd.Series:
    return df["display_name"].notnull()


def not_paratext_or_retracted(df: pd.DataFrame) -> pd.Series:
    return ~df["is_paratext"] & ~df["is_retracted"]


def is_english(df: pd.DataFrame) -> pd.Series:
    return df["lang"] == "en"


def abstract_length_ok(df: pd.DataFrame) -> pd.Series:
    return (df["abstract"].str.len() >= MIN_ABSTRACT_LENGTH) & (
        df["abstract"].str.len() <= MAX_ABSTRACT_LENGTH
    )


def has_no_latex(df: pd.DataFrame) -> pd.Series:
    return ~df["abstract"].str.contains(r"\\[a-z]+")


def topic_related(df: pd.DataFrame) -> pd.Series:
    return df["topic_score"] > 0


def extract_concept_score(c_list: str, concept):
    c_list_parsed = literal_eval(c_list)
    for _concept, _, score in c_list_parsed:
        if _concept == concept:
            return round(float(score), 3)
    return -1


def add_topic_score(df: pd.DataFrame, topic: str):
    df["topic_score"] = df["concepts"].apply(
        lambda string: extract_concept_score(string, concept=topic)
    )
    return df


def apply_in_parallel(df, func, n_jobs=4):
    if n_jobs == 1:
        return func(df)

    tasks = np.array_split(df, n_jobs, axis=0)  # split df along row axis
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        result = executor.map(func, tasks)

    return pd.concat(result)


def verify_topic_is_openalex_concept(topic: str):
    logger.debug("Verifying topic is an OpenAlex concept...")
    import requests

    url = f"https://api.openalex.org/concepts?filter=display_name.search:{topic}"

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    if data["meta"]["count"] == 0:
        raise SystemExit(f"Concept '{topic}' not found in OpenAlex concepts.")


def filter_df(df: pd.DataFrame, n_jobs) -> pd.DataFrame:
    df = filter_(df, has_title, name="Has title")
    df = filter_(df, has_abstract, name="Has abstract")
    df = filter_(df, not_paratext_or_retracted, name="Is not paratext or retracted")

    df["length"] = df["abstract"].str.len()
    df = filter_(
        df,
        abstract_length_ok,
        name=f"Abstract length between {MIN_ABSTRACT_LENGTH} and {MAX_ABSTRACT_LENGTH} characters",
    )

    df = filter_(df, has_no_latex, name="No latex code present")

    if TOPIC is not None:
        verify_topic_is_openalex_concept(TOPIC)

        logger.info(f"Extracting {TOPIC} score...")
        df = add_topic_score(df, TOPIC)
        df = filter_(
            df,
            topic_related,
            name="Materials science related",
        )

    logger.info("Detecting primary language...")
    df = apply_in_parallel(df, add_primary_language, n_jobs=n_jobs)
    df = filter_(df, is_english, name="Primary language is english")

    df = filter_(df, is_english, name="No other languages present")

    df = df.drop(
        columns=[
            "is_paratext",
            "is_retracted",
            "lang",
        ]
    )
    return df


@click.command()
@click.option(
    "--source",
    default="data/table/materials-science.works.csv",
    help="Path to the input .csv file",
)
@click.option(
    "--out",
    default="data/table/materials-science.filtered.works.csv",
    help="Path to the output folder",
)
@click.option(
    "--njobs",
    default=8,
    help="How many processes should be used for the heavier tasks. Defaults to 8.",
    type=int,
)
@click.option(
    "--min-abstract-length",
    default=250,
    help="Minimum length of the abstract. Defaults to 250.",
    type=int,
)
@click.option(
    "--max-abstract-length",
    default=3000,
    help="Maximum length of the abstract. Defaults to 3000.",
    type=int,
)
@click.option(
    "--topic",
    default=None,
    help="An OpenAlex concept to filter the works. Defaults to None (no filtering). Works must contain this concept (tagged by OpenAlex) with score > 0.",
    # Spreadsheet of all concepts: https://docs.google.com/spreadsheets/d/1LBFHjPt4rj_9r0t0TTAlT68NwOtNH8Z21lBMsJDMoZg/edit#gid=575855905
)
def filter_works(
    source: str,
    out: str,
    njobs: int,
    min_abstract_length: int = 250,
    max_abstract_length: int = 3000,
    topic: str | None = None,
):
    global MIN_ABSTRACT_LENGTH, MAX_ABSTRACT_LENGTH, TOPIC
    MIN_ABSTRACT_LENGTH = min_abstract_length
    MAX_ABSTRACT_LENGTH = max_abstract_length
    TOPIC = topic

    df = pd.read_csv(source)
    df = filter_df(df, n_jobs=njobs)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    filter_works()
