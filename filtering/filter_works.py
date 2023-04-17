import pandas as pd
from langdetect import detect
from langdetect import DetectorFactory
from tqdm import tqdm
from ast import literal_eval
from concurrent.futures import ProcessPoolExecutor
import numpy as np

tqdm.pandas()

DetectorFactory.seed = 0  # deterministic results: https://pypi.org/project/langdetect/

MIN_ABSTRACT_LENGTH = 100
MAX_ABSTRACT_LENGTH = 4000


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
    before = len(df)
    if name is not None:
        print("Applying filter: ", name, end="")
    print(f" ({before} -> ", end="")
    filtered = df[func(df)]
    after = len(filtered)
    print(f"{after}) [{before - after} filtered]")
    return filtered


has_abstract = lambda df: df["abstract"].notnull()
has_title = lambda df: df["display_name"].notnull()
not_paratext_or_retracted = lambda df: ~df["is_paratext"] & ~df["is_retracted"]
is_english = lambda df: df["lang"] == "en"
abstract_length_ok = lambda df: (df.length >= MIN_ABSTRACT_LENGTH) & (
    df.length <= MAX_ABSTRACT_LENGTH
)
has_no_latex = lambda df: ~df.abstract.str.contains(r"\\[a-z]+")
mat_science_related = lambda df: df["mat_score"] > 0


def extract_concept_score(c_list: str, concept):
    c_list_parsed = literal_eval(c_list)
    for _concept, _, score in c_list_parsed:
        if _concept == concept:
            return round(float(score), 3)
    return -1


def add_materials_science_score(df):
    df["mat_score"] = df["concepts"].apply(
        lambda string: extract_concept_score(string, concept="Materials science")
    )
    return df


def apply_in_parallel(df, func, n_jobs=4):
    tasks = np.array_split(df, n_jobs, axis=0)  # split df along row axis
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        result = executor.map(func, tasks)

    return pd.concat(result)


def filter_df(df, n_jobs):
    df = filter(df, has_title, name="Has title")
    df = filter(df, has_abstract, name="Has abstract")
    df = filter(df, not_paratext_or_retracted, name="Is not paratext or retracted")

    df["length"] = df["abstract"].str.len()
    df = filter(
        df,
        abstract_length_ok,
        name=f"Abstract length between {MIN_ABSTRACT_LENGTH} and {MAX_ABSTRACT_LENGTH} characters",
    )

    df = filter(df, has_no_latex, name="No latex code present")

    print("Extracting materials science score...")
    df = add_materials_science_score(df)
    df = filter(df, mat_science_related, name="Materials science related")

    print("Detecting primary language...")
    df = apply_in_parallel(df, add_primary_language, n_jobs=n_jobs)
    df = filter(df, is_english, name="Primary language is english")

    df = add_secondary_language(df)  # if any other language parts are still present
    df = filter(df, is_english, name="No other languages present")

    df = df.drop(
        columns=[
            "is_paratext",
            "is_retracted",
            "lang",
            "length",
            "mat_score",
        ]
    )
    return df


def main(works_file, folder, n_jobs):
    import os

    input_file = os.path.join(folder, works_file)
    df = pd.read_csv(input_file)
    df = filter_df(df, n_jobs=n_jobs)

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

    parser.add_argument(
        "--njobs",
        help="How many processes should be used for the heavier tasks. Defaults to 8.",
        default=8,
    )

    args = parser.parse_args()

    main(works_file=args.works_file, folder=args.folder, n_jobs=args.njobs)
