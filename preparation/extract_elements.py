import pandas as pd
from tqdm import tqdm
from chem_tokenizer import get_tokens, merge_tokens, filter_element_tokens

import sys, os

# Add the parent directory to sys.path
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_directory not in sys.path:
    sys.path.append(parent_directory)

from utils.utils import apply_in_parallel

tqdm.pandas()

COMMON_ERRORS = {
    "In",  #        word
    "At",  #        word
    "VIP",  #       abbr
    "C",  #         Celsius
    "K",  #         Kelvin
    "As",  #        word
    "I",  #         Roman numeral
    "II",  #        Roman numeral
    "III",  #       Roman numeral
    "IV",  #        Roman numeral
    "V",  #         Roman numeral || Volt
    "Pa",  #        Pascal
    "UV",  #        Ultraviolet
    "U",  #         ?
    "K1",  #         Thermal conductivity
}


def filter_common_errors(formulas: set) -> set:
    return formulas - COMMON_ERRORS


def set_to_str(x):
    return ",".join(sorted(x)) if x else ""


def extract_elements(partial_df):
    """
    Extract elements from abstracts to new column 'elements' and
    replace abstracts with corresponding version where all detected
    elements are merged into one 'word' (e.g. 'NaCl' instead of 'Na Cl').
    """

    partial_df["tokens"] = partial_df["abstract"].progress_apply(get_tokens)

    partial_df["abstract"] = partial_df.tokens.apply(
        merge_tokens
    )  # merge tokens back to abstract

    partial_df["elements"] = (
        partial_df.tokens.apply(filter_element_tokens)
        .apply(filter_common_errors)
        .apply(set_to_str)
    )

    del partial_df["tokens"]

    return partial_df


def main(input_file, folder, n_jobs):
    df = pd.read_csv(os.path.join(folder, input_file))

    df = apply_in_parallel(df, extract_elements, n_jobs)

    topic = input_file.split(".")[0]
    output_file = os.path.join(folder, f"{topic}.elements.works.csv")
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script to extract elements from abstracts and merge detected ones into one word in the abstract text."
    )
    parser.add_argument(
        "works_file",
        help="The .csv file containing the works whose abstracts should be processed.",
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

    main(input_file=args.works_file, folder=args.folder, n_jobs=args.njobs)
