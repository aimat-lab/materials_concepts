import click
import pandas as pd
from tqdm import tqdm
from utils.utils import apply_in_parallel

from materials_concepts.dataset.preparation.chem.chem_tokenizer import (
    filter_element_tokens,
    get_tokens,
    merge_tokens,
)

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


@click.command()
@click.option("--input", type=click.Path(exists=True))
@click.option("--output", type=click.Path())
@click.option(
    "--n_jobs",
    help="How many processes should be used for the heavier tasks. Defaults to 1.",
    default=1,
)
def main(input, output, n_jobs):
    df = pd.read_csv(input)

    df = (
        apply_in_parallel(df, extract_elements, n_jobs)
        if n_jobs > 1
        else extract_elements(df)
    )

    df.to_csv(output, index=False)


if __name__ == "__main__":
    main()
