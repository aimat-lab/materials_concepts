import re
import pandas as pd
import sys
import os
from tqdm import tqdm

# Add the parent directory to sys.path
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_directory not in sys.path:
    sys.path.append(parent_directory)

from utils.utils import apply_in_parallel

tqdm.pandas()


def filter_out_ascii(text):
    return "".join(
        char for char in text if ord(char) < 128
    )  # remove non-ascii (no whitespace)


def remove_multiple_whitespaces(text):
    text = re.sub(r"\s+", " ", text)
    return text


def remove_parenthesis(text):
    """TODO: test if this can be done after parsing chemical formulas"""
    text = re.sub(r"[\(\)]", "", text)
    return text


def remove_text_enumerations(text):
    # with roman numerals
    text = re.sub(r"\s+(i|ii|iii|iv|v)[\.\)]", "", text)
    # with a, b, c, d
    text = re.sub(r"\s+[a-e][\.\)]\s", "", text)

    return text


def basic_text_cleaning(text):
    text = filter_out_ascii(text)
    text = re.sub(r"\t+", " ", text)
    text = remove_multiple_whitespaces(text)
    text = remove_text_enumerations(text)
    text = text.strip()
    return text


def remove_authors(text):
    text = re.sub(r"authors.*abstract", "", text, flags=re.IGNORECASE)
    return text


def remove_acknowledgements(text):
    text = re.sub(r"acknowledgements", "", text, flags=re.IGNORECASE)
    return text


def remove_word(text, word):
    text = re.sub(word, "", text, flags=re.IGNORECASE)
    return text


def remove_intro(text):
    text = re.sub(r"i(\s*)ntroduction.*", "", text, flags=re.IGNORECASE)
    return text


def replace_oxide_numbers(text):
    # replace roman numerals with numbers
    text = text.replace("(I)", "1+")
    text = text.replace("(II)", "2+")
    text = text.replace("(III)", "3+")
    text = text.replace("(IV)", "4+")
    text = text.replace("(V)", "5+")
    text = text.replace("(VI)", "6+")
    text = text.replace("(VII)", "7+")
    text = text.replace("(VIII)", "8+")
    text = text.replace("(IX)", "9+")
    text = text.replace("(X)", "10+")
    return text


PAREN_REGEX = re.compile(r"[\(\)\[\]\|]")


def replace_parenthesis(text):
    text = PAREN_REGEX.sub("", text)
    return text


ABBR_REGEX = re.compile(r"[\s\(][A-Z]+s[\s\)\.,;]")


def replace_abbreviations(text):
    # Replaces abbreviations in upper case only if they are followed by an s and surrounded by either parenthesis or whitespace
    text = ABBR_REGEX.sub("", text)
    return text


SUBSUP_LATEX_REGEX = re.compile(r"\{su[bp]\s*([^}]+)\}")
SUBSUP_XML_REGEX = re.compile(r"<sup>([^<]+)</sup>|<sub>([^<]+)</sub>")


def sub_sup_script_cleaning(text):
    # Replaces sup/sub script: e.g. Eu{sup 3+} or {sup 5}D{sub 0}-{sup7}F{sub 2}
    text = SUBSUP_LATEX_REGEX.sub(r"\1", text)

    # Replaces sup/sub script: e.g. Cd<sup>2+</sup> or POCl<sub>3</sub>
    text = SUBSUP_XML_REGEX.sub(r"\1\2", text)
    return text


# 1500 o C/K or 1400oC
TEMP_C_K_REGEX = re.compile(r"\s+\d+(\.\d+)?\s*o?\s*[CK]")


def replace_temperatures(text):
    text = TEMP_C_K_REGEX.sub("", text)
    return text


# 10-20, 100-125, ...
NUMBER_RANGES = re.compile(r"\s(\d+)-(\d+)\s")


def replace_number_ranges(text):
    text = NUMBER_RANGES.sub("", text)
    return text


def clean_abstract(text):
    text = remove_authors(text)
    text = remove_word(text, "abstract")
    text = remove_word(text, "acknowledgements")
    text = remove_intro(text)
    return text


def clean_materials_science_abstract(text):
    text = sub_sup_script_cleaning(text)
    text = replace_temperatures(text)  # replace temperatures before number ranges
    text = replace_number_ranges(text)
    text = replace_oxide_numbers(text)
    text = replace_parenthesis(text)
    text = replace_abbreviations(text)
    return text


def prepare(text):
    text = basic_text_cleaning(text)
    text = clean_abstract(text)
    text = clean_materials_science_abstract(text)
    text = remove_multiple_whitespaces(text)
    return text.strip()


def prepare_df(df):
    df["abstract"] = df["abstract"].progress_apply(prepare)
    return df


def main(csv_file, folder, n_jobs):
    input_file = os.path.join(folder, csv_file)
    topic = csv_file.split(".")[0]
    output_file = os.path.join(folder, f"{topic}.cleaned.works.csv")

    df = pd.read_csv(input_file)

    df = apply_in_parallel(df, prepare_df, n_jobs=n_jobs)

    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Script to clean abstracts.")
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
    main(args.works_file, args.folder, n_jobs=args.njobs)
