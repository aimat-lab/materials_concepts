import re
from ast import literal_eval
from rake_nltk import Rake
import os, sys
from tqdm import tqdm
import pandas as pd
from keybert import KeyBERT

# Add the parent directory to sys.path
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_directory not in sys.path:
    sys.path.append(parent_directory)

from utils.utils import apply_in_parallel

tqdm.pandas()

model = KeyBERT("distilbert-base-nli-mean-tokens")
KEYWORD_LENGTH = 1

to_replace = {
    "bar",
    "cf",
    "cm",
    "et",
    "fg",
    "g",
    "ghz",
    "grit",
    "h",
    "k",
    "kda",
    "kg",
    "khz",
    "mah",
    "microg",
    "microl",
    "min",
    "mol",
    "mv",
    "nm",
    "pa",
    "ppm",
    "slpm",
    "u",
    "vol",
    "wt",
}


def before_rake(text):
    # Remove special characters, symbols, and numbers
    text = text.lower()
    text = re.sub(r"[\.,;]", " ", text)  # replace punctuation with space
    text = re.sub(r"[^a-z\s]", "", text)  # remove all non-alphabet characters
    text = re.sub(r"\s+", " ", text)  # remove extra spaces

    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in to_replace])
    return text


def list_to_string(l):
    return ",".join(l)


def parse_keywords(keywords):
    if keywords is None:
        return []

    comma_count = keywords.count(",")
    semicolon_count = keywords.count(";")
    delimiter = "," if comma_count > semicolon_count else ";"

    keywords = keywords.split(delimiter)

    TO_REMOVE = ".;,: "  # often a dot was found at the end of the last keyword
    stripped_keywords = [keyword.strip(TO_REMOVE) for keyword in keywords]
    return stripped_keywords


def get_keywords(text):
    splitted = re.split("Keywords", text)
    if len(splitted) > 1:
        return splitted[-1]
    else:
        return None


def extract_relevant_oa_concepts(text, threshold_score=0.5, threshold_level=1):
    c_list_parsed = literal_eval(text)
    keep = []
    for concept, level, score in c_list_parsed:
        if float(score) > threshold_score and int(level) >= threshold_level:
            keep.append(concept)
    return list_to_string(keep)


def extract_concepts(text):
    r = Rake(min_length=2, max_length=6, language="english")

    r.extract_keywords_from_text(before_rake(text))
    return list_to_string(r.get_ranked_phrases())


def extract_keywords(text):
    return list_to_string(parse_keywords(get_keywords(text)))


def extract_keybert(text):
    keywords = model.extract_keywords(
        text, keyphrase_ngram_range=(KEYWORD_LENGTH, KEYWORD_LENGTH), top_n=10
    )

    return list_to_string([keyword for keyword, _ in keywords])


METHODS = {
    "rake": extract_concepts,
    "keywords": extract_keywords,
    "openalex": extract_relevant_oa_concepts,
    "keyBERT": extract_keybert,
}


class FunctionApplication:
    def __init__(self, target, colname, method):
        self.target = target
        self.colname = colname
        self.method = method

    def __call__(self, df):
        df[self.colname] = df[self.target].progress_apply(self.method)
        return df


def main(filename, folder, n_jobs, method, colname):
    input_file = os.path.join(folder, filename)
    topic = filename.split(".")[0]
    output_file = os.path.join(folder, f"{topic}.{method}.works.csv")

    df = pd.read_csv(input_file)

    targetcolname = "abstract" if method != "openalex" else "concepts"
    apply_func = FunctionApplication(targetcolname, colname, METHODS[method])

    df = apply_in_parallel(
        df,
        apply_func,
        n_jobs=n_jobs,
    )

    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Script to extract concepts.")
    parser.add_argument(
        "works_file",
        help="The .csv file containing the works which should be filtered.",
    )

    parser.add_argument("method", help="rake|keywords|keyBERT", default="rake")

    parser.add_argument(
        "colname",
        help="The column name where the (extracted) concepts should be stored.",
    )

    parser.add_argument(
        "--folder",
        help="Where input file is located and where output file will be created. Defaults to 'data/'",
        default="data/",
    )

    parser.add_argument(
        "--njobs",
        help="How many processes should be used for the heavier tasks. Defaults to 8.",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    main(
        args.works_file,
        args.folder,
        n_jobs=args.njobs,
        method=args.method,
        colname=args.colname,
    )

# After concept extraction: Normalization via stemming and lemmatization
# what about n-grams?
