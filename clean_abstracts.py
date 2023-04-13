import pandas as pd
from tqdm import tqdm
from utils.preprocessing import prepare, filter_common_errors
from utils.chem_tokenizer import get_tokens, merge_tokens, filter_element_tokens
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from utils.utils import Timer

tqdm.pandas()


def set_to_str(x):
    return ",".join(sorted(x)) if x else ""


def str_to_set(x):
    return set(x.split(",")) if x else set()


def process_df(partial_df):
    partial_df["new_abstract"] = partial_df["abstract"].copy()
    partial_df["new_abstract"] = partial_df["new_abstract"].progress_apply(prepare)
    partial_df["tokens"] = partial_df["new_abstract"].progress_apply(get_tokens)

    partial_df["new_abstract"] = partial_df.tokens.progress_apply(merge_tokens)
    partial_df["elements"] = partial_df.tokens.progress_apply(
        filter_element_tokens
    ).progress_apply(filter_common_errors)
    return partial_df


X = 3
N_CHUNKS = 8

with Timer("Processing dfs..."):
    df = pd.read_csv(f"data/split/output_{X}.csv")

    tasks = np.array_split(df, N_CHUNKS, axis=0)

    with ThreadPoolExecutor() as executor:
        result = executor.map(process_df, tasks)

    df = pd.concat(result)

    all_elements = set().union(*df.elements)
    df["elements"] = df.elements.progress_apply(set_to_str)

    del df["tokens"]
    del df["abstract"]
    df.rename(columns={"new_abstract": "abstract"}, inplace=True)

    # save df
    df.to_csv(f"data/split/elements_output_{X}.csv", index=False)
    # save all elements
    open(f"all_elements_{X}.txt", "w").write("\n".join(sorted(all_elements)))
