import pandas as pd
from tqdm import tqdm
from preprocessing import prepare, filter_common_errors
from chem_tokenizer import get_tokens, merge_tokens, filter_element_tokens
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from utils.utils import Timer

tqdm.pandas()


def set_to_str(x):
    return ",".join(sorted(x)) if x else ""


def process_df(partial_df):
    partial_df["new_abstract"] = partial_df["abstract"].copy()
    partial_df["new_abstract"] = partial_df["new_abstract"].progress_apply(prepare)
    partial_df["tokens"] = partial_df["new_abstract"].progress_apply(get_tokens)

    partial_df["new_abstract"] = partial_df.tokens.progress_apply(merge_tokens)
    partial_df["elements"] = (
        partial_df.tokens.progress_apply(filter_element_tokens)
        .progress_apply(filter_common_errors)
        .progress_apply(set_to_str)
    )

    del partial_df["tokens"]
    del partial_df["abstract"]
    partial_df.rename(columns={"new_abstract": "abstract"}, inplace=True)

    return partial_df


X = 4
N_CHUNKS = 8


def main():
    df = pd.read_csv(f"data/split/output_{X}.csv")

    tasks = np.array_split(df, N_CHUNKS, axis=0)

    result = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for processed_df in executor.map(process_df, tasks):
            result.append(processed_df)

    print("Concatenating dfs...")

    df = pd.concat(result)

    # save df
    df.to_csv(f"data/split/elements_output_{X}.csv", index=False)


if __name__ == "__main__":
    with Timer("Main:"):
        main()
