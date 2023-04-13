import pandas as pd
from tqdm import tqdm
from utils.preprocessing import prepare, filter_common_errors
from utils.chem_tokenizer import get_tokens, merge_tokens, filter_element_tokens

tqdm.pandas()


def set_to_str(x):
    return ",".join(sorted(x)) if x else ""


def str_to_set(x):
    return set(x.split(",")) if x else set()


df = pd.read_csv("data/subset.csv")

df["new_abstract"] = df["abstract"].copy()
df["new_abstract"] = df["new_abstract"].progress_apply(prepare)
df["tokens"] = df["new_abstract"].progress_apply(get_tokens)

df["new_abstract"] = df.tokens.progress_apply(merge_tokens)
df["elements"] = df.tokens.progress_apply(filter_element_tokens).progress_apply(
    filter_common_errors
)

all_elements = set().union(*df.elements)
df["elements"] = df.elements.progress_apply(set_to_str)


del df["tokens"]
del df["abstract"]
df.rename(columns={"new_abstract": "abstract"}, inplace=True)

# save df
df.to_csv("data/prepared-compressed_works.csv", index=False)
# save all elements
open("all_elements.txt", "w").write("\n".join(sorted(all_elements)))
