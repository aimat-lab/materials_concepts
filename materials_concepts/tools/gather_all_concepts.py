import pandas as pd
from collections import Counter
from tqdm import tqdm

CONCEPTS_COLNAME = "rake_concepts"


def str_to_set(str):
    if isinstance(str, float):
        return set()
    if str == "":
        return set()
    return set(str.split(","))


df = pd.read_csv("data/materials-science.rake.works.csv")
df[CONCEPTS_COLNAME] = df[CONCEPTS_COLNAME].apply(str_to_set)


phrases = [phrase for s in tqdm(df[CONCEPTS_COLNAME]) for phrase in s]

concept_counter = Counter(phrases)


concepts_formatted = [
    f"{phrase}  {occurance}" for phrase, occurance in concept_counter.most_common()
]

print(f"Found {len(concepts_formatted)} concepts")

with open("data/all_concepts.txt", "w") as f:
    f.write("\n".join(concepts_formatted))
