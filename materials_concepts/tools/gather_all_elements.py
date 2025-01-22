import pandas as pd
from collections import Counter

df = pd.read_csv("data/materials-science.elements.works.csv")


def str_to_set(str):
    if isinstance(str, float):
        return set()
    if str == "":
        return set()
    return set(str.split(","))


counter = Counter(
    element for elements in df.elements.apply(str_to_set) for element in elements
)

all_elements = set().union(*df.elements.apply(str_to_set))

all_elements = []

elements_formatted = [
    f"{phrase}  ({occurance})" for phrase, occurance in counter.most_common()
]

with open("data/all_elements.txt", "w") as f:
    f.write("\n".join(elements_formatted))

s = pd.Series(counter)
