import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from ast import literal_eval
from nltk import pos_tag

CONCEPTS_COLNAME = "rake_concepts"
INPUT_FILE = "data/materials-science.rake.works.csv"
ORIGIN_DAY = pd.to_datetime("1970-01-01")


def str_to_set(str):
    if isinstance(str, float):
        return set()
    if str == "":
        return set()
    return set(str.split(","))


parse_func = {"rake_concepts": str_to_set, "llama_concepts": literal_eval}


class OccuranceFilter:
    def __init__(self, min_occurance=None, max_occurance=None):
        self.min_occurance = min_occurance
        self.max_occurance = max_occurance

    def __call__(self, concepts):
        print(
            f"Applying occurance filter: min={self.min_occurance}, max={self.max_occurance}"
        )
        concepts = {
            concept: n
            for concept, n in tqdm(concepts.items())
            if (self.min_occurance is None or n >= self.min_occurance)
            and (self.max_occurance is None or n <= self.max_occurance)
        }

        return concepts


class MinLenFilter:
    def __init__(self, min_len):
        self.min_len = min_len

    def __call__(self, concepts):
        print(f"Applying min len filter: min={self.min_len}")
        concepts = {
            concept: n
            for concept, n in tqdm(concepts.items())
            if len(concept) >= self.min_len
        }

        return concepts


class RakeFilter:
    REMOVE_IF_LEN_2 = [
        "important",
        "also",
        "due",
        "thoroughly",
        "using",
        "showed",
        "much",
        "times",
    ]

    REMOVE_ALWAYS = [
        "results",
        "result",
        "recent",
        "recently",
        "present",
        "presented",
        "presents",
        "work",
        "works",
        "experiment",
        "experiments",
        "experimental",
        "paper",
        "papers",
    ]

    def __call__(self, concepts):
        print("Applying rake filter")

        concepts = {
            concept: n
            for concept, n in tqdm(concepts.items())
            if all(c not in self.REMOVE_IF_LEN_2 for c in concept.split(" "))
            and concept.count(" ") <= 1
        }

        concepts = {
            concept: n
            for concept, n in tqdm(concepts.items())
            if all(c not in self.REMOVE_ALWAYS for c in concept.split(" "))
        }

        concepts = {
            concept: n
            for concept, n in tqdm(concepts.items())
            if RakeFilter.is_meaningful(concept)
        }

        return concepts

    @staticmethod
    def is_meaningful(phrase):
        pos_tags = pos_tag(phrase.split(" "))
        pos_tags = [tag[1] for tag in pos_tags]

        # Remove phrases with adverbs, prepositions, conjunctions, or determiners in them if they are <= 2 words
        if (
            any(tag in {"RB", "IN", "CC", "DT", "JJ", "JJR"} for tag in pos_tags)
            and len(phrase.split(" ")) <= 2
        ):
            return False

        return True


df = pd.read_csv(INPUT_FILE)
df[CONCEPTS_COLNAME] = df[CONCEPTS_COLNAME].apply(parse_func[CONCEPTS_COLNAME])
df.elements = df.elements.apply(str_to_set)

print("Building concept list")
concepts = [concept for s in tqdm(df[CONCEPTS_COLNAME]) for concept in s]
elements = [element for s in tqdm(df.elements) for element in s]

concepts = Counter(concepts)
elements = Counter(elements)

c_filters = [OccuranceFilter(min_occurance=3), MinLenFilter(7), RakeFilter()]
e_filters = [OccuranceFilter(min_occurance=3)]

print(f"Number of concepts: {len(concepts)}")
for filter in c_filters:
    concepts = filter(concepts)
print(f"Number of concepts: {len(concepts)}")

print(f"Number of elements: {len(elements)}")
for filter in e_filters:
    elements = filter(elements)
print(f"Number of elements: {len(elements)}")

concept_list = list(concepts.keys()) + list(elements.keys())

# transform concepts into numbers
lookup = {}
for index, concept in enumerate(sorted(concept_list)):
    concept = concept.strip()
    lookup[concept] = index


# encode publication date as days since origin
df["pub_date_days"] = pd.to_datetime(df.publication_date).apply(
    lambda ts: (ts - ORIGIN_DAY).days
)


def get_pairs(items):
    pairs = []
    for i1 in items:
        for i2 in items:
            if i1 == i2:
                # this ensures that we don't get to the diagonal line in the pairing matrix
                # as order doesn't matter, this yields just half of the matrix (excluding the diagonal)
                break
            pairs.append((i1, i2))
    return pairs


print("Building edge list")
all_edges = []
for concept_list, pub_date in tqdm(list(zip(df[CONCEPTS_COLNAME], df.pub_date_days))):
    concept_ids = {
        lookup[c] for c in concept_list if lookup.get(c) is not None
    }  # set comprehension because rake doesn't filter out duplicates

    for v1, v2 in get_pairs(concept_ids):
        all_edges.append(np.array((v1, v2, pub_date)))


all_edges = np.array(all_edges)
np.savez_compressed("graph/edges.npz", all_edges)

# load edges
# edges = np.load("graph/edges.npz", allow_pickle=True)["arr_0"]
