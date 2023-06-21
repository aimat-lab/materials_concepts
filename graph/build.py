import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from ast import literal_eval
from nltk import pos_tag
import re
from fire import Fire

ORIGIN_DAY = pd.to_datetime("1970-01-01")


def str_to_set(str):
    if isinstance(str, float):
        return set()
    if str == "":
        return set()
    return set(str.split(","))


class OccurenceFilter:
    def __init__(self, min_occurence=None, max_occurence=None):
        self.min_occurence = min_occurence
        self.max_occurence = max_occurence

    def __call__(self, concepts):
        print(
            f"Applying occurence filter: min={self.min_occurence}, max={self.max_occurence}"
        )
        concepts = {
            concept: n
            for concept, n in tqdm(concepts.items())
            if (self.min_occurence is None or n >= self.min_occurence)
            and (self.max_occurence is None or n <= self.max_occurence)
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


class LlamaFilter:
    def __call__(self, concepts):
        print("Applying LLaMa filter")
        concepts = {
            concept: n
            for concept, n in tqdm(concepts.items())
            if not re.findall(r"\b\d+\b", concept)  # no standalone numbers
        }

        return concepts


class NonAsciiFilter:
    def __call__(self, concepts):
        print("Applying non-ascii filter")
        concepts = {
            concept: n
            for concept, n in tqdm(concepts.items())
            if all(ord(c) < 128 for c in concept)
        }

        return concepts


class WordFilter:
    def __init__(self, min_n=1, max_n=10):
        self.min_n = min_n
        self.max_n = max_n

    def __call__(self, concepts):
        print(f"Applying word filter: min={self.min_n}, max={self.max_n}")
        concepts = {
            concept: n
            for concept, n in tqdm(concepts.items())
            if self.min_n <= len(concept.split(" ")) <= self.max_n
        }

        return concepts


class ElementFilter:
    """Filters non-valid elements"""

    def __init__(self, min_amount_elements=2):
        self.min_amount_elements = min_amount_elements

    @staticmethod
    def get_elements(string):
        counts = {}
        _buffer = ""
        for c in string:
            if c.isupper():  # new element
                if _buffer != "":
                    counts[_buffer] = counts.get(_buffer, 0) + 1

                _buffer = c

            elif c.isalpha():  # no number
                _buffer += c

        counts[_buffer] = counts.get(_buffer, 0) + 1

        return counts

    @staticmethod
    def is_valid_element(elements: dict):
        return all(
            n <= 1 for n in elements.values()
        )  # no duplicates in elements allowed

    def __call__(self, elements):
        print("Applying elements filter")
        elements = {
            element: n
            for element, n in tqdm(elements.items())
            if len(self.get_elements(element)) >= self.min_amount_elements
            and self.is_valid_element(self.get_elements(element))
        }

        return elements


E_FILTERS = [OccurenceFilter(min_occurence=3), ElementFilter(min_amount_elements=2)]

settings = {
    "rake_concepts": {
        "parse_func": str_to_set,
        "filters": [OccurenceFilter(min_occurence=3), RakeFilter()],
    },
    "llama_concepts": {
        "parse_func": literal_eval,
        "filters": [
            OccurenceFilter(min_occurence=3),
            LlamaFilter(),
            NonAsciiFilter(),
            WordFilter(min_n=3, max_n=10),
        ],
    },
}


def main(
    input_file="data/materials-science.llama.works.csv",
    colname="llama_concepts",
):
    df = pd.read_csv(input_file)
    df[colname] = df[colname].apply(settings[colname]["parse_func"])
    df.elements = df.elements.apply(str_to_set)

    concepts = [concept for s in tqdm(df[colname]) for concept in s]
    elements = [element for s in tqdm(df.elements) for element in s]

    concepts = Counter(concepts)
    elements = Counter(elements)

    for filter in settings[colname]["filters"]:
        concepts = filter(concepts)

    for filter in E_FILTERS:
        elements = filter(elements)

    concept_list = list(concepts.keys()) + list(elements.keys())

    # transform concepts into numbers
    lookup = {}
    lookup_list = []

    for index, concept in enumerate(sorted(concept_list)):
        concept = concept.strip()
        lookup_list.append(
            {
                "id": index,
                "concept": concept,
                "count": concepts.get(concept, elements.get(concept, 0)),
            }
        )
        lookup[concept] = index

    # save lookup as csv
    lookup_df = pd.DataFrame(lookup_list)
    lookup_df.to_csv("graph/lookup.csv", index=False)

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
    for concept_list, pub_date in tqdm(list(zip(df[colname], df.pub_date_days))):
        concept_ids = {
            lookup[c] for c in concept_list if lookup.get(c) is not None
        }  # set comprehension because rake/llms don't necessarily filter out duplicates

        for v1, v2 in get_pairs(concept_ids):
            all_edges.append(np.array((v1, v2, pub_date)))

    all_edges = np.array(all_edges)

    print(f"# edges: {len(all_edges): .0f}")

    np.savez_compressed("graph/edges.npz", all_edges)

    # load edges
    # edges = np.load("graph/edges.npz", allow_pickle=True)["arr_0"]


if __name__ == "__main__":
    Fire(main)
