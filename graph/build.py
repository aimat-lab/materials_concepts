import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from ast import literal_eval
from nltk import pos_tag
import re
from fire import Fire
import pickle

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


class LengthFilter:
    def __init__(self, min_length=2, max_length=100):
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self, concepts):
        print(f"Applying length filter: min={self.min_length}, max={self.max_length}")
        concepts = {
            concept: n
            for concept, n in tqdm(concepts.items())
            if self.min_length <= len(concept) <= self.max_length
        }

        return concepts


def main(
    input_path="data/materials-science.llama.works.csv",
    output_path="data/graph/edges.pkl",
    lookup_path="data/table/lookup.csv",
    colname="llama_concepts",
    min_occurence=3,
    min_words=1,
    max_words=10,
    min_length=5,
    min_occurence_elements=3,
    min_amount_elements=2,
):
    settings = {
        "rake_concepts": {
            "parse_func": str_to_set,
            "filters": [OccurenceFilter(min_occurence=min_occurence), RakeFilter()],
        },
        "llama_concepts": {
            "parse_func": lambda l: set(literal_eval(l)),
            "filters": [
                OccurenceFilter(min_occurence=min_occurence),
                LlamaFilter(),
                NonAsciiFilter(),
                WordFilter(min_n=min_words, max_n=max_words),
                LengthFilter(min_length=min_length),
            ],
        },
    }

    E_FILTERS = [
        OccurenceFilter(min_occurence=min_occurence_elements),
        ElementFilter(min_amount_elements=min_amount_elements),
    ]

    df = pd.read_csv(input_path)
    df[colname] = df[colname].apply(settings[colname]["parse_func"])

    df.elements = df.elements.apply(str_to_set)

    concepts = [concept.lower() for s in tqdm(df[colname]) for concept in s]
    elements = [element for s in tqdm(df.elements) for element in s]

    concepts = Counter(concepts)
    elements = Counter(elements)

    for filter in settings[colname]["filters"]:
        concepts = filter(concepts)

    for filter in E_FILTERS:
        elements = filter(elements)

    concept_list = [c.lower() for c in concepts.keys()] + list(elements.keys())

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

    for concept_list, element_list, pub_date, date, work_id in tqdm(
        list(
            zip(df[colname], df.elements, df.pub_date_days, df.publication_date, df.id)
        )
    ):
        works_concepts = [c.lower() for c in concept_list] + list(element_list)
        concept_ids = {
            lookup[c] for c in works_concepts if lookup.get(c) is not None
        }  # set comprehension because rake/llms don't necessarily filter out duplicates

        clique_pairs = get_pairs(concept_ids)
        for v1, v2 in clique_pairs:
            all_edges.append(np.array((v1, v2, pub_date)))

    all_edges = np.array(all_edges)
    nodes = set(all_edges[:, 0]).union(all_edges[:, 1])

    print(f"# nodes: {len(nodes):,.0f}")
    print(f"# edges: {len(all_edges):,.0f}")

    lookup_df["in_graph"] = lookup_df.id.apply(lambda id: id in nodes)
    lookup_df.to_csv(lookup_path, index=False)

    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "num_of_vertices": len(nodes),
                "edges": all_edges,
                "input_path": input_path,
                "lookup_path": lookup_path,
                "colname": colname,
                "min_occurence": min_occurence,
                "min_words": min_words,
                "max_words": max_words,
                "min_occurence_elements": min_occurence_elements,
                "min_amount_elements": min_amount_elements,
            },
            f,
        )


if __name__ == "__main__":
    Fire(main)
