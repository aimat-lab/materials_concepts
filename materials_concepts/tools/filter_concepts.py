from nltk import pos_tag
from tqdm import tqdm

MIN_OCCURENCES = 2
MIN_LEN = 7

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

CONSIDER_REMOVING = [
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


INPUT_FILE = "data/lists/all_concepts.txt"
OUTPUT_FILE = "data/lists/all_concepts_filtered.txt"

concepts = [
    line.split("  ")[0].strip()
    for line in open(INPUT_FILE).readlines()
    if int(line.split("  ")[1].strip()[1:-1]) >= MIN_OCCURENCES
]


m_concepts = [
    concept
    for concept in tqdm(concepts)
    if all(c not in REMOVE_IF_LEN_2 for c in concept.split(" "))
    and concept.count(" ") <= 1
]

m_concepts = [
    concept
    for concept in tqdm(m_concepts)
    if all(c not in CONSIDER_REMOVING for c in concept.split(" "))
]

m_concepts = [concept for concept in tqdm(m_concepts) if len(concept) >= MIN_LEN]

m_concepts = [concept for concept in tqdm(m_concepts) if is_meaningful(concept)]

print(f"Len after filtering: {len(m_concepts)}")
open(OUTPUT_FILE, "w").write("\n".join(m_concepts))
