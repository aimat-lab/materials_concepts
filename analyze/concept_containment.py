import pandas as pd
from ast import literal_eval

df_v1 = pd.read_csv("data/table/materials-science.llama.works.csv")
df_v2 = pd.read_csv("data/table/materials-science.llama2.works.csv")


def prepare(df: pd.DataFrame):
    df.abstract = df.abstract.str.lower()
    df.llama_concepts = df.llama_concepts.apply(literal_eval).apply(set)


def compute_contained_ratio(df: pd.DataFrame):
    df["contained"] = df.apply(
        lambda row: sum(
            int(concept.lower() in row["abstract"]) for concept in row["llama_concepts"]
        ),
        axis=1,
    )

    df["contained_ratio"] = df["contained"] / df["llama_concepts"].apply(len)


def compute_containment_per_concept(df: pd.DataFrame):
    df["conceptwise_containment"] = df.apply(
        lambda row: [
            (concept, int(concept.lower() in row["abstract"]))
            for concept in row["llama_concepts"]
        ],
        axis=1,
    )

    concept_counts = {}
    concept_containment = {}
    for row in df["conceptwise_containment"]:
        for concept, is_contained in row:
            if concept not in concept_counts:
                concept_counts[concept] = 0
                concept_containment[concept] = 0

            concept_counts[concept] += 1
            concept_containment[concept] += is_contained

    return pd.concat(
        [pd.Series(concept_counts), pd.Series(concept_containment)], axis=1
    ).rename(columns={0: "count", 1: "containment"})


def analyze(df):
    df["ratio"] = df["containment"] / df["count"]
    df = df[df["count"] >= 3]
    print(
        "% of concepts contained 0 times: ", len(df[df["containment"] == 0] / len(df))
    )
    print("% of concepts where ratio is < 0.5: ", len(df[df["ratio"] < 0.5] / len(df)))


compute_contained_ratio(df_v1)
compute_contained_ratio(df_v2)

print("LLAMA 1 - more unnormalized concepts")
print(df_v1["contained_ratio"].describe())

print()
print("-" * 80)
print()

print("LLAMA 2 - less unnormalized concepts")
print(df_v2["contained_ratio"].describe())

print("\n" * 2)

prepare(df_v1)
prepare(df_v2)

df_c_v1 = compute_containment_per_concept(df_v1)
df_c_v2 = compute_containment_per_concept(df_v2)

print("LLAMA 1 - more unnormalized concepts")
analyze(df_c_v1)

print()
print("-" * 80)
print()

print("LLAMA 2 - less unnormalized concepts")
analyze(df_c_v2)
