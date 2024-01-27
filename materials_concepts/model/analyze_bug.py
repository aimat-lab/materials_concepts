from graph import Graph
from utils import prepare_dataframe
import pandas as pd
import pickle
import gzip

ORIGIN_DAY = pd.to_datetime("1970-01-01")


def load_compressed(path):
    with open(path, "rb") as f:
        compressed = f.read()
    return pickle.loads(gzip.decompress(compressed))


def get_concepts_until_table(df, year):
    cutoff = pd.to_datetime(f"{year+1}-01-01")

    # natual way: df = df[df.publication_date < cutoff]

    df = df[df.publication_date < cutoff]

    all_concepts = set()
    for concepts in df.concepts:
        if len(concepts) <= 1:
            continue
        all_concepts.update(concepts)

    return all_concepts


def get_concepts_until_graph(graph: Graph, year):
    return graph.get_vertices(until_year=year, min_degree=1)


df = prepare_dataframe(
    df=pd.read_csv("data/table/materials-science.llama.works.csv"),
    lookup_df=pd.read_csv("data/table/lookup/lookup_small.csv"),
    cols=["id", "concepts", "publication_date"],
)

table_concepts = set([c for concepts in df.concepts for c in concepts])
print("# concepts", len(table_concepts))


df_l = pd.read_csv("data/table/lookup/lookup_small.csv")
lookup_cs = {id: concept for id, concept in zip(df_l.id, df_l.concept)}
lookup_concepts = set(lookup_cs.values())
print("# lookup concepts", len(lookup_concepts))


print("\nLookup-Table:", len(lookup_concepts - table_concepts))
print("Table-Lookup:", len(table_concepts - lookup_concepts))

g = Graph.from_path("data/graph/edges_small.pkl")

print("Total vertices: ", len(g.vertices))
print("\n")


year = 2016
t_concepts = set(get_concepts_until_table(df, year))
g_concepts_id = {id for id in get_concepts_until_graph(g, year)}
g_concepts = {lookup_cs[id] for id in g_concepts_id}

avg_concepts = set(
    load_compressed("data/model/concept_embs/real_av_embs_2016.pkl.gz").keys()
)


print(year)
print("table", len(t_concepts))
print("graph", len(g_concepts))
print("av", len(avg_concepts))

print("diff", len(t_concepts - g_concepts))
print("diff", len(g_concepts - avg_concepts))

print()


# amt concepts 51874
# amt lookup concepts 51874

# Lookup-Table: 0
# Table-Lookup: 0
# Total vertices:  51818


# 2016
# table 48809
# graph 48809
# diff 0
