import pandas as pd
from ast import literal_eval

df = pd.read_csv("report/fixed_concepts.csv")
df["llama_concepts"] = df.llama_concepts.apply(literal_eval).apply(set)


author_concepts = df.groupby("source")["llama_concepts"].agg(
    lambda sets: set().union(*sets)
)

author_concepts.reset_index()
