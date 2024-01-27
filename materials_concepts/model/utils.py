from ast import literal_eval
import pandas as pd


def prepare_dataframe(df, lookup_df, cols):
    lookup = {key: True for key in lookup_df["concept"]}

    df.llama_concepts = df.llama_concepts.apply(literal_eval).apply(
        lambda x: list({c.lower() for c in x if lookup.get(c.lower())})
    )

    df.elements = df.elements.apply(
        lambda str: list({e for e in str.split(",") if lookup.get(e)})
        if not pd.isna(str)
        else []
    )

    df.publication_date = pd.to_datetime(df.publication_date)

    df.concepts = df.llama_concepts + df.elements
    df.concepts = df.concepts.apply(lambda x: sorted(x))  # sort

    return df[cols]
