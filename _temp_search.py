import pandas as pd

concepts = [
    "pxrd",
    "pxrd data",
    "pxrd dataset",
    "power x-ray diffraction",
    "power xray diffraction",
    "xray diffraction",
    "xrd",
    "power xray",
    "power xray dataset",
    "diffractogram",
]

df = pd.read_csv("data/table/materials-science.llama.works.csv")
df["abstract"] = df["abstract"].str.lower()

for concept in concepts:
    results = df[df["abstract"].str.contains(concept)]
    print(
        f"Concept: {concept} ->",
        len(results),
    )
