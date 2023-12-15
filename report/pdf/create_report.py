import pandas as pd

all_data = pd.read_csv("fixed_concepts.csv")

source = "pascal_friederich.txt"

author_name = "Pascal Friederich"

df = all_data[all_data["source"] == source]

import pickle

print("Loading data from friedrich.pickle")
with open("friederich.pickle", "rb") as f:
    data: dict = pickle.load(f)

# Example
# data = {
#     "concept1": [{"concept": "XYZ", "score": 0.5}],
#     "concept2": [{"concept": "ABC", "score": 0.99}],
# }

TOP_N = 150

all_pairs = []

print("Converting to pairs")
for concept, predictions in data.items():
    for prediction in predictions:
        all_pairs.append((concept, prediction["concept"], prediction["score"]))

print("Sorting pairs")
all_pairs = sorted(all_pairs, key=lambda x: x[2], reverse=True)

all_pairs = all_pairs[:TOP_N]

print(all_pairs)


keywords = data.keys()
keyword_combinations = all_pairs

from pylatex import Document, Section, Command, NoEscape, Itemize
from pylatex.utils import italic, bold

# Create a document
doc = Document("basic")

# Add document title
doc.preamble.append(Command("title", "Research Keywords Analysis"))
doc.preamble.append(Command("author", author_name))
doc.preamble.append(Command("date", Command("today")))
doc.append(NoEscape(r"\maketitle"))

# Add introductory section
with doc.create(Section("Introduction")):
    doc.append("This document presents a comprehensive analysis of research keywords.")

# Add section for listing keywords
with doc.create(Section("Researcher Keywords")):
    doc.append(
        "The following keywords were extracted using a fine-tuned version of 'LlaMa-2-13B' from abstracts of your published papers:\n"
    )

    # Assuming you have a list of keywords

    doc.append(", ".join(keywords))

    # with doc.create(Itemize()) as itemize:
    #     for keyword in keywords:
    #         itemize.add_item(keyword)

# Add section for keyword combinations and scores
with doc.create(Section(f"Top {TOP_N} Suggestions of Research Directions")):
    doc.append("This section lists possible keyword combinations and their scores.")

    # Assuming you have a dictionary of keyword combinations and their scores

    with doc.create(Itemize()) as itemize:
        for concept_1, concept_2, score in keyword_combinations:
            itemize.add_item(f"{concept_1} & {concept_2}: {round(score, 4)}")

with doc.create(Section("Used Papers")):
    doc.append("The following papers were considered for keyword extraction:")

    all_titles = df["title"].tolist()

    with doc.create(Itemize()) as itemize:
        for title in all_titles:
            itemize.add_item(title)

# Generate PDF
doc.generate_pdf("research_keywords_analysis", clean_tex=True)
