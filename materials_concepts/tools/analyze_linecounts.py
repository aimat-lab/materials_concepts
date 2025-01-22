import os
import pandas as pd

directory = "./data/materials-science_sources"

file_line_counts = []

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)

    if os.path.isfile(filepath):
        # Open the file and count the number of lines
        with open(filepath, "r") as file:
            line_count = sum(1 for line in file) - 1  # subtract 1 for the header

        file_line_counts.append({"filename": filename[:-4], "line_count": line_count})

# Create a pandas DataFrame from the list
df = pd.DataFrame(file_line_counts)
df = df.rename(columns={"filename": "id"})

df_sources = pd.read_csv("./data/materials-science_sources.csv")
