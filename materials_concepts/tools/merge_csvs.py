import pandas as pd

# List of input CSV files
file_names = [f"data/split/elements_output_{i}.csv" for i in range(5)]

# Read each CSV file and store them in a list of DataFrames
dataframes = [pd.read_csv(file_name) for file_name in file_names]

# Concatenate all the DataFrames into a single DataFrame
merged_df = pd.concat(dataframes, ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv("data/merged_elements_output.csv", index=False)
