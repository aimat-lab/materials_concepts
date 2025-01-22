import pandas as pd

# set the number of files to split into
num_files = 5

# read the input file into a pandas DataFrame
df = pd.read_csv("data/compressed_works.csv")

# calculate the number of rows per file
rows_per_file = (len(df) + num_files - 1) // num_files

# loop over the number of output files
for file_num in range(num_files):
    # calculate the start and end rows for the current file
    start_row = file_num * rows_per_file
    end_row = (file_num + 1) * rows_per_file

    # get the rows for the current file
    rows = df.iloc[start_row:end_row]

    # write the rows to the current output file
    rows.to_csv(f"data/split/output_{file_num}.csv", index=False)
