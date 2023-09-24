import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/table/materials-science.cleaned.works.csv")


df["publication_date"] = pd.to_datetime(df["publication_date"])
df["year"] = df["publication_date"].dt.year

df = df[df["year"] <= 2022]

# Find Min and Max Dates
min_date = df["publication_date"].min()
max_date = df["publication_date"].max()
print("Min Date:", min_date)
print("Max Date:", max_date)

yearly_counts = df.groupby("year").size()


# Define the calculate_period function to calculate 5-year period
def calculate_period(year):
    base_year = (year // 5) * 5
    return f"{base_year}-{base_year + 4}"


# Apply the function to the index of yearly_counts
yearly_counts.index = yearly_counts.index.map(calculate_period)

# Group by the new 5-year period index and calculate the mean
average_counts = yearly_counts.groupby(level=0).mean()
# Adjust figure size
plt.figure(figsize=(10, 6))

# Plot
ax = average_counts.plot(
    kind="line", title="Average Number of Works per Year", marker="o"
)
plt.ylabel("Average Number of Works")
plt.xlabel("5 Year Period")
ax.set_xticks(range(len(average_counts.index.unique())))
ax.set_xticklabels(average_counts.index.unique(), rotation=45)
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()
