import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LinearRegression


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

# Extract the years and counts for regression
X = yearly_counts.index.values.reshape(-1, 1)  # Years as independent variable
y = yearly_counts.values  # Counts as dependent variable

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Degree of the polynomial. You can adjust based on your data.
degree = 3

# Create a polynomial regression model
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Fit the model
polyreg.fit(X, y)

# Predict the values for 2023 and 2024
predicted_counts = polyreg.predict(np.array([[2023], [2024]]))


# Add the predicted counts to the original yearly_counts Series
yearly_counts[2023] = predicted_counts[0]
yearly_counts[2024] = predicted_counts[1]


# Define the calculate_period function to calculate 5-year period
def calculate_period(year):
    base_year = (year // 5) * 5
    return f"{base_year}-{base_year + 4}"


# Apply the function to the index of yearly_counts
yearly_counts.index = yearly_counts.index.map(calculate_period)

# Group by the new 5-year period index and calculate the mean
average_counts = yearly_counts.groupby(level=0).mean()

# Adjust figure size
plt.figure(figsize=(10, 7))

# Plot the actual data points with solid lines
# 1. Plot data up to second-to-last point with a solid line
plt.plot(
    average_counts.index[:-1],
    average_counts.values[:-1],
    marker="o",
    linestyle="-",
    color="blue",
)

# 2. Plot just the last two points with a dashed line
plt.plot(
    average_counts.index[-2:],
    average_counts.values[-2:],
    marker="o",
    linestyle="--",
    color="blue",
)

plt.ylabel("Average Number of Papers")
plt.xlabel("5 Year Period")
plt.xticks(rotation=45)
# ax.set_xticks(range(len(average_counts.index.unique())))
# ax.set_xticklabels(average_counts.index.unique(), rotation=45)
plt.title(
    "Average Number of Papers per 5 Year Period",
    fontsize=20,
)
plt.show()
