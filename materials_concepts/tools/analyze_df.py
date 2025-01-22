import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("data/materials-science.elements.works.csv")

df.publication_date = pd.to_datetime(df.publication_date)
df["pub_year"] = df.publication_date.dt.year
sns.histplot(data=df, x="pub_year", kde=True)
