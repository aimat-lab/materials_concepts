from pathlib import Path
import pandas as pd

data = Path("report/abstracts")


def read(filename) -> pd.DataFrame:
    links = []
    titles = []
    authors = []
    abstracts = []
    lines = [line.replace("\n", "") for line in open(filename, "r").readlines()]
    for lineidx, line in enumerate(lines[:-4]):
        if len(line.split()) == 0:
            links.append(lines[lineidx + 1])
            titles.append(lines[lineidx + 2])
            authors.append(lines[lineidx + 3])
            abstracts.append(lines[lineidx + 4])
    return pd.DataFrame(
        columns=["link", "title", "author", "abstract"],
        data=zip(links, titles, authors, abstracts),
    )


all_dfs = []
for txt_file in data.glob("*.txt"):
    print(txt_file.name)
    df = read(txt_file)
    df["display_name"] = df["title"]
    df["source"] = pd.Series([txt_file.name] * len(df))
    all_dfs.append(df)

works = pd.concat(all_dfs)
works["id"] = works.reset_index().index
works.to_csv("report/analysis_works.csv", index=False)
