## Fetching Works

- Relvant data: Select fields `https://api.openalex.org/works?select=id,doi,display_name,publication_date,is_retracted,is_paratext,concepts,abstract_inverted_index`

- Search for sources ('venues'): `https://api.openalex.org/sources?search=materials%20science`
- Filter works based on venue: `https://api.openalex.org/works?filter=host_venue.id:S37840076`
- Page navigation: `https://api.openalex.org/works?page=2&per-page=200`
- Without cursor: limit 10.000. Else use [cursor paging](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/paging)

## Cleaning

- Filter out works (no title, no abstract, retracted, paratext)
- Invert abstracts

```
def invert_abstract_to_abstract_(invert_abstract):
    invert_abstract = json.loads(invert_abstract)
    ab_len = invert_abstract['IndexLength']

    abstract = [" "]*ab_len
    for key, value in invert_abstract['InvertedIndex'].items():
        for i in value:
            abstract[i] = key
    return " ".join(abstract)
```

- Filter out abstracts: len < 30 and len > 1000 (=> get distribution first)
- Remove: 'abstract', 'authors ....'
- Remove starting introduction
- Keywords?
-
- Keep: [a-zA-Z0-9 \.]+
  - What about mu and lambda?
- Clean multiple spaces

TODO:

- Replace C<sub>15</sub>H<sub>9</sub>
- Is Material Science in concepts?
- Indication of other language present ('resumo', 'auteurs', 'autoren', 'autores')
- Indication of latex code present ('\')
- Replace alone numbers?

- Extract elements from abstract and normalize numbers

```
# Possible cleaning function
def clean_text_(text):
    try:
        text = text.lower()

        text = re.sub('[^a-zA-Z0-9 ]+', ' ', text)
        text = re.sub(' +', ' ', text)
        text = text.strip()

    except:
        text = ""
    return text
```

- Apply keyword extraction to abstract and title (?)
  - Rake: Apply to abstract
  - n-grams: Generate or fetch afterwards
  - Use concepts (filter: level > 1/2, score > 0.3)
  - Large Language Models: Use them directly or filter afterwards (normalize singular/plural)

```
# r = Rake(min_length=2, max_length=5, language="english")
# r.extract_keywords_from_text(df.loc[1]["abstract_inverted_index"])
# keywords = r.get_ranked_phrases()
```

## TODO Process

- [x] Create README with instructions to recreate progress
- [x] Implement cursor fetching
- [x] Work cleaning: Filter out works (no title, no abstract, retracted, paratext, english lang)
- [x] Abstract cleaning: Clean chemical elements
- [ ] Generate cleaned 'list' of all concepts
- [ ] Build graph with histogram edges
- [ ] Implement top performing model from kaggle challange
- [ ] Store model and graph
- [ ] Build API to query prediction service

## TODO Optimization

- [ ] Where to store the data?
- [ ] Data storing for works: How to store concepts (fetched and generated)
- [ ] How to speed up text processing in pandas? Pandas 2.0 or other option to achieve pyarrow backend
- [ ] Dockerize what comes after data fetching
