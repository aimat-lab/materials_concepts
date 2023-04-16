## Cleaning

- [x] Remove: 'abstract', 'authors ....'
- [x] Remove starting introduction
- [ ] Replace C<sub>15</sub>H<sub>9</sub>
- [ ] If latex present -> substitute

## Preparation

- [ ] Extract elements from abstract and normalize numbers (Investigate common errors (large numbers or number '1') LLMs can be used?)
- [ ] Extract existing keywords
- [ ] Apply keyword extraction to abstract (and title?)
- [ ] Think about n-grams: Generate or fetch afterwards (maybe: try downloading, keep first 5)
- [ ] Use concepts (filter: level > 1/2, score > 0.5)
- [ ] Think about Large Language Models: Filter afterwards (normalize singular/plural)

```
# r = Rake(min_length=2, max_length=5, language="english")
# r.extract_keywords_from_text(df.loc[1]["abstract_inverted_index"])
# keywords = r.get_ranked_phrases()
```
