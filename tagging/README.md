## Annotation Format

Annotation is done via plane .txt file. Abstracts are visually separated by "====..." lines. Work ID is included.
To ease parsing, concepts should appear one per line surrounded by "§" signs:

```
================================================================================

W1980885978

Review of recent studies in magnesium matrix composites. In this paper, recent
progress in magnesium matrix composite technologies is reviewed. The
conventional and new processes for the fabrication of magnesium matrix
composites are summarized. The composite microstructure is subsequently
discussed with respect to grain refinement, reinforcement distribution, and
interfacial characteristics. The mechanical properties of the magnesium matrix
composites are also reported.

§
magnesium matrix composite technology
magnesium matrix composite
matrix composite
conventional fabrication process
new fabrication process
fabrication process
composite microstructure
grain refinement
reinforcement distribution
interfacial characteristic
mechanical property
§

================================================================================
```

## Create Annotation-ready File

`$ python tagging/create_abstracts.py data/materials-science.elements.works.csv tagging/abstracts.txt`

The generated file can be annotated and later be parsed back into a more structured format.

## Parsing Annotations

`$ python tagging/extract_tags.py tagging/abstracts.txt tagging/out.csv`

This .csv file will contain two columns, mapping IDs to concepts (which are stored as comma separated string).
