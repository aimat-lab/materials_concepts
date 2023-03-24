def inverted_abstract_to_abstract(inverted_abstract):
    if not inverted_abstract:
        return ""

    ab_len = -1
    for _, value in inverted_abstract.items():
        for index in value:
            ab_len = max(ab_len, index)

    abstract = [" "] * (ab_len + 1)
    for key, value in inverted_abstract.items():
        for i in value:
            abstract[i] = key
    return " ".join(abstract)


def extract_concepts(concepts):
    return [(c["display_name"], c["level"], c["score"]) for c in concepts]
