import re


def replace_equation_symbols(text):
    symbols = ["⩾", "⩽", "×", ">", "<", "=", "±", "%", "〈", "〉", "/", "∼"]
    for symbol in symbols:
        text = text.replace(symbol, " ")
    return text


def filter_out_ascii(text):
    text = replace_equation_symbols(text)  # keep a whitespace here
    return "".join(
        char for char in text if ord(char) < 128
    )  # remove non-ascii (no whitespace)


def remove_multiple_whitespaces(text):
    text = re.sub(r"\s+", " ", text)
    return text


def remove_parenthesis(text):
    """TODO: test if this can be done after parsing chemical formulas"""
    text = re.sub(r"[\(\)]", "", text)
    return text


def remove_text_enumerations(text):  # RISKY: Dont remove element numbers
    # with roman numerals
    text = re.sub(r"\s+(i|ii|iii|iv|v)[\.\)]", "", text)
    # with a, b, c, d
    text = re.sub(r"\s+[a-e][\.\)]", "", text)

    return text


def clean_text(text):
    text = filter_out_ascii(text)
    # text = re.sub(" +", " ", text) ===> TEST
    text = re.sub(r"\t+", " ", text)
    text = remove_multiple_whitespaces(text)
    # text = remove_parenthesis(text) ===> TEST
    text = remove_text_enumerations(text)
    text = text.strip()

    return text


def remove_authors(text):
    text = re.sub(r"authors.*abstract", "", text, flags=re.IGNORECASE)
    return text


def remove_acknowledgements(text):
    text = re.sub(r"acknowledgements", "", text, flags=re.IGNORECASE)
    return text


def remove_word(text, word):
    text = re.sub(word, "", text, flags=re.IGNORECASE)
    return text


def parse_keywords(keywords):
    comma_count = keywords.count(",")
    semicolon_count = keywords.count(";")
    delimiter = "," if comma_count > semicolon_count else ";"

    keywords = keywords.split(delimiter)

    TO_REMOVE = ".;,:"  # often a dot was found at the end of the last keyword
    stripped_keywords = [keyword.strip(TO_REMOVE) for keyword in keywords]
    return stripped_keywords


def split_keywords(text):
    text = re.split("keywords", text, flags=re.IGNORECASE)
    return text


def remove_intro(text):
    text = re.sub(r"i(\s*)ntroduction.*", "", text, flags=re.IGNORECASE)
    return text


def clean_abstract(text):
    # It's easier to catch elements if capitalized
    text = clean_text(text)
    text = remove_authors(text)
    text = remove_word(text, "abstract")
    text = remove_word(text, "acknowledgements")
    text = remove_intro(text)
    return text.strip()


def language_cleaning(text):
    """TODO: This can be used to detect if there are any non-english paragraphs in the text."""
    LANGUAGE = {
        "resumo": "pt",
        "autores": "pt",
        "auteurs": "fr",
        "autoren": "de",
    }
