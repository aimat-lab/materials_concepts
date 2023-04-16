import re


def filter_out_ascii(text):
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


def basic_text_cleaning(text):
    text = filter_out_ascii(text)
    text = re.sub(r"\t+", " ", text)
    text = remove_multiple_whitespaces(text)
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


def replace_oxide_numbers(text):
    # replace roman numerals with numbers
    text = text.replace("(I)", "1+")
    text = text.replace("(II)", "2+")
    text = text.replace("(III)", "3+")
    text = text.replace("(IV)", "4+")
    text = text.replace("(V)", "5+")
    text = text.replace("(VI)", "6+")
    text = text.replace("(VII)", "7+")
    text = text.replace("(VIII)", "8+")
    text = text.replace("(IX)", "9+")
    text = text.replace("(X)", "10+")
    return text


PAREN_REGEX = re.compile(r"[\(\)\[\]\|]")


def replace_parenthesis(text):
    text = PAREN_REGEX.sub("", text)
    return text


ABBR_REGEX = re.compile(r"[\s\(][A-Z]+s[\s\)]")


def replace_abbreviations(text):
    # Replaces abbreviations in upper case only if they are followed by an s and surrounded by either parenthesis or whitespace
    text = ABBR_REGEX.sub("", text)
    return text


SUBSUP_REGEX = re.compile(r"\{su[bp]\s*([^}]+)\}")


def chemic_cleaning(text):
    # Replaces sup/sub script: Eu{sup 3+} {sup 5}D{sub 0}-{sup7}F{sub 2} (red)
    text = SUBSUP_REGEX.sub(r"\1", text)
    return text


# 1500 o C/K or 1400oC
TEMP_C_K_REGEX = re.compile(r"\s+\d+(\.\d+)?\s*o?\s*[CK]")


def replace_temperatures(text):
    text = TEMP_C_K_REGEX.sub("", text)
    return text


# 10-20, 100-125, ...
NUMBER_RANGES = re.compile(r"\s(\d+)-(\d+)\s")


def replace_number_ranges(text):
    text = NUMBER_RANGES.sub("", text)
    return text


def clean_abstract(text):
    text = remove_authors(text)
    text = remove_word(text, "abstract")
    text = remove_word(text, "acknowledgements")
    text = remove_intro(text)
    return text


def clean_materials_science_abstract(text):
    text = chemic_cleaning(text)
    text = replace_temperatures(text)  # replace temperatures before number ranges
    text = replace_number_ranges(text)
    text = replace_oxide_numbers(text)
    text = replace_parenthesis(text)
    text = replace_abbreviations(text)
    return text


def prepare(text):
    text = basic_text_cleaning(text)
    text = clean_abstract(text)
    text = clean_materials_science_abstract(text)
    text = remove_multiple_whitespaces(text)
    return text.strip()


def detect_language(text):
    """This can be used to detect if there are any non-english paragraphs in the text."""
    LANGUAGE = {
        "resumo": "pt",
        "autores": "pt",
        "auteurs": "fr",
        "autoren": "de",
    }

    # not a complete list of all occuring languages
    # but the ones that were not filtered out by language detection
    # already will be filtered manually
    for word, lang in LANGUAGE.items():
        if word in text:
            return lang

    return "en"


COMMON_ERRORS = {
    "In",  #        word
    "At",  #        word
    "VIP",  #       abbr
    "C",  #         Celsius
    "K",  #         Kelvin
    "As",  #        word
    "I",  #         Roman numeral
    "II",  #        Roman numeral
    "III",  #       Roman numeral
    "IV",  #        Roman numeral
    "V",  #         Roman numeral || Volt
    "Pa",  #        Pascal
    "UV",  #        Ultraviolet
    "U",  #         ?
    "K1",  #         Thermal conductivity
}


# => instead of deleting: calculate embeddings?
def filter_common_errors(formulas: set) -> set:
    return formulas - COMMON_ERRORS
