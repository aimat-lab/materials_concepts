import re
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

between_brackets = re.compile(r"\[([^]]+)\]")  # extract text between square brackets

EL_FILTER = ["As", "At", "In"]  # these elements also occur in concepts

elements = {
    "H": "hydrogen",
    "He": "helium",
    "Li": "lithium",
    "Be": "beryllium",
    "B": "boron",
    "C": "carbon",
    "N": "nitrogen",
    "O": "oxygen",
    "F": "fluorine",
    "Ne": "neon",
    "Na": "sodium",
    "Mg": "magnesium",
    "Al": "aluminium",
    "Si": "silicon",
    "S": "sulfur",
    "Cl": "chlorine",
    "Ar": "argon",
    "K": "potassium",
    "Ca": "calcium",
    "Sc": "scandium",
    "Ti": "titanium",
    "V": "vanadium",
    "Cr": "chromium",
    "Mn": "manganese",
    "Fe": "iron",
    "Co": "cobalt",
    "Ni": "nickel",
    "Cu": "copper",
    "Zn": "zinc",
    "Ga": "gallium",
    "Ge": "germanium",
    "As": "arsenic",
    "Se": "selenium",
    "Br": "bromine",
    "Kr": "krypton",
    "Rb": "rubidium",
    "Sr": "strontium",
    "Y": "yttrium",
    "Zr": "zirconium",
    "Nb": "niobium",
    "Mo": "molybdenum",
    "Tc": "technetium",
    "Ru": "ruthenium",
    "Rh": "rhodium",
    "Pd": "palladium",
    "Ag": "silver",
    "Cd": "cadmium",
    "In": "indium",
    "Sn": "tin",
    "Sb": "antimony",
    "Te": "tellurium",
    "I": "iodine",
    "Xe": "xenon",
    "Cs": "caesium",
    "Ba": "barium",
    "La": "lanthanum",
    "Ce": "cerium",
    "Pr": "praseodymium",
    "Nd": "neodymium",
    "Pm": "promethium",
    "Sm": "samarium",
    "Eu": "europium",
    "Gd": "gadolinium",
    "Tb": "terbium",
    "Dy": "dysprosium",
    "Ho": "holmium",
    "Er": "erbium",
    "Tm": "thulium",
    "Yb": "ytterbium",
    "Lu": "lutetium",
    "Hf": "hafnium",
    "Ta": "tantalum",
    "W": "tungsten",
    "Re": "rhenium",
    "Os": "osmium",
    "Ir": "iridium",
    "Pt": "platinum",
    "Au": "gold",
    "Hg": "mercury",
    "Tl": "thallium",
    "Pb": "lead",
    "Bi": "bismuth",
    "Po": "polonium",
    "At": "astatine",
    "Rn": "radon",
    "Fr": "francium",
    "Ra": "radium",
    "Ac": "actinium",
    "Th": "thorium",
    "Pa": "protactinium",
    "U": "uranium",
    "Np": "neptunium",
    "Pu": "plutonium",
    "Am": "americium",
    "Cm": "curium",
    "Bk": "berkelium",
    "Cf": "californium",
    "Es": "einsteinium",
    "Fm": "fermium",
    "Md": "mendelevium",
    "No": "nobelium",
    "Lr": "lawrencium",
    "Rf": "rutherfordium",
    "Db": "dubnium",
    "Sg": "seaborgium",
    "Bh": "bohrium",
    "Hs": "hassium",
    "Mt": "meitnerium",
    "Ds": "darmstadtium",
    "Rg": "roentgenium",
    "Cn": "copernicium",
    "Nh": "nihonium",
    "Fl": "flerovium",
    "Mc": "moscovium",
    "Lv": "livermorium",
    "Ts": "tennessine",
    "Og": "oganesson",
}


def keep_first_valid(text):
    return [line.strip() for line in text.split("\n") if line.strip()][0]


def add_last_bracket(text: str):
    if "]" not in text:
        return text[: text.rindex(",")] + "]"  # add last bracket after last comma
    else:
        return text


def trim_to_first_close_bracket(text: str):
    return text[: text.index("]") + 1]  # trim to first close bracket


def _eval(text):
    return [
        concept.strip()
        for concept in text.replace("'", "").split(",")
        if concept.strip()
    ]


def process_work(text):
    try:
        text = keep_first_valid(text)  # keep first valid line
        text = add_last_bracket(text)  # if bracket missing
        text = trim_to_first_close_bracket(text)
        text = between_brackets.search(text).group(1)  # extract text between brackets
        return _eval(text)
    except Exception as e:
        print(e, "in", text)
        return []


def clean_concepts(concepts):
    concepts = [
        concept.lower()
        .replace("/", " ")
        .replace("-", " ")
        .replace("aluminum", "aluminium")
        for concept in concepts
    ]

    return concepts


def substitute_single_elements(row):
    concepts = row.llama_concepts
    els = row.elements.split(",")

    if len(els) == 0:
        return concepts

    search_concepts = [" " + concept + " " for concept in concepts]

    concepts = []

    for concept in search_concepts:
        temp = concept

        for el in elements:
            if el not in els and len(el) == 1:
                continue

            if el in EL_FILTER:
                continue

            temp = temp.replace(
                " " + el.lower() + " ", " " + elements[el] + " "
            )  # ensures to replace only whole words

        concepts.append(temp.strip())

    return sorted(set(concepts))


if __name__ == "__main__":
    import os

    WORK_FILE = "report/analysis_works.elements.works.csv"
    OUT_FILE = "report/fixed_concepts.csv"

    df = pd.read_csv("report/concepts.csv")

    df.concepts = df.concepts.progress_apply(process_work)
    df.concepts = df.concepts.progress_apply(clean_concepts)

    # merge with original data

    df = df.rename(columns={"concepts": "llama_concepts"})

    original = pd.read_csv(WORK_FILE)

    m = original.merge(df, on="id")

    # substitute single elements

    m["elements"] = m["elements"].fillna("")
    m["llama_concepts"] = m.progress_apply(substitute_single_elements, axis=1)

    m.to_csv(OUT_FILE, index=False)
