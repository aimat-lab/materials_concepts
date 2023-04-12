import preprocessing

elements = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

from enum import Enum


class TokenType(Enum):
    INFER_TYPE = 0  # sentinal value
    ELEMENT = 1
    NUMBER = 2
    DELIM = 3
    HYPHEN = 4
    PLUS = 5
    WHITESPACE = 8
    CHAR = 9
    SLASH = 11


class Automaton:
    def __init__(self, text) -> None:
        self.text = text
        self.pos = 0
        self.tokens = []

    def advance(self, n=1):
        self.pos += n

    @property
    def current(self):
        if self.pos >= len(self.text):
            return ""  # simulate EOF

        return self.text[self.pos]

    def peek(self, n):
        if (
            self.pos + n >= len(self.text) or self.pos + n < 0
        ):  # out of bounds: simulate EOF
            return ""

        return self.text[self.pos + n]

    def is_finished(self):
        return self.pos >= len(self.text)

    def tokenize(self):
        while not self.is_finished():
            # 2 letter elements
            if self.current + self.peek(1) in elements:
                self.tokens.append(
                    Token(self.current + self.peek(1), TokenType.ELEMENT)
                )
                self.advance(2)
                self.merge_element()
                continue

            # 1 letter elements
            if self.current in elements:
                self.tokens.append(Token(self.current, TokenType.ELEMENT))
                self.advance()
                self.merge_element()
                continue

            # detect numbers
            if self.current.isdigit():
                num = [self.current]
                self.advance()

                while self.current.isdigit():
                    num.append(self.current)
                    self.advance()

                if self.current == "." and self.peek(1).isdigit():
                    num.append(self.current)
                    self.advance()

                    while self.current.isdigit():
                        num.append(self.current)
                        self.advance()

                if self.current == "+" or self.current == "-":
                    num.append(self.current)
                    self.advance()

                self.tokens.append(Token("".join(num), TokenType.NUMBER))
                self.merge_number()
                continue

            # detect symbols
            if self.current in ("+/-"):
                self.tokens.append(Token(self.current, TokenType.INFER_TYPE))
                self.advance()
                continue

            # detect whitespaces
            if self.current == " ":
                self.tokens.append(Token(self.current, TokenType.WHITESPACE))
                self.advance()
                self.merge_whitespace()
                continue

            # detect comma or dot (end of sentence)
            if self.current == "." or self.current == ",":
                self.tokens.append(Token(self.current, TokenType.DELIM))
                self.advance()
                continue

            self.tokens.append(Token(self.current, TokenType.CHAR))

            # merge if both are chars or before was element
            self.merge_char()
            self.advance()

        return self.tokens

    def merge_char(self):
        if len(self.tokens) > 1 and (
            self.tokens[-2].type == TokenType.CHAR
            or self.tokens[-2].type == TokenType.ELEMENT
        ):
            self.tokens[-2].value += self.tokens[-1].value
            self.tokens[-2].type = TokenType.CHAR
            self.tokens.pop()

    def merge_element(self):
        # before was element
        if len(self.tokens) > 1 and self.tokens[-2].type == TokenType.ELEMENT:
            self.tokens[-2].value += self.tokens[-1].value
            self.tokens.pop()

        # before was (whitespace or hyphen) and before that was element
        elif (
            len(self.tokens) > 2
            and (
                self.tokens[-2].type == TokenType.WHITESPACE
                or self.tokens[-2].type == TokenType.HYPHEN
            )
            and self.tokens[-3].type == TokenType.ELEMENT
        ):
            self.tokens[-3].value += self.tokens[-1].value
            self.tokens.pop()  # remove element (now stored in previous)
            self.tokens.pop()  # remove whitespace

        # before was char
        elif len(self.tokens) > 1 and self.tokens[-2].type == TokenType.CHAR:
            self.tokens[-2].value += self.tokens[-1].value
            self.tokens.pop()

        # TODO: Does this ever happen?
        # elif ( # before was number
        #     len(self.tokens) > 1
        #     and self.tokens[-2].type == TokenType.NUMBER
        # ):
        #     self.tokens[-2].value += self.tokens[-1].value
        #     self.tokens[-2].type = TokenType.ELEMENT
        #     self.tokens.pop()

    def merge_number(self):
        # before was slash and before that was number
        if (
            len(self.tokens) > 2
            and self.tokens[-2].type == TokenType.SLASH
            and self.tokens[-3].type == TokenType.NUMBER
        ):
            self.tokens[-3].value += self.tokens[-2].value + self.tokens[-1].value
            self.tokens.pop()

        # before was element
        elif len(self.tokens) > 1 and self.tokens[-2].type == TokenType.ELEMENT:
            self.tokens[-2].value += self.tokens[-1].value
            self.tokens.pop()

        # before was whitespace and before that was element
        elif (
            len(self.tokens) > 2
            and self.tokens[-2].type == TokenType.WHITESPACE
            and self.tokens[-3].type == TokenType.ELEMENT
        ):
            self.tokens[-3].value += self.tokens[-1].value
            self.tokens.pop()

    def merge_whitespace(self):
        # only need 1 whitespace
        if len(self.tokens) > 1 and self.tokens[-2].type == TokenType.WHITESPACE:
            self.tokens.pop()


class Token:
    def __init__(self, value, type: TokenType = TokenType.INFER_TYPE) -> None:
        self.value = value
        self.type = Token.get_type(value) if type == TokenType.INFER_TYPE else type

    @staticmethod
    def get_type(value):
        if value in elements:
            return TokenType.ELEMENT
        elif value == "+":
            return TokenType.PLUS
        elif value == "-":
            return TokenType.HYPHEN
        elif value == "/":
            return TokenType.SLASH
        elif value == " ":
            return TokenType.WHITESPACE
        elif value == ",":
            return TokenType.DELIM
        else:
            return TokenType.CHAR

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"Token({self.value}, {self.type})"


def multiline_input():
    text = ""
    while True:
        line = input()
        if line == "":
            break
        text += line + ""
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


def get_elements(tokens):
    return {token.value for token in tokens if token.type == TokenType.ELEMENT}


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    import re

    tqdm.pandas()

    df = pd.read_csv("data/subset.csv")

    zeroth = lambda text: preprocessing.clean_abstract(text)

    def clean(text):
        text = re.sub(r"o\s*C", "", text)
        text = re.sub(r"\d+\s*C", "", text)
        text = re.sub(r"{sub ([0-9]+)}", r"\1", text)
        return text

    cleaning = lambda text: clean(text)
    first = lambda text: replace_oxide_numbers(text)
    second = lambda text: (
        text.replace("(", " ")
        .replace(")", " ")
        .replace("[", " ")
        .replace("]", " ")
        .replace("|", " ")
    )

    third = lambda text: get_elements(Automaton(text).tokenize())

    df["text"] = df["abstract"].copy()
    df["pre"] = (
        df.text.progress_apply(cleaning)
        .progress_apply(zeroth)
        .progress_apply(first)
        .progress_apply(second)
    )

    df.text = df.pre.progress_apply(third)
    df["tokens"] = df.pre.progress_apply(lambda x: Automaton(x).tokenize())

    print(df[["id", "text"]])

COMMON_NO_ELEMENTS = [
    "In",  #        word
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
]
# => instead of deleting: calculate embeddings?
