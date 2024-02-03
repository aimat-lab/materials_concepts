from enum import Enum

from materials_concepts.dataset.preparation.chem.elements import elements


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


class Tokenizer:
    def __init__(self, text) -> None:
        self.text = text
        self.pos = 0
        self.tokens: list[Token] = []

    def _advance(self, n=1):
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

    def tokenize(self) -> list[Token]:
        while not self.is_finished():
            # 2 letter elements
            if self.current + self.peek(1) in elements:
                self.tokens.append(
                    Token(self.current + self.peek(1), TokenType.ELEMENT)
                )
                self._advance(2)
                self.merge_element()
                continue

            # 1 letter elements
            if self.current in elements:
                self.tokens.append(Token(self.current, TokenType.ELEMENT))
                self._advance()
                self.merge_element()
                continue

            # detect numbers
            if self.current.isdigit():
                num = [self.current]
                self._advance()

                while self.current.isdigit():
                    num.append(self.current)
                    self._advance()

                if self.current == "." and self.peek(1).isdigit():
                    num.append(self.current)
                    self._advance()

                    while self.current.isdigit():
                        num.append(self.current)
                        self._advance()

                if self.current == "+":
                    num.append(self.current)
                    self._advance()

                self.tokens.append(Token("".join(num), TokenType.NUMBER))
                self.merge_number()
                continue

            # detect symbols
            if self.current in ("+-/"):
                self.tokens.append(Token(self.current, TokenType.INFER_TYPE))
                self._advance()
                continue

            # detect whitespaces
            if self.current == " ":
                self.tokens.append(Token(self.current, TokenType.WHITESPACE))
                self._advance()
                self.merge_whitespace()
                continue

            # detect comma or dot (end of sentence)
            if self.current == "." or self.current == ",":
                self.tokens.append(Token(self.current, TokenType.DELIM))
                self._advance()
                continue

            self.tokens.append(Token(self.current, TokenType.CHAR))
            self.merge_char()
            self._advance()

        return self.tokens

    def merge_char(self):
        if len(self.tokens) > 1 and (
            self.tokens[-2].type == TokenType.CHAR
            or self.tokens[-2].type == TokenType.ELEMENT
        ):
            self.tokens[-2].value += self.tokens[-1].value
            self.tokens[-2].type = TokenType.CHAR
            self.tokens.pop()

    def merge_element(self):  # currently: Number + Element =/> Element
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


def filter_tokens(tokens: list[Token], type: TokenType):
    return {token.value for token in tokens if token.type == type}


def filter_element_tokens(tokens):
    return filter_tokens(tokens, TokenType.ELEMENT)


def get_tokens(text) -> list[Token]:
    return Tokenizer(text).tokenize()


def merge_tokens(tokens: list[Token]):
    return "".join(token.value for token in tokens)


def get_elements(text) -> set:
    tokens = Tokenizer(text).tokenize()
    return filter_tokens(tokens, TokenType.ELEMENT)
