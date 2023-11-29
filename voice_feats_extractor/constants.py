PHONEME_LIST = [
    "pau",
    "A",
    "E",
    "I",
    "N",
    "O",
    "U",
    "a",
    "b",
    "by",
    "ch",
    "cl",
    "d",
    "dy",
    "e",
    "f",
    "g",
    "gw",
    "gy",
    "h",
    "hy",
    "i",
    "j",
    "k",
    "kw",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "p",
    "py",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "v",
    "w",
    "y",
    "z",
]

UNVOICED_PHONEME_LIST = ["A", "I", "U", "E", "O", "cl", "pau"]
VOICED_VOWEL_LIST = ["a", "i", "u", "e", "o"]
MORA_PHONEME_LIST = [*VOICED_VOWEL_LIST, "N", *UNVOICED_PHONEME_LIST]

ACCENT_SYMBOL_LIST = ["[", "]", "#", "?", "_"]
EXTRA_SYMBOL_LIST = ["^", "$", *ACCENT_SYMBOL_LIST]