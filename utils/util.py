from pathlib import Path
from typing import List
import difflib
from typing import Any, Union
from src.utils.rules import *

def list_files_top_level(folder: str | Path) -> List[str]:
    folder = Path(folder)
    return [f.name for f in folder.iterdir() if f.is_file()]


import re
import ast
from typing import List, Pattern


# ------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------
def contains_any(keywords: List[str], text: str) -> bool:
    text = text.lower()
    return any(k in text for k in keywords)


def is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


# ------------------------------------------------------------
# Persian text detection
# ------------------------------------------------------------
_PERSIAN_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)

def has_persian_chars(text: str) -> bool:
    return bool(text and _PERSIAN_RE.search(text))


# ------------------------------------------------------------
# Reference value detection
# ------------------------------------------------------------
_REFERENCE_RE: Pattern[str] = re.compile(r"""
    (?:
        Deficient|Insufficient|Sufficient|Potential Intoxication|
        Negative|Positive|Normal|Prediabets|Diabets|
        Low risk|High risk|No risk|
        NCEP guidelines for adults|use only for
    )
    |
    (
        [^\d<>\-]*?
        (
            \d+(\.\d+)?\s*-\s*\d+(\.\d+)?
            |
            \d+(\.\d+)?\s+to\s+\d+(\.\d+)?
            |
            <\s*\d+(\.\d+)?
            |
            >\s*\d+(\.\d+)?
            |
            <=\s*\d+(\.\d+)?
            |
            >=\s*\d+(\.\d+)?
        )
        .*?
    )
    \s*$
""", re.I | re.VERBOSE)

def is_reference_value(text: str) -> bool:
    return contains_any(TEST_REFERENCES, text) or bool(_REFERENCE_RE.search(text))


# ------------------------------------------------------------
# OCR row file reader
# ------------------------------------------------------------
def read_ocr_rows(filename: str) -> List[List[str]]:
    with open(filename, encoding="utf-8") as f:
        return [ast.literal_eval(line.strip()) for line in f]






def normalize_value(v: Any) -> Any:
    if isinstance(v, str):
        return (
            v.lower()
            .strip("*")
            .strip("h")
            .strip("l")
            .strip("high")
            .strip("low")
            .strip()
        )
    return v


def to_numeric(value: Union[int, float, str]) -> Union[int, float, str]:
    value = normalize_value(value)

    if isinstance(value, (int, float)):
        return value
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def string_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def most_similar_item(target: str, items: List[str]):
    items = [i.lower() for i in items]
    target = target.lower()
    matches = difflib.get_close_matches(target, items, n=1, cutoff=0.0)
    return matches[0], difflib.SequenceMatcher(None, target, matches[0]).ratio()

def list_similarity(target: str, items: List[str]) -> float:
    _, score = most_similar_item(target, items)
    return score
