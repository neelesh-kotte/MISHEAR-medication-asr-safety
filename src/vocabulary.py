"""Read a normalized medication vocabulary and rank sound-alike candidates.

These scores are transparent string/phonetic screening features. They are not
clinical risk estimates and do not reproduce the FDA POCA algorithm.
"""

from __future__ import annotations

import argparse
import csv
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class MedicationName:
    name: str
    rxcui: str = ""
    ingredient_rxcui: str = ""
    tty: str = ""
    source: str = ""


@dataclass(frozen=True)
class CandidatePair:
    name_a: str
    name_b: str
    ingredient_a: str
    ingredient_b: str
    jaro_winkler: float
    soundex_a: str
    soundex_b: str
    phonetic_similarity: float
    candidate_score: float
    high_similarity_candidate: bool


def normalize_name(text: str) -> str:
    """Lowercase a medication name and remove punctuation/spacing."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return "".join(ch for ch in ascii_text.lower() if ch.isalpha())


def jaro_winkler(left: str, right: str, prefix_scale: float = 0.1) -> float:
    """Return Jaro-Winkler similarity in [0, 1], with no third-party dependency."""
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0

    match_distance = max(len(left), len(right)) // 2 - 1
    match_distance = max(match_distance, 0)
    left_matches = [False] * len(left)
    right_matches = [False] * len(right)
    matches = 0
    for i, char in enumerate(left):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len(right))
        for j in range(start, end):
            if not right_matches[j] and char == right[j]:
                left_matches[i] = right_matches[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0

    left_ordered = [left[i] for i, found in enumerate(left_matches) if found]
    right_ordered = [right[i] for i, found in enumerate(right_matches) if found]
    transpositions = sum(a != b for a, b in zip(left_ordered, right_ordered)) / 2
    jaro = (matches / len(left) + matches / len(right) + (matches - transpositions) / matches) / 3
    prefix = 0
    for a, b in zip(left, right):
        if a != b or prefix == 4:
            break
        prefix += 1
    return jaro + prefix * prefix_scale * (1 - jaro)


def soundex(text: str) -> str:
    """Return a simple English Soundex code; this is a coarse candidate signal."""
    letters = normalize_name(text).upper()
    if not letters:
        return ""
    groups = {
        **dict.fromkeys("BFPV", "1"),
        **dict.fromkeys("CGJKQSXZ", "2"),
        **dict.fromkeys("DT", "3"),
        **dict.fromkeys("L", "4"),
        **dict.fromkeys("MN", "5"),
        **dict.fromkeys("R", "6"),
    }
    result = [letters[0]]
    previous = groups.get(letters[0], "")
    for char in letters[1:]:
        digit = groups.get(char, "")
        if digit and digit != previous:
            result.append(digit)
        previous = digit
    return ("".join(result) + "000")[:4]


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def phonetic_similarity(left: str, right: str) -> float:
    """Compare Soundex codes by normalized edit distance (heuristic only)."""
    a, b = soundex(left), soundex(right)
    return 1.0 - _edit_distance(a, b) / max(len(a), len(b), 1)


def _first(row: Mapping[str, str], names: Sequence[str]) -> str:
    lowered = {key.strip().lower(): (value or "").strip() for key, value in row.items()}
    for name in names:
        if lowered.get(name):
            return lowered[name]
    return ""


def load_vocabulary_csv(path: str | Path) -> list[MedicationName]:
    """Read a CSV with name/STR, RxCUI, ingredient RxCUI, and optional TTY.

    Accepted header aliases include ``name``/``STR``, ``rxcui``/``RXCUI``,
    and ``ingredient_rxcui``/``ingredient_id``. Ingredient identity should
    point to the normalized IN concept so brand/generic variants stay grouped.
    """
    output: list[MedicationName] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            name = _first(row, ("name", "str", "medication_name"))
            if not name:
                continue
            output.append(MedicationName(
                name=name,
                rxcui=_first(row, ("rxcui", "rx_cui")),
                ingredient_rxcui=_first(row, ("ingredient_rxcui", "ingredient_id", "ingredient")),
                tty=_first(row, ("tty", "term_type")),
                source=_first(row, ("source", "sab")),
            ))
    return output


def load_rxnorm_conso(path: str | Path, term_types: Iterable[str] = ("IN", "MIN")) -> list[MedicationName]:
    """Read selected rows from RxNorm RXNCONSO.RRF (pipe-delimited, no header).

    RXNCONSO contains names/identifiers but does not by itself resolve every
    term to its active ingredient. The returned ingredient_rxcui is populated
    only for TTY=IN; map SCD/SBD/SY variants to IN using RXNREL or a curated
    ingredient mapping before pair generation.
    """
    allowed = set(term_types)
    output: list[MedicationName] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("|")
            if len(fields) < 15 or fields[1] != "ENG" or fields[12] not in allowed:
                continue
            rxcui, tty, name = fields[0], fields[12], fields[14]
            output.append(MedicationName(
                name=name,
                rxcui=rxcui,
                ingredient_rxcui=rxcui if tty == "IN" else "",
                tty=tty,
                source=fields[11],
            ))
    return output


def generate_candidate_pairs(
    vocabulary: Sequence[MedicationName],
    *,
    min_jaro_winkler: float = 0.72,
    min_phonetic_similarity: float = 0.66,
) -> list[CandidatePair]:
    """Rank distinct-ingredient name pairs meeting either similarity threshold.

    Thresholds are pilot defaults only and must be selected/documented before
    the main benchmark. Missing ingredient IDs are treated as distinct entries;
    production data should have ingredient IDs resolved to avoid leakage.
    """
    candidates: list[CandidatePair] = []
    for index, item_a in enumerate(vocabulary):
        for item_b in vocabulary[index + 1:]:
            if item_a.ingredient_rxcui and item_a.ingredient_rxcui == item_b.ingredient_rxcui:
                continue
            a, b = normalize_name(item_a.name), normalize_name(item_b.name)
            if not a or not b or a == b:
                continue
            jw = jaro_winkler(a, b)
            p_sim = phonetic_similarity(item_a.name, item_b.name)
            flagged = jw >= min_jaro_winkler or p_sim >= min_phonetic_similarity
            if flagged:
                candidates.append(CandidatePair(
                    name_a=item_a.name,
                    name_b=item_b.name,
                    ingredient_a=item_a.ingredient_rxcui,
                    ingredient_b=item_b.ingredient_rxcui,
                    jaro_winkler=round(jw, 4),
                    soundex_a=soundex(item_a.name),
                    soundex_b=soundex(item_b.name),
                    phonetic_similarity=round(p_sim, 4),
                    candidate_score=round((jw + p_sim) / 2, 4),
                    high_similarity_candidate=flagged,
                ))
    return sorted(candidates, key=lambda pair: pair.candidate_score, reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="CSV vocabulary file with name and ingredient ID columns")
    args = parser.parse_args()
    rows = load_vocabulary_csv(args.csv_path)
    for pair in generate_candidate_pairs(rows):
        print("\t".join(str(asdict(pair)[key]) for key in CandidatePair.__dataclass_fields__))


if __name__ == "__main__":
    main()
